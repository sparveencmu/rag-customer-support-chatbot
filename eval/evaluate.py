"""Compare generator models on the same retrieval pipeline.

Retrieval (gte-small + FAISS, k=8) and the prompt are identical to app.py;
only the model that writes the answer changes. Any OpenAI-compatible
endpoint works: local Ollama, Groq, OpenAI, etc.

Usage:
    python ingest.py   # once, builds faiss_index/
    python eval/evaluate.py --model gemma3:1b --base-url http://localhost:11434/v1
    GROQ_API_KEY=... python eval/evaluate.py --model llama-3.1-8b-instant \
        --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY
"""
import argparse
import ast
import json
import os
import pathlib
import re

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text())


def load_system_template():
    # Read the prompt from app.py without importing it (importing would start Streamlit).
    tree = ast.parse((ROOT / "app.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SYSTEM_TEMPLATE":
            return ast.literal_eval(node.value)
    raise RuntimeError("SYSTEM_TEMPLATE not found in app.py")


SYSTEM_TEMPLATE = load_system_template()


def normalize(text):
    # LLMs often emit non-breaking spaces and Unicode hyphens; fold them so plain patterns match.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[‐-―−]", "-", text)
    return text.replace("*", "").lower()


def check(answer, item):
    text = normalize(answer)
    missing = [p for p in item["must_include"] if not re.search(p, text)]
    forbidden = [p for p in item["must_not_include"] if re.search(p, text)]
    return not missing and not forbidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    vectordb = FAISS.load_local(
        str(ROOT / "faiss_index"),
        HuggingFaceEmbeddings(model_name="thenlper/gte-small"),
        allow_dangerous_deserialization=True,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": args.k})
    prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
    api_key = os.environ[args.api_key_env] if args.api_key_env else "ollama"
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    results = []
    for item in QUESTIONS:
        docs = retriever.invoke(item["question"])
        context = "\n\n".join(d.page_content for d in docs)
        norm = lambda t: " ".join(t.split())
        retrieved = None if item["evidence"] is None else norm(item["evidence"]) in norm(context)
        content = prompt.format(context=context, question=item["question"])
        reply = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
        )
        answer = reply.choices[0].message.content.strip()
        passed = check(answer, item)
        results.append({"id": item["id"], "question": item["question"],
                        "expected": item["expected"], "answer": answer, "retrieved": retrieved, "auto_pass": passed})
        print(f"[{'PASS' if passed else 'FAIL'}] {item['id']}  (evidence retrieved: {retrieved})")

    out = ROOT / "eval" / "results" / f"{args.model.replace(':', '_').replace('/', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    score = sum(r["auto_pass"] for r in results)
    print(f"\n{args.model}: {score}/{len(results)} passed automatic checks -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
