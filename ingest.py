"""Build the FAISS index used by app.py from the policy PDFs in data/.

Mirrors the ingestion steps in rag_chatbot.ipynb so the app can run
without executing the notebook first.
"""
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "thenlper/gte-small"
INDEX_DIR = "faiss_index"


def main():
    raw_docs = []
    for path in sorted(glob.glob("data/Everstorm_*.pdf")):
        raw_docs.extend(PyPDFLoader(path).load())
    print(f"Loaded {len(raw_docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks")

    vectordb = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    vectordb.save_local(INDEX_DIR)
    print(f"Saved {vectordb.index.ntotal} embeddings to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
