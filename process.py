import os
import shutil
import atexit
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def load_and_embed_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    persist_dir = "vector_db"  # consistent directory for production use

    # Optional: remove old data if any
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()

    # Optional cleanup on exit
    atexit.register(lambda: shutil.rmtree(persist_dir, ignore_errors=True))

    return vectordb