from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil


def load_documents(dir_path: str):
    doc_loader = PyPDFDirectoryLoader(dir_path)
    return doc_loader.load()


def split_text(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            ". ",
            ".",
        ],
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )

    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    return chunks


def chroma(chunks: list[Document], path: str):
    abs_path = os.path.abspath(path)

    if os.path.exists(abs_path):
        shutil.rmtree(abs_path, ignore_errors=True)

    os.makedirs(abs_path, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=abs_path,
    )

    db.persist()
    print(f"Saved {len(chunks)} chunks to {abs_path}")


def generate_store(path: str, chroma_path: str):
    documents = load_documents(path)
    chunks = split_text(documents)
    chroma(chunks, chroma_path)
