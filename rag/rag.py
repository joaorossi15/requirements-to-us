from langchain.document_loaders.pdf import PyPDFDirectoryLoader # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations


def load_documents(dir: str):
    doc_loader = PyPDFDirectoryLoader(dir)
    return doc_loader.load()


def split_text(docs: list[Document]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    
    return chunks


def chroma(chunks: list[Document]):
    path = './chroma/'

    if os.path.exists(path):
        shutil.rmtree(path)

    db = Chroma.from_documents(
            chunks, 
            HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'), 
            persist_directory=path)
    
    db.persist()
    print(f'Saved {len(chunks)} chunks to {path}')


def generate_store(path: str):
    documents = load_documents(path)
    chunks = split_text(documents)
    chroma(chunks)

generate_store('/home/joaorossi/us-translator/rag-data')

