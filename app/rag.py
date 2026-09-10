"""Document ingestion and vector store helpers used by the upload/chat routes."""

import secrets

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter


def load_document(path, ext):
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def load_and_split(path, ext, chunk_size=1000, chunk_overlap=100):
    documents = load_document(path, ext)
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def build_vectorstore(chunks, embedding_model, session_id):
    # In-memory (no persist_directory): each upload gets its own uniquely-named
    # collection. langchain_chroma.Chroma defaults collection_name to the fixed
    # string "langchain" — without overriding it here, every upload from every
    # session would land in that same shared collection instead of a fresh,
    # isolated one.
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=f"session_{session_id}_{secrets.token_hex(4)}",
        collection_metadata={"hnsw:space": "cosine"},
    )
