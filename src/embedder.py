import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


VECTORSTORE_PATH = "vectorstore/faiss_index"

# Using a lightweight but powerful sentence transformer model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Load HuggingFace embeddings model (runs locally, no API key needed)."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


def build_vectorstore(chunks):
    """
    Build a FAISS vectorstore from document chunks and save to disk.
    Returns the vectorstore object.
    """
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save locally for reuse
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    return vectorstore


def load_vectorstore():
    """Load an existing FAISS vectorstore from disk."""
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore
