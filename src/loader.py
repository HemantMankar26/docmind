import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_chunk_pdfs(upload_dir: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Load all PDFs from a directory and split into chunks.
    Returns a list of LangChain Document objects.
    """
    all_docs = []

    pdf_files = [f for f in os.listdir(upload_dir) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError("No PDF files found in the upload directory.")

    for pdf_file in pdf_files:
        path = os.path.join(upload_dir, pdf_file)
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(all_docs)
    return chunks
