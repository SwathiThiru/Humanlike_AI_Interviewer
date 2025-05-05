from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import re


def load_and_clean_pdf(path: str) -> str:
    """
    Load a PDF via LangChain's PyPDFLoader and return cleaned text.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    # Concatenate all page contents
    full_text = " ".join([doc.page_content for doc in documents])
    # Basic cleaning: collapse whitespace
    cleaned = re.sub(r"\s+", " ", full_text).strip()
    return cleaned


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Split cleaned text into chunks for embedding and retrieval.
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
