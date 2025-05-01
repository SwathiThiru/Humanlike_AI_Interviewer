from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_rag_index(
    resume_text: str,
    jd_text: str,
    persist_directory: str = './chromadb',
    collection_name: str = 'interview_rag',
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    embed_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
) -> Chroma:
    """
    Build and persist a RAG index from one resume and JD.
    Returns a chroma vectorstore.
    """
    # 1 Wrap the raw text into Langchain Documents with metadata
    docs = [
        Document(page_content=resume_text, metadata={"type": "resume"}),
        Document(page_content=jd_text, metadata={"type": "jd"})
    ]
    
    # 2 Chunk them
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    
    # 3 Embed & index into Chroma
    embedder = HuggingFaceEmbeddings(model_name=embed_model)
    vectordb = Chroma.from_documents(
        documents = chunks,
        embedding = embedder,
        persist_directory = persist_directory,
        collection_name = collection_name
    )
    vectordb.persist()
    return vectordb

def retrieve_chunks(
    query:str,
    vectordb: Chroma, 
    k: int = 4,
) -> List[str]:
    """
    Given a query string and a built Chroma index,
    return the top-k most relevant document chunks.
    """
    results = vectordb.similarity_search(query=query, k=k)
    return [doc.page_content for doc in results]
