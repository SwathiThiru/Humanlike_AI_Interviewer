from embeddings import VectorStore

# initialize once at startup
vetor_store = VectorStore()
# vetor_store.load()  # if pre-built

def build_index(resume_chunks: list[str], jd_chunks: list[str]):
    vetor_store.add(resume_chunks + jd_chunks)
    vetor_store.save()

def retrieve_context(prompt: str, k: int = 5) -> str:
    # return concatenated top-k chunks
    chunks = vetor_store.query(prompt, k)
    print(f"Retrieved {len(chunks)} context chunks")
    return "\n".join(chunks)
