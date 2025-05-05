from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight & fast

class VectorStore:
    def __init__(self, dim: int = 384):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []  # to map index → chunk text

    def add(self, chunks: list[str]):
        embs = self.model.encode(chunks, convert_to_tensor=False, normalize_embeddings=True)
        arr = np.vstack(embs).astype("float32")
        self.index.add(arr)
        self.texts.extend(chunks)

    def save(self, path: str = "faiss_index.pkl"):
        faiss.write_index(self.index, "faiss.index")
        with open(path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, path: str = "faiss_index.pkl"):
        self.index = faiss.read_index("faiss.index")
        with open(path, "rb") as f:
            self.texts = pickle.load(f)

    def query(self, query_text: str, k: int = 5) -> list[str]:
        q_emb = self.model.encode([query_text], convert_to_tensor=False, normalize_embeddings=True)
        D, I = self.index.search(np.array(q_emb, dtype="float32"), k)
        # Only return chunks for valid indices
        valid = [i for i in I[0] if 0 <= i < len(self.texts)]
        return [self.texts[i] for i in valid]
