import threading
from typing import List
import faiss
import numpy as np
import os


class FaissManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, dim=1024, index_path="storage/faiss_index.index"):
        if not hasattr(self, "initialized"):  # Ensure __init__ runs only once
            self.index_path = index_path
            self.dim = dim
            self._initialize_index()
            self.initialized = True  # Mark as initialized

    def _initialize_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlat(self.dim)
            self._save_index()

    def add_vectors(self, vectors: np.ndarray) -> List[int]:
        if not self.index.is_trained:
            raise ValueError("FAISS index is not trained yet.")

        new_ids = np.arange(self.index.ntotal, self.index.ntotal + vectors.shape[0])
        self.index.add(vectors)
        self._save_index()
        return new_ids.tolist()

    def search(self, vector: np.ndarray, k: int = 3):
        distances, indices = self.index.search(vector.reshape(1, -1), k)
        return list(zip(indices[0], distances[0]))

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)


# Singleton Instance
faiss_manager = FaissManager(dim=1024)
