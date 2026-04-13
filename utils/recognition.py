"""
recognition.py
--------------
Loads stored embeddings and identifies faces by cosine similarity.
"""

import json
import os
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(_BASE_DIR, "embeddings", "embeddings.json")

DEFAULT_THRESHOLD = 0.45


class EmbeddingStore:
    """In-memory cache of {name: embedding_vector} loaded from JSON."""

    def __init__(self, path=EMBEDDINGS_PATH):
        self._path = path
        self._db = {}
        self.reload()

    def reload(self):
        """Re-read the JSON file from disk."""
        if not os.path.exists(self._path):
            self._db = {}
            return
        try:
            with open(self._path, "r") as f:
                raw = json.load(f)
            self._db = {
                name: np.array(vec, dtype=np.float32)
                for name, vec in raw.items()
            }
        except (json.JSONDecodeError, ValueError):
            self._db = {}

    def save(self, name, embedding):
        """Persist a single embedding to the JSON store."""
        self.reload()
        self._db[name] = embedding
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        serialisable = {n: v.tolist() for n, v in self._db.items()}
        with open(self._path, "w") as f:
            json.dump(serialisable, f, indent=2)

    def identify(self, embedding, threshold=DEFAULT_THRESHOLD):
        """
        Compare *embedding* against all stored embeddings.
        Returns (name, confidence) tuple.
        """
        if not self._db:
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = -1.0

        for name, stored_emb in self._db.items():
            score = float(cosine_similarity(embedding, stored_emb))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < threshold:
            return "Unknown", round(best_score, 4)

        return best_name, round(best_score, 4)

    def list_users(self):
        """Return all registered user names."""
        return list(self._db.keys())

    def remove_user(self, name):
        """Remove a user from the store."""
        if name in self._db:
            del self._db[name]
            serialisable = {n: v.tolist() for n, v in self._db.items()}
            with open(self._path, "w") as f:
                json.dump(serialisable, f, indent=2)
            return True
        return False


def cosine_similarity(a, b):
    """Cosine similarity between two L2-normalised vectors (= dot product)."""
    dot = np.dot(a, b)
    return float(np.clip(dot, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────
_store = None


def get_store():
    global _store
    if _store is None:
        _store = EmbeddingStore()
    return _store


def identify_face(embedding, threshold=DEFAULT_THRESHOLD):
    """Convenience wrapper — identify a face via the singleton store."""
    return get_store().identify(embedding, threshold)


def reload_embeddings():
    """Force-reload embeddings from disk."""
    get_store().reload()
