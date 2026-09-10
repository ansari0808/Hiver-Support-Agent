"""Retrieval over historically-resolved threads for grounding replies.

Uses TF-IDF + cosine similarity rather than a neural embedding model.
Deliberate choice: no extra model download/dependency, fully deterministic,
fast enough for a few thousand threads, and easy to explain live.
See DECISION_LOG.md #6.
"""
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config


class ThreadRetriever:
    def __init__(self, threads_path):
        self.threads = [json.loads(l) for l in open(threads_path)]
        corpus = [t["customer_text"] for t in self.threads]
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query_text: str, k: int = config.TOP_K_RETRIEVAL) -> list[dict]:
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for idx in top_idx:
            item = dict(self.threads[idx])
            item["similarity"] = float(sims[idx])
            results.append(item)
        return results


def load_retriever(brand: str = config.BRAND) -> ThreadRetriever:
    path = config.DATA_PROCESSED / f"threads_{brand}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m src.data_prep --brand {brand}` first.")
    return ThreadRetriever(path)
