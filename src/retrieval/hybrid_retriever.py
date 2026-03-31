"""
Hybrid Retriever: BM25 (sparse) + BGE-M3 (dense)
Combines keyword matching with semantic similarity for improved recall.
"""

import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import faiss
from FlagEmbedding import BGEM3FlagModel


class BM25Retriever:
    def __init__(self, corpus: List[str]):
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]


class DenseRetriever:
    def __init__(self, corpus: List[str], model_name: str = "BAAI/bge-m3"):
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
        self.corpus = corpus
        print("Encoding corpus with BGE-M3...")
        embeddings = self.model.encode(corpus, batch_size=32)["dense_vecs"]
        self.embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(self.embeddings)

        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        q_emb = self.model.encode([query])["dense_vecs"].astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, top_k)
        return [(int(indices[0][i]), float(scores[0][i])) for i in range(top_k)]


class HybridRetriever:
    """
    Combines BM25 and dense retrieval using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        corpus: List[str],
        dense_model: str = "BAAI/bge-m3",
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ):
        self.corpus = corpus
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25 = BM25Retriever(corpus)
        self.dense = DenseRetriever(corpus, model_name=dense_model)

    def _rrf_score(self, rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank + 1)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        bm25_results = self.bm25.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)

        # Normalize scores per method
        scores: Dict[int, float] = {}

        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + self.bm25_weight * self._rrf_score(rank)

        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0) + self.dense_weight * self._rrf_score(rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {"doc_id": idx, "score": score, "text": self.corpus[idx]}
            for idx, score in ranked
        ]


def build_retriever(corpus: List[str], config: Dict) -> HybridRetriever:
    retriever_type = config.get("type", "hybrid")

    if retriever_type == "hybrid":
        return HybridRetriever(
            corpus=corpus,
            dense_model=config.get("dense_model", "BAAI/bge-m3"),
            bm25_weight=config.get("bm25_weight", 0.4),
            dense_weight=config.get("dense_weight", 0.6),
        )
    elif retriever_type == "bm25":
        return BM25Retriever(corpus)
    elif retriever_type == "dense":
        return DenseRetriever(corpus, model_name=config.get("dense_model", "BAAI/bge-m3"))
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
