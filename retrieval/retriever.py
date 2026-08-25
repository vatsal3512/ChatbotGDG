"""
retrieval/retriever.py
=======================
Hybrid retrieval implementation using BM25 + Dense Chroma + Reciprocal Rank Fusion + Reranking.
"""

import json
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

# For cross-encoder reranker
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Must match exactly for Streamlit patch compatibility
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass


@dataclass
class SearchResult:
    problem_id: str
    chunk_type: str
    score: float
    metadata: dict[str, Any]


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """
    Combines ranked lists using RRF.
    rank_lists: list of lists of item IDs, ordered by rank (best first)
    Returns: dict mapping item ID to its RRF score.
    """
    fused_scores = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += 1.0 / (k + rank + 1)
    return fused_scores


class HybridRetriever:
    """
    Performs hybrid search by combining BM25 and Chroma dense embeddings,
    fusing with RRF, and finally reranking the top candidates with a cross-encoder.
    """
    
    def __init__(
        self,
        chroma_path: str | None = None,
        bm25_path: str | None = None,
        reranker_model_name: str = "BAAI/bge-reranker-base"
    ):
        self.chroma_path = chroma_path or os.getenv("CHROMA_PATH", "retrieval/chroma_db")
        self.bm25_path = bm25_path or os.getenv("BM25_INDEX_PATH", "retrieval/bm25_index.pkl")
        
        # 1. Load Chroma
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-base-en-v1.5"
            )
            self.collection = self.chroma_client.get_collection(
                name="codeforces_problems",
                embedding_function=self.bge_ef
            )
        except Exception as e:
            logger.error("Failed to load Chroma collection from %s: %s", self.chroma_path, e)
            self.collection = None

        # 2. Load BM25
        try:
            with open(self.bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.bm25_ids = data["ids"]
                self.bm25_metadatas = data["metadatas"]
        except Exception as e:
            logger.error("Failed to load BM25 index from %s: %s", self.bm25_path, e)
            self.bm25 = None

        # 3. Load Reranker
        try:
            # CPU compatible reranker
            self.reranker = CrossEncoder(reranker_model_name)
        except Exception as e:
            logger.error("Failed to load reranker %s: %s", reranker_model_name, e)
            self.reranker = None

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        tag_filter: str | None = None,
        rating_min: int | None = None,
        rating_max: int | None = None
    ) -> list[SearchResult]:
        """
        Executes the hybrid search pipeline.
        """
        if not self.collection or not self.bm25:
            logger.warning("Search indices are not fully loaded. Returning empty.")
            return []

        # -- Build where clause for Chroma metadata filtering --
        where = {}
        filters = []
        if tag_filter:
            # We stored tags as string representation of list in SQLite,
            # but in Chroma metadata we might have stored it as string.
            # Usually Chroma where clauses on strings require exact matches for basic ops,
            # but Chroma supports $contains for strings. Let's do a simple check.
            filters.append({"tags": {"$contains": tag_filter}})
        if rating_min is not None:
            filters.append({"rating": {"$gte": rating_min}})
        if rating_max is not None:
            filters.append({"rating": {"$lte": rating_max}})

        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        # Number of candidates to fetch from each sub-system
        fetch_k = max(20, k * 4)

        # 1. Sparse Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get top fetch_k indices
        top_n_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]
        
        # Apply sparse metadata filters manually since BM25 doesn't do it natively
        sparse_candidates = []
        sparse_metadata_map = {}
        for idx in top_n_idx:
            chunk_id = self.bm25_ids[idx]
            meta = self.bm25_metadatas[idx]
            
            # manual filter check
            pass_tag = True
            if tag_filter and tag_filter.lower() not in str(meta.get("tags", "")).lower():
                pass_tag = False
            pass_rating_min = True
            if rating_min is not None and meta.get("rating", -1) < rating_min:
                pass_rating_min = False
            pass_rating_max = True
            if rating_max is not None and meta.get("rating", -1) > rating_max:
                pass_rating_max = False
                
            if pass_tag and pass_rating_min and pass_rating_max:
                sparse_candidates.append(chunk_id)
                sparse_metadata_map[chunk_id] = meta

        # 2. Dense Search (Chroma)
        try:
            dense_res = self.collection.query(
                query_texts=[query],
                n_results=fetch_k,
                where=where if where else None
            )
            dense_candidates = dense_res["ids"][0] if dense_res["ids"] else []
            dense_metadatas = dense_res["metadatas"][0] if dense_res["metadatas"] else []
            dense_metadata_map = {cid: meta for cid, meta in zip(dense_candidates, dense_metadatas)}
        except Exception as e:
            logger.warning("Dense search failed: %s", e)
            dense_candidates = []
            dense_metadata_map = {}

        # Merge metadata maps
        full_metadata_map = {**sparse_metadata_map, **dense_metadata_map}

        # 3. Reciprocal Rank Fusion
        fused_scores = reciprocal_rank_fusion([sparse_candidates, dense_candidates], k=60)
        
        # Sort by RRF score
        fused_ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        # Take top ~20 for reranking
        top_candidates = fused_ranked[:20]

        if not top_candidates:
            return []

        # 4. Reranking
        if self.reranker:
            # CrossEncoder expects pairs: [[query, doc1], [query, doc2], ...]
            # We need the actual text for the docs. 
            # We can get the texts from Chroma.
            candidate_ids = [c[0] for c in top_candidates]
            try:
                # Fetch documents by ID
                docs_res = self.collection.get(ids=candidate_ids)
                # Ensure they align with candidate_ids order
                doc_map = {cid: txt for cid, txt in zip(docs_res["ids"], docs_res["documents"])}
                pairs = [[query, doc_map.get(cid, "")] for cid in candidate_ids]
                
                rerank_scores = self.reranker.predict(pairs)
                
                # Combine scores
                for i, cid in enumerate(candidate_ids):
                    top_candidates[i] = (cid, float(rerank_scores[i]))
                    
                # Re-sort by reranker score
                top_candidates = sorted(top_candidates, key=lambda x: x[1], reverse=True)
            except Exception as e:
                logger.warning("Reranking failed, falling back to RRF scores: %s", e)

        # 5. Format results and return top k
        results = []
        for cid, score in top_candidates[:k]:
            meta = full_metadata_map.get(cid, {})
            results.append(SearchResult(
                problem_id=meta.get("problem_id", ""),
                chunk_type=meta.get("chunk_type", ""),
                score=score,
                metadata=meta
            ))
            
        return results
