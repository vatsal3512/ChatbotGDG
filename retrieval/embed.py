"""
retrieval/embed.py
===================
Builds the Chroma vector index and BM25 sparse index for the problem corpus.
"""

import json
import logging
import os
import pickle
import sys

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from data.models import Problem

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Must match exactly for Streamlit patch compatibility
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

# BM25 requires tokenized text
try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    logger.error("pip install rank-bm25")
    raise e


def chunk_problem(problem: Problem) -> list[dict]:
    """
    Split a problem into semantic chunks.
    Since full statements might not be present for all problems, we use
    the name and tags as primary chunks, plus the statement if available.
    """
    chunks = []
    pid = problem.id

    # 1. Name chunk
    chunks.append({
        "id": f"{pid}_name",
        "text": f"Problem Name: {problem.name}",
        "metadata": {
            "problem_id": pid,
            "chunk_type": "name",
            "rating": problem.rating or -1,
            "tags": problem.tags
        }
    })

    # 2. Tags chunk
    if problem.tags and problem.tags != "[]":
        tags_list = problem.get_tags()
        tags_str = ", ".join(tags_list)
        chunks.append({
            "id": f"{pid}_tags",
            "text": f"Tags: {tags_str}",
            "metadata": {
                "problem_id": pid,
                "chunk_type": "tags",
                "rating": problem.rating or -1,
                "tags": problem.tags
            }
        })

    # 3. Statement chunk (if available)
    if problem.statement:
        # For a full implementation we'd split long statements into 512-token windows,
        # but for CP problems, we can often embed the whole thing or a large prefix.
        # Here we just take the first ~2000 chars to avoid exceeding context lengths.
        text = problem.statement[:2000]
        chunks.append({
            "id": f"{pid}_statement",
            "text": f"Problem Statement:\n{text}",
            "metadata": {
                "problem_id": pid,
                "chunk_type": "statement",
                "rating": problem.rating or -1,
                "tags": problem.tags
            }
        })

    return chunks


def build_indices():
    db_path = os.getenv("DB_PATH", "data/cf_problems.db")
    chroma_path = os.getenv("CHROMA_PATH", "retrieval/chroma_db")
    bm25_path = os.getenv("BM25_INDEX_PATH", "retrieval/bm25_index.pkl")
    
    if not os.path.exists(db_path):
        logger.error("Database not found at %s. Run ingest_codeforces.py first.", db_path)
        sys.exit(1)

    # 1. Load data from SQLite
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    problems = session.query(Problem).all()
    session.close()
    
    logger.info("Loaded %d problems from SQLite.", len(problems))

    # 2. Chunking
    all_chunks = []
    for p in problems:
        all_chunks.extend(chunk_problem(p))
        
    logger.info("Generated %d chunks.", len(all_chunks))

    # 3. ChromaDB setup with BGE embeddings
    # Using BGE base english version 1.5
    model_name = "BAAI/bge-base-en-v1.5"
    logger.info("Initializing ChromaDB with embedding model: %s", model_name)
    bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    
    # We use a PersistentClient to write to disk
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    
    collection = chroma_client.get_or_create_collection(
        name="codeforces_problems",
        embedding_function=bge_ef
    )

    # Prepare data for Chroma in batches to avoid memory/API limits
    ids = [c["id"] for c in all_chunks]
    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    batch_size = 5000
    logger.info("Upserting to ChromaDB in batches of %d...", batch_size)
    for i in tqdm(range(0, len(ids), batch_size)):
        collection.upsert(
            ids=ids[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    logger.info("ChromaDB index built at %s", chroma_path)

    # 4. BM25 setup
    logger.info("Building BM25 sparse index...")
    # Simple whitespace tokenization is usually fine for a baseline
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save BM25 object and the corresponding chunk IDs so we can map ranks back to IDs
    os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "metadatas": metadatas}, f)
        
    logger.info("BM25 index built at %s", bm25_path)


if __name__ == "__main__":
    build_indices()
