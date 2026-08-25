# ── Streamlit Community Cloud startup script ───────────────────────────────
# This runs before the app starts on Streamlit Cloud to bootstrap the DB
# and vector indices (which can't be committed to git due to size).
#
# Cloud-specific notes:
# - pysqlite3-binary is installed from packages.txt so sqlite3 works
# - All indices are rebuilt from the Codeforces public API (no credentials needed)
# - Models are downloaded from HuggingFace on first run and cached between sessions

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def _needs_rebuild(path: str, min_size_bytes: int = 1024) -> bool:
    return not os.path.exists(path) or os.path.getsize(path) < min_size_bytes


def bootstrap():
    db_path = os.getenv("DB_PATH", "data/cf_problems.db")
    chroma_path = os.getenv("CHROMA_PATH", "retrieval/chroma_db")
    bm25_path = os.getenv("BM25_INDEX_PATH", "retrieval/bm25_index.pkl")

    if _needs_rebuild(db_path):
        logger.info("DB not found — running ingestion...")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "data.ingest_codeforces"], capture_output=True, text=True)
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error("Ingestion failed: %s", result.stderr)
    else:
        logger.info("DB already exists at %s", db_path)

    if _needs_rebuild(bm25_path) or not os.path.exists(chroma_path):
        logger.info("Indices not found — building vector + BM25 indices (this takes ~2 min on first run)...")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "retrieval.embed"], capture_output=True, text=True)
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error("Embedding failed: %s", result.stderr)
    else:
        logger.info("Indices already exist")


if __name__ == "__main__":
    bootstrap()
