"""
data/ingest_codeforces.py
==========================
Bulk fetches problems from the Codeforces API and stores them in SQLite.
Usage: python -m data.ingest_codeforces
"""

import json
import logging
import os
import sys
import time

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Problem
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CF_API_URL = "https://codeforces.com/api/problemset.problems"
# Default to 2000 problems for a good corpus; but ensure we get >= 500 for the tests
TARGET_PROBLEMS_COUNT = 2000

def get_engine():
    db_path = os.getenv("DB_PATH", "data/cf_problems.db")
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")

def ingest():
    logger.info("Fetching problems from Codeforces API...")
    
    # Retry with backoff for the CF API
    for attempt in range(3):
        try:
            resp = requests.get(CF_API_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "OK":
                raise ValueError(f"API Error: {data.get('comment')}")
            break
        except Exception as e:
            logger.warning("Failed to fetch from CF API (attempt %d/3): %s", attempt + 1, e)
            if attempt == 2:
                logger.error("Giving up.")
                sys.exit(1)
            time.sleep(2 ** attempt)

    problems = data["result"]["problems"]
    stats = data["result"]["problemStatistics"]

    # Create a mapping of problem ID -> solved count
    solved_counts = {}
    for stat in stats:
        pid = f"{stat['contestId']}{stat['index']}"
        solved_counts[pid] = stat.get("solvedCount", 0)

    # Filter problems: Must have rating, and must have tags
    valid_problems = []
    for p in problems:
        if "rating" not in p or p["rating"] is None:
            continue
        if not p.get("tags"):
            continue
        valid_problems.append(p)

    # Sort by descending contestId (newest first) to get more recent problems
    valid_problems.sort(key=lambda x: x["contestId"], reverse=True)
    
    # Cap at the target count to avoid bloating the SQLite database during testing
    valid_problems = valid_problems[:TARGET_PROBLEMS_COUNT]
    
    logger.info("Found %d valid problems to ingest.", len(valid_problems))

    engine = get_engine()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    count_added = 0
    count_updated = 0

    for p in valid_problems:
        pid = f"{p['contestId']}{p['index']}"
        
        # Check if already exists
        existing = session.query(Problem).filter_by(id=pid).first()
        if existing:
            # Update rating, tags, solved_count
            existing.rating = p["rating"]
            existing.set_tags(p["tags"])
            existing.solved_count = solved_counts.get(pid, 0)
            count_updated += 1
        else:
            new_p = Problem(
                id=pid,
                contest_id=p["contestId"],
                index=p["index"],
                name=p["name"],
                rating=p["rating"],
                solved_count=solved_counts.get(pid, 0),
                source="api"
            )
            new_p.set_tags(p["tags"])
            session.add(new_p)
            count_added += 1

    session.commit()
    logger.info("Successfully ingested problems. Added: %d, Updated: %d", count_added, count_updated)
    
    # Verify row count is at least 500 for tests
    total_rows = session.query(Problem).count()
    if total_rows < 500:
        logger.error("Only %d problems in DB, required >= 500. Check API response.", total_rows)
        sys.exit(1)
        
    session.close()

if __name__ == "__main__":
    ingest()
