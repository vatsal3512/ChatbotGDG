"""
tests/test_ingest.py
=====================
Tests for data ingestion and the SQLite schema.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Problem


def test_db_has_minimum_rows():
    """
    Verify that cf_problems.db has been populated with at least 500 rows,
    and that tags and ratings are not null.
    """
    db_path = os.getenv("DB_PATH", "data/cf_problems.db")
    
    # If the DB doesn't exist, the ingestion script hasn't run or failed
    assert os.path.exists(db_path), f"Database not found at {db_path}"

    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    total_problems = session.query(Problem).count()
    assert total_problems >= 500, f"Expected >= 500 problems, found {total_problems}"

    # Take a sample of 100 problems to verify schema invariants
    sample = session.query(Problem).limit(100).all()
    for p in sample:
        assert p.id is not None
        assert p.rating is not None, f"Problem {p.id} has no rating"
        assert p.tags is not None, f"Problem {p.id} has no tags string"
        
        # Tags should decode to a list
        tags_list = p.get_tags()
        assert isinstance(tags_list, list), f"Problem {p.id} tags did not decode to list"
        # Since we filtered out untagged problems, it should have at least one tag
        assert len(tags_list) > 0, f"Problem {p.id} has empty tags list"

    session.close()
