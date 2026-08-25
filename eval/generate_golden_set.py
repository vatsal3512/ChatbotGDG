"""
eval/generate_golden_set.py
===========================
Generates a golden set by sampling problems from the SQLite database.
"""

import json
import os
import random

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Problem


def generate():
    db_path = os.getenv("DB_PATH", "data/cf_problems.db")
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    problems = session.query(Problem).all()
    if not problems:
        print("No problems found in DB.")
        return

    random.seed(42)
    sample = random.sample(problems, min(30, len(problems)))

    golden_set = []
    
    for p in sample:
        # Create a retrieval query based on problem name and a tag
        tags = p.get_tags()
        tag = tags[0] if tags else "competitive programming"
        query_retrieval = f"{p.name} {tag}"
        
        # Create a generation query asking for a hint
        query_generation = f"I'm stuck on problem {p.name} (ID: {p.id}). Give me a nudge."
        
        entry = {
            "retrieval_query": query_retrieval,
            "expected_problem_ids": [p.id],
            "generation_query": query_generation,
            "rubric": (
                f"The response should mention the problem '{p.name}', "
                f"suggest an approach related to '{tag}', and NOT give full code."
            )
        }
        golden_set.append(entry)
        
    session.close()

    out_path = "eval/golden_set.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, indent=2)
        
    print(f"Generated {len(golden_set)} golden set examples at {out_path}")


if __name__ == "__main__":
    generate()
