"""
agent/tools.py
===============
Tool implementations and their JSON schema definitions for the LLM.
"""

import json
import logging
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Problem
from retrieval.retriever import HybridRetriever
from sandbox.executor import execute

logger = logging.getLogger(__name__)

# Initialize dependencies lazily
_retriever = None
_engine = None
_Session = None


def get_db_session():
    global _engine, _Session
    if _engine is None:
        db_path = os.getenv("DB_PATH", "data/cf_problems.db")
        _engine = create_engine(f"sqlite:///{db_path}")
        _Session = sessionmaker(bind=_engine)
    return _Session()


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def search_problems(
    query: str,
    tags: str | None = None,
    rating_min: int | None = None,
    rating_max: int | None = None
) -> str:
    """Find problems matching query and filters."""
    retriever = get_retriever()
    results = retriever.hybrid_search(
        query=query,
        k=5,
        tag_filter=tags,
        rating_min=rating_min,
        rating_max=rating_max
    )
    
    if not results:
        return "No problems found matching the criteria."
        
    formatted = []
    for r in results:
        pid = r.problem_id
        meta = r.metadata
        chunk_type = meta.get("chunk_type", "unknown")
        rating = meta.get("rating", "N/A")
        tags_str = meta.get("tags", "")
        formatted.append(f"- ID: {pid} (Rating: {rating}, Tags: {tags_str}) [Matched on {chunk_type}]")
        
    return "Found problems:\n" + "\n".join(formatted)


def get_problem(problem_id: str) -> str:
    """Retrieve full statement and samples for a problem."""
    session = get_db_session()
    try:
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if not problem:
            return f"Problem {problem_id} not found in the database."
            
        statement = problem.statement or "Statement text not available offline."
        samples = problem.get_samples()
        
        out = [f"Problem {problem.id}: {problem.name}"]
        out.append(f"Rating: {problem.rating}")
        out.append(f"Tags: {problem.tags}")
        out.append("\nStatement:")
        out.append(statement)
        
        if samples:
            out.append("\nSample Tests:")
            for i, s in enumerate(samples):
                out.append(f"--- Sample {i+1} ---")
                out.append("Input:\n" + s.get("input", "").strip())
                out.append("Output:\n" + s.get("output", "").strip())
                
        return "\n".join(out)
    finally:
        session.close()


def get_editorial(problem_id: str) -> str:
    """Retrieve editorial link/text for a problem."""
    session = get_db_session()
    try:
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if not problem:
            return f"Problem {problem_id} not found."
            
        if problem.editorial_url:
            return f"Editorial available at: {problem.editorial_url}"
        else:
            # We don't scrape full editorials offline in this build, so point to CF
            pid = problem.id
            if len(pid) > 1 and pid[0].isdigit():
                # E.g. 1234A -> contest 1234
                contest_id = "".join([c for c in pid if c.isdigit()])
                return f"No offline editorial found. Check the contest materials: https://codeforces.com/contest/{contest_id}"
            return "Editorial not available offline."
    finally:
        session.close()


def run_code(code: str, language: str, problem_id: str) -> str:
    """Execute code against problem sample tests."""
    session = get_db_session()
    try:
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if not problem:
            return f"Problem {problem_id} not found."
            
        samples = problem.get_samples()
        if not samples:
            return f"No sample tests available for problem {problem_id} to run against."
            
        result = execute(code, language, samples)
        
        out = [f"Execution Results (Language: {language}):"]
        out.append(f"Passed {result.passed} out of {result.total} sample tests.")
        
        if result.error:
            out.append(f"\nSystem Error:\n{result.error}")
            
        for i, tc in enumerate(result.details):
            out.append(f"\n--- Test Case {tc.get('test_case', i+1)} ---")
            status = tc.get("status")
            out.append(f"Verdict: {status}")
            out.append(f"Runtime: {tc.get('runtime', 0.0):.3f}s")
            
            if status != "AC":
                out.append("Expected:")
                out.append(tc.get("expected", "").strip())
                out.append("Got:")
                out.append(tc.get("stdout", "").strip())
                if tc.get("stderr"):
                    out.append("Stderr:")
                    out.append(tc.get("stderr").strip())
                    
        return "\n".join(out)
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# JSON Schemas for LLM
# ─────────────────────────────────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_problems",
            "description": "Search for competitive programming problems matching a query, topic, tags, or rating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'shortest path in unweighted graph' or 'find the maximum subarray sum'"
                    },
                    "tags": {
                        "type": "string",
                        "description": "Optional tag to filter by, e.g. 'graphs', 'dp', 'math'"
                    },
                    "rating_min": {
                        "type": "integer",
                        "description": "Optional minimum problem rating (difficulty), e.g. 1200"
                    },
                    "rating_max": {
                        "type": "integer",
                        "description": "Optional maximum problem rating, e.g. 2000"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_problem",
            "description": "Retrieve the full problem statement and sample test cases for a specific problem ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_id": {
                        "type": "string",
                        "description": "The exact problem ID, e.g. '1234A' or '1500C'"
                    }
                },
                "required": ["problem_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_editorial",
            "description": "Get the editorial or hint link for a specific problem ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_id": {
                        "type": "string",
                        "description": "The exact problem ID, e.g. '1234A'"
                    }
                },
                "required": ["problem_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute a submitted code solution against a problem's sample test cases and return the pass/fail results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The complete source code to run"
                    },
                    "language": {
                        "type": "string",
                        "description": "The programming language. Must be 'python' or 'cpp'"
                    },
                    "problem_id": {
                        "type": "string",
                        "description": "The exact problem ID to test against, e.g. '1234A'"
                    }
                },
                "required": ["code", "language", "problem_id"]
            }
        }
    }
]

# Dispatch map
TOOL_FUNCTIONS = {
    "search_problems": search_problems,
    "get_problem": get_problem,
    "get_editorial": get_editorial,
    "run_code": run_code,
}
