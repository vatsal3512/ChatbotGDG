"""
data/models.py
===============
SQLAlchemy models for Codeforces problems.
"""

import json
from typing import Any

from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Problem(Base):
    """
    Represents a Codeforces problem.
    """
    __tablename__ = "problems"

    # We use composite contestId + index as the primary key string, e.g., "1234A"
    id = Column(String, primary_key=True)
    
    contest_id = Column(Integer, nullable=True)
    index = Column(String, nullable=False)
    name = Column(String, nullable=False)
    rating = Column(Integer, nullable=True)
    
    # Store tags as JSON string to easily query/deserialize
    tags = Column(Text, nullable=False, default="[]")
    
    # Full problem statement text
    statement = Column(Text, nullable=True)
    
    # Sample test cases as JSON string: [{"input": "...", "output": "..."}, ...]
    samples = Column(Text, nullable=True, default="[]")
    
    # Editorial URL if available
    editorial_url = Column(String, nullable=True)
    
    # Track origin (e.g. "api", "scraped")
    source = Column(String, nullable=True)
    
    # Number of users who solved it (useful for popularity metrics)
    solved_count = Column(Integer, nullable=True)

    def set_tags(self, tags_list: list[str]) -> None:
        self.tags = json.dumps(tags_list)

    def get_tags(self) -> list[str]:
        if not self.tags:
            return []
        try:
            return json.loads(self.tags)
        except json.JSONDecodeError:
            return []

    def set_samples(self, samples_list: list[dict[str, str]]) -> None:
        self.samples = json.dumps(samples_list)

    def get_samples(self) -> list[dict[str, str]]:
        if not self.samples:
            return []
        try:
            return json.loads(self.samples)
        except json.JSONDecodeError:
            return []
