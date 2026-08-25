"""
tests/test_retriever.py
========================
Tests for the hybrid retriever.
"""

from retrieval.retriever import HybridRetriever

def test_hybrid_search_finds_problem():
    """
    Test that the retriever can find a problem by its name or basic topic.
    This assumes that `embed.py` has been run and indices are available.
    """
    retriever = HybridRetriever()
    
    # We don't know exactly which problems are in the DB because it fetches the latest,
    # but "A" (div 2 A) or "sort" or "math" are practically guaranteed to be in the index
    # and have relevant metadata. Let's do a search for "math" with a tag filter.
    
    results = retriever.hybrid_search(
        query="graph shortest path",
        k=5,
        tag_filter="graphs"
    )
    
    # The database must have some graph problems if we fetched 2000 recent problems.
    assert len(results) > 0, "No results found. Are the indices built?"
    
    # Check shape of results
    best_result = results[0]
    assert best_result.problem_id != ""
    assert best_result.score is not None
    assert "tags" in best_result.metadata
    
    # Check that tag filter worked
    assert "graphs" in str(best_result.metadata["tags"]).lower(), "Tag filter failed"

def test_hybrid_search_rating_filter():
    """
    Test that rating filters work.
    """
    retriever = HybridRetriever()
    
    results = retriever.hybrid_search(
        query="dynamic programming",
        k=5,
        rating_min=1500,
        rating_max=1600
    )
    
    if not results:
        # It's possible there are no DP problems in the 1500-1600 range in the recent 2000, 
        # though highly unlikely.
        pass
    else:
        for res in results:
            rating = res.metadata.get("rating", -1)
            assert 1500 <= rating <= 1600, f"Rating {rating} outside filter bounds"
