"""
eval/retrieval_eval.py
=======================
Evaluates the hybrid retriever against the golden set.
"""

import json
import logging
import os
import time
from datetime import datetime

from retrieval.retriever import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def mrr_score(expected_ids: list[str], retrieved_ids: list[str]) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid in expected_ids:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(expected_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    # Recall is 1 if any expected ID is in top k, since usually we look for 1 specific problem
    for eid in expected_ids:
        if eid in retrieved_k:
            return 1.0
    return 0.0

def evaluate():
    golden_path = "eval/golden_set.json"
    if not os.path.exists(golden_path):
        logger.error("Golden set not found at %s", golden_path)
        return

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    logger.info("Initializing retriever...")
    retriever = HybridRetriever()

    results = []
    
    total_mrr = 0.0
    total_r5 = 0.0
    total_r10 = 0.0
    
    logger.info("Evaluating %d queries...", len(golden_set))
    for entry in golden_set:
        query = entry["retrieval_query"]
        expected_ids = entry["expected_problem_ids"]
        
        # Retrieve top 10
        start_t = time.time()
        search_res = retriever.hybrid_search(query, k=10)
        latency = time.time() - start_t
        
        retrieved_ids = [r.problem_id for r in search_res]
        
        mrr = mrr_score(expected_ids, retrieved_ids)
        r5 = recall_at_k(expected_ids, retrieved_ids, 5)
        r10 = recall_at_k(expected_ids, retrieved_ids, 10)
        
        total_mrr += mrr
        total_r5 += r5
        total_r10 += r10
        
        results.append({
            "query": query,
            "expected": expected_ids,
            "retrieved": retrieved_ids,
            "mrr": mrr,
            "recall@5": r5,
            "recall@10": r10,
            "latency_s": latency
        })

    n = len(golden_set)
    summary = {
        "mean_mrr": total_mrr / n,
        "mean_recall@5": total_r5 / n,
        "mean_recall@10": total_r10 / n,
        "total_queries": n
    }
    
    logger.info("--- Retrieval Evaluation Summary ---")
    logger.info("MRR:        %.3f", summary["mean_mrr"])
    logger.info("Recall@5:   %.3f", summary["mean_recall@5"])
    logger.info("Recall@10:  %.3f", summary["mean_recall@10"])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"eval/results/{timestamp}_retrieval.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
        
    logger.info("Results saved to %s", out_path)
    
    # Threshold check
    if summary["mean_recall@10"] < 0.70:
        logger.warning("FAILED: Recall@10 is below threshold (0.70)")
    else:
        logger.info("PASSED: Recall@10 meets threshold")

if __name__ == "__main__":
    evaluate()
