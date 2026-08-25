"""
eval/generation_eval.py
========================
Evaluates generation using LLM-as-a-judge.
"""

import json
import logging
import os
import time
from datetime import datetime

from agent.llm_client import get_llm_client
from agent.loop import AgentLoop
from agent.prompts import JUDGE_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def evaluate():
    golden_path = "eval/golden_set.json"
    if not os.path.exists(golden_path):
        logger.error("Golden set not found at %s", golden_path)
        return

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
        
    # Take a smaller sample for generation eval to save time and API quota
    # We will evaluate 5 examples
    sample = golden_set[:5]

    logger.info("Initializing Agent Loop and Judge...")
    agent = AgentLoop()
    judge = get_llm_client() # same provider as agent

    results = []
    total_scores = {"faithfulness": 0, "relevance": 0, "non_reveal": 0}
    
    for i, entry in enumerate(sample):
        query = entry["generation_query"]
        rubric = entry["rubric"]
        
        logger.info("Evaluating [%d/%d] Query: %s", i+1, len(sample), query)
        
        try:
            # 1. Run agent
            start_t = time.time()
            agent_response = agent.chat(query)
            latency = time.time() - start_t
        except Exception as e:
            logger.warning("Agent call failed for query %d: %s — skipping.", i+1, e)
            results.append({"query": query, "response": f"[SKIPPED: {e}]", "scores": {}, "latency_s": 0})
            time.sleep(15)
            continue

        try:
            # 2. Judge response
            judge_prompt = f"""Evaluate this AI response based on the rubric.
        
Query: {query}
Rubric: {rubric}

AI Response:
{agent_response}
"""
            judge_res = judge.chat(
                messages=[{"role": "user", "content": judge_prompt}],
                system=JUDGE_SYSTEM_PROMPT
            )
            
            # Parse JSON from judge (strip markdown block if any)
            text = judge_res.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            try:
                scores = json.loads(text.strip())
            except json.JSONDecodeError:
                logger.warning("Judge returned invalid JSON. Defaulting to 1s. Raw: %s", text)
                scores = {"faithfulness": 1, "relevance": 1, "non_reveal": 1, "reasoning": "JSON parse error"}
        except Exception as e:
            logger.warning("Judge call failed for query %d: %s — defaulting to 1s.", i+1, e)
            scores = {"faithfulness": 1, "relevance": 1, "non_reveal": 1, "reasoning": f"Judge error: {e}"}
            
        for k in total_scores:
            total_scores[k] += scores.get(k, 1)
            
        results.append({
            "query": query,
            "response": agent_response,
            "scores": scores,
            "latency_s": latency
        })
        
        # Increase sleep to respect free-tier rate limits (20 req/day)
        time.sleep(10)

    n = len(sample)
    summary = {k: v / n for k, v in total_scores.items()}
    summary["mean_total"] = sum(summary.values()) / 3
    summary["total_queries"] = n
    
    logger.info("--- Generation Evaluation Summary ---")
    logger.info("Faithfulness: %.2f/5", summary["faithfulness"])
    logger.info("Relevance:    %.2f/5", summary["relevance"])
    logger.info("Non-reveal:   %.2f/5", summary["non_reveal"])
    logger.info("Overall Mean: %.2f/5", summary["mean_total"])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"eval/results/{timestamp}_generation.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
        
    logger.info("Results saved to %s", out_path)
    
    if summary["mean_total"] < 3.5:
        logger.warning("FAILED: Mean generation score is below 3.5")
    else:
        logger.info("PASSED: Generation score meets threshold")

if __name__ == "__main__":
    evaluate()
