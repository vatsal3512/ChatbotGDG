"""
agent/loop.py
==============
The core tool-calling agent loop.
"""

import json
import logging
import os
import time

from agent.llm_client import get_llm_client
from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOL_FUNCTIONS, TOOLS_SCHEMA
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Make sure log dir exists
log_path = os.getenv("LOG_PATH", "logs/agent.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)


class AgentLoop:
    def __init__(self, provider: str | None = None):
        self.client = get_llm_client(provider)
        self.max_turns = int(os.getenv("MAX_TOOL_TURNS", "6"))
        self.system_prompt = SYSTEM_PROMPT

    def _log_turn(self, tool_name: str, args: dict, result: str, latency_ms: float):
        """Append a JSON line to the agent log."""
        log_entry = {
            "timestamp": time.time(),
            "tool": tool_name,
            "args": args,
            # We can parse retrieved IDs if it's search_problems, but for now we just log a snippet of result
            "result_snippet": result[:200] + "..." if len(result) > 200 else result,
            "latency_ms": latency_ms
        }
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning("Failed to write to agent log: %s", e)

    def chat(self, user_message: str, history: list[dict] | None = None) -> str:
        """
        Executes the agent loop for a user message.
        """
        if history is None:
            messages = []
        else:
            messages = list(history)
            
        messages.append({"role": "user", "content": user_message})

        for turn in range(self.max_turns):
            logger.info("Agent turn %d...", turn + 1)
            
            # 1. Call LLM
            start_t = time.time()
            response = self.client.chat(
                messages=messages,
                tools=TOOLS_SCHEMA,
                system=self.system_prompt
            )
            latency = (time.time() - start_t) * 1000
            
            # If there's text, we can append it as an assistant message
            # Sometimes models return both text (thought) and tool calls
            if response.text or response.tool_calls:
                # Build the assistant message carefully to include tool_calls for history
                asst_msg = {"role": "assistant"}
                if response.text:
                    asst_msg["content"] = response.text
                if response.tool_calls:
                    asst_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                            }
                        } for tc in response.tool_calls
                    ]
                # Preserve the raw Gemini Content object so thought_signature is
                # passed through verbatim on the next turn (required for thinking models)
                if response._raw_content is not None:
                    asst_msg["_gemini_raw_content"] = response._raw_content
                messages.append(asst_msg)

            # 2. Check if we need to stop
            if response.stop_reason != "tool_calls" or not response.tool_calls:
                # Guard against thinking-model returning a fully empty response
                if not response.text and not response.tool_calls:
                    logger.warning(
                        "LLM returned empty response on turn %d. Prompting for answer.", turn + 1
                    )
                    messages.append({"role": "user", "content": "Please provide your answer."})
                    continue
                return response.text

            # 3. Execute tools
            for tc in response.tool_calls:
                func_name = tc.name
                func_args = tc.arguments
                
                logger.info("Calling tool: %s(%s)", func_name, func_args)
                
                if func_name in TOOL_FUNCTIONS:
                    try:
                        tool_start = time.time()
                        func_result = TOOL_FUNCTIONS[func_name](**func_args)
                        tool_latency = (time.time() - tool_start) * 1000
                        
                        self._log_turn(func_name, func_args, func_result, tool_latency)
                        
                    except Exception as e:
                        logger.error("Tool %s failed: %s", func_name, e)
                        func_result = f"Error executing tool {func_name}: {e}"
                else:
                    func_result = f"Error: Tool {func_name} is not available."
                    
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": func_name,
                    "content": str(func_result)
                })

        # Exceeded max turns
        messages.append({"role": "user", "content": "You have reached the maximum number of tool calls. Please give your final answer based on the information you have."})
        final_response = self.client.chat(messages=messages, system=self.system_prompt)
        return final_response.text


if __name__ == "__main__":
    # CLI REPL
    loop = AgentLoop()
    print("Codeforces RAG Assistant started. Type 'quit' to exit.")
    history = []
    while True:
        try:
            q = input("\nUser> ")
            if q.strip().lower() in ("quit", "exit"):
                break
            if not q.strip():
                continue
                
            ans = loop.chat(q, history=history)
            print("\nAgent>", ans)
            
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": ans})
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("\nError:", e)
