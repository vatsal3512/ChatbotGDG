"""
tests/test_agent_loop.py
=========================
Tests for the agent loop and tool dispatch logic.
"""

from unittest.mock import MagicMock, patch

from agent.llm_client import LLMResponse, ToolCall
from agent.loop import AgentLoop


@patch("agent.loop.get_llm_client")
def test_agent_loop_search(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    # 1. First turn: LLM decides to search
    # 2. Second turn: LLM gives text answer
    mock_client.chat.side_effect = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(name="search_problems", arguments={"query": "dp"}, id="call_1")],
            stop_reason="tool_calls"
        ),
        LLMResponse(
            text="I found some DP problems.",
            tool_calls=[],
            stop_reason="stop"
        )
    ]
    
    with patch("agent.loop.TOOL_FUNCTIONS", {"search_problems": lambda query: "Found 1234A"}):
        loop = AgentLoop()
        final_answer = loop.chat("Find me DP problems")
        
    assert final_answer == "I found some DP problems."
    # Client should have been called twice
    assert mock_client.chat.call_count == 2
    
    # The second call's messages should contain the tool output
    second_call_args = mock_client.chat.call_args_list[1][1]
    messages = second_call_args["messages"]
    
    # Check that tool response was appended
    assert messages[-2]["role"] == "tool"
    assert messages[-2]["name"] == "search_problems"
    assert messages[-2]["content"] == "Found 1234A"


@patch("agent.loop.get_llm_client")
def test_agent_loop_get_problem(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_client.chat.side_effect = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(name="get_problem", arguments={"problem_id": "1234A"}, id="call_1")],
            stop_reason="tool_calls"
        ),
        LLMResponse(
            text="Here is the problem statement.",
            tool_calls=[],
            stop_reason="stop"
        )
    ]
    
    with patch("agent.loop.TOOL_FUNCTIONS", {"get_problem": lambda problem_id: "Statement for 1234A"}):
        loop = AgentLoop()
        loop.chat("What is problem 1234A?")
        
    messages = mock_client.chat.call_args_list[1][1]["messages"]
    assert messages[-2]["role"] == "tool"
    assert "Statement for 1234A" in messages[-2]["content"]


@patch("agent.loop.get_llm_client")
def test_agent_loop_run_code(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_client.chat.side_effect = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(name="run_code", arguments={"code": "print(1)", "language": "python", "problem_id": "1234A"}, id="call_1")],
            stop_reason="tool_calls"
        ),
        LLMResponse(
            text="Your code passed.",
            tool_calls=[],
            stop_reason="stop"
        )
    ]
    
    with patch("agent.loop.TOOL_FUNCTIONS", {"run_code": lambda code, language, problem_id: "AC"}):
        loop = AgentLoop()
        loop.chat("Check this code")
        
    messages = mock_client.chat.call_args_list[1][1]["messages"]
    assert messages[-2]["role"] == "tool"
    assert messages[-2]["name"] == "run_code"
    assert messages[-2]["content"] == "AC"
