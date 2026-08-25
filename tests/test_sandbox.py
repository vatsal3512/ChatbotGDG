"""
tests/test_sandbox.py
======================
Tests for the sandboxed code executor.
"""

from sandbox.executor import execute

def test_execute_python_ac():
    code = "import sys\nfor line in sys.stdin:\n    a, b = map(int, line.split())\n    print(a + b)"
    cases = [
        {"input": "1 2\n", "output": "3\n"},
        {"input": "100 200\n", "output": "300\n"}
    ]
    
    res = execute(code, "python", cases)
    assert res.all_passed is True
    assert res.passed == 2
    assert len(res.details) == 2
    for detail in res.details:
        assert detail["status"] == "AC"


def test_execute_python_wa():
    code = "import sys\nfor line in sys.stdin:\n    a, b = map(int, line.split())\n    print(a * b)" # wrong operator
    cases = [
        {"input": "1 2\n", "output": "3\n"}
    ]
    
    res = execute(code, "python", cases)
    assert res.all_passed is False
    assert res.passed == 0
    assert res.details[0]["status"] == "WA"


def test_execute_python_tle():
    code = "while True: pass"
    cases = [
        {"input": "1\n", "output": "1\n"}
    ]
    
    # Use a short timeout for the test
    res = execute(code, "python", cases, timeout_sec=0.5)
    assert res.all_passed is False
    assert res.details[0]["status"] == "TLE"
