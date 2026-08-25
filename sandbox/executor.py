"""
sandbox/executor.py
====================
Executes code submissions in a sandboxed subprocess.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Try to use resource for memory limits, but it's Unix only.
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


@dataclass
class ExecutionResult:
    passed: int
    total: int
    details: list[dict[str, Any]]
    error: str | None = None
    all_passed: bool = False


def _set_memory_limit(mb: int):
    """Sets a memory limit for the subprocess (Unix only)."""
    if HAS_RESOURCE:
        bytes_limit = mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))


def _run_test_case(
    run_cmd: list[str],
    input_data: str,
    expected_output: str,
    timeout_sec: float
) -> dict[str, Any]:
    """Runs a single test case and compares the output."""
    try:
        start_time = time.time()
        
        # Pre-exec fn only works on Unix for setting limits
        preexec_fn = None
        if HAS_RESOURCE and os.name != "nt":
            # Cap memory to 256MB
            preexec_fn = lambda: _set_memory_limit(256)
            
        process = subprocess.Popen(
            run_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=preexec_fn
        )
        
        stdout, stderr = process.communicate(input=input_data, timeout=timeout_sec)
        runtime = time.time() - start_time
        
        if process.returncode != 0:
            return {
                "status": "RE", # Runtime Error
                "runtime": runtime,
                "stdout": stdout,
                "stderr": stderr,
                "expected": expected_output
            }
            
        # Compare whitespace-tolerantly
        out_tokens = stdout.strip().split()
        exp_tokens = expected_output.strip().split()
        
        if out_tokens == exp_tokens:
            return {
                "status": "AC", # Accepted
                "runtime": runtime,
                "stdout": stdout,
                "stderr": stderr,
                "expected": expected_output
            }
        else:
            return {
                "status": "WA", # Wrong Answer
                "runtime": runtime,
                "stdout": stdout,
                "stderr": stderr,
                "expected": expected_output
            }
            
    except subprocess.TimeoutExpired as e:
        process.kill()
        return {
            "status": "TLE", # Time Limit Exceeded
            "runtime": timeout_sec,
            "stdout": (e.stdout or b"").decode("utf-8", "ignore"),
            "stderr": (e.stderr or b"").decode("utf-8", "ignore"),
            "expected": expected_output
        }
    except Exception as e:
        return {
            "status": "ERR",
            "runtime": 0.0,
            "stdout": "",
            "stderr": str(e),
            "expected": expected_output
        }


def execute(
    code: str,
    language: str,
    test_cases: list[dict[str, str]],
    timeout_sec: float = 5.0
) -> ExecutionResult:
    """
    Executes the given code against the test cases.
    Supported languages: "python", "cpp".
    """
    if language not in ("python", "cpp"):
        return ExecutionResult(0, len(test_cases), [], error=f"Unsupported language: {language}")

    temp_dir = tempfile.mkdtemp(prefix="cf_sandbox_")
    
    try:
        run_cmd = []
        
        if language == "python":
            source_file = os.path.join(temp_dir, "solution.py")
            with open(source_file, "w", encoding="utf-8") as f:
                f.write(code)
            run_cmd = [sys.executable if "sys" in globals() else "python", source_file]
            
        elif language == "cpp":
            source_file = os.path.join(temp_dir, "solution.cpp")
            executable = os.path.join(temp_dir, "solution" + (".exe" if os.name == "nt" else ""))
            with open(source_file, "w", encoding="utf-8") as f:
                f.write(code)
                
            # Compile
            compile_proc = subprocess.run(
                ["g++", "-O2", source_file, "-o", executable],
                capture_output=True,
                text=True
            )
            if compile_proc.returncode != 0:
                return ExecutionResult(
                    passed=0,
                    total=len(test_cases),
                    details=[],
                    error=f"Compilation Error:\n{compile_proc.stderr}"
                )
            run_cmd = [executable]

        # Run test cases
        details = []
        passed = 0
        
        import sys # make sure sys is imported for python executable path
        if language == "python" and not run_cmd[0].endswith("python") and "python" not in run_cmd[0]:
             run_cmd[0] = sys.executable

        for i, tc in enumerate(test_cases):
            res = _run_test_case(run_cmd, tc.get("input", ""), tc.get("output", ""), timeout_sec)
            res["test_case"] = i + 1
            details.append(res)
            if res["status"] == "AC":
                passed += 1
                
        return ExecutionResult(
            passed=passed,
            total=len(test_cases),
            details=details,
            all_passed=(passed == len(test_cases) and len(test_cases) > 0)
        )
        
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("Failed to clean up temp dir %s: %s", temp_dir, e)

