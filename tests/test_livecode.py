import json

import pytest

from banana_harness.adapters.livecode import livecode_task
from banana_harness.environment import NativeEnvironment
from banana_harness.grading import _stdio_equal, grade_livecode


@pytest.mark.parametrize(
    "code,expected",
    [
        ("print(int(input()) + 1)", "passed"),
        ("print(100)", "failed"),
        ("raise RuntimeError('bug')", "failed"),
    ],
)
def test_stdio(code, expected, tmp_path):
    task = livecode_task(
        {
            "question_id": "synthetic",
            "question_content": "Add one",
            "public_test_cases": json.dumps(
                [{"input": "1\n", "output": "2\n", "testtype": "stdin"}]
            ),
            "private_test_cases": json.dumps(
                [{"input": "7\n", "output": "8\n", "testtype": "stdin"}]
            ),
        }
    )
    (tmp_path / "solution.py").write_text(code)
    grade = grade_livecode(task, NativeEnvironment(tmp_path), tmp_path)
    assert grade.status == expected
    assert grade.details["cases"] == 2


def test_functional_solution_class_and_standard_imports(tmp_path):
    task = livecode_task(
        {
            "question_id": "synthetic",
            "question_content": "Sum",
            "metadata": '{"func_name":"add"}',
            "public_test_cases": json.dumps(
                [{"input": "[1,2]\n3", "output": "6", "testtype": "functional"}]
            ),
        }
    )
    (tmp_path / "solution.py").write_text(
        "class Solution:\n    def add(self, nums: List[int], x: int):\n        return sum(nums)+x\n"
    )
    grade = grade_livecode(task, NativeEnvironment(tmp_path), tmp_path)
    assert grade.status == "passed"


def test_infinite_loop_timeout(tmp_path):
    task = livecode_task(
        {
            "question_id": "synthetic",
            "question_content": "Task",
            "public_test_cases": '[{"input":"1","output":"1"}]',
        }
    )
    task.grading["case_timeout"] = 0.05
    (tmp_path / "solution.py").write_text("while True: pass")
    grade = grade_livecode(task, NativeEnvironment(tmp_path), tmp_path)
    assert grade.status == "failed"
    assert json.loads((tmp_path / "test_cases.json").read_text())[0]["timed_out"]


def test_whitespace_numeric_lines_and_output_order():
    assert _stdio_equal(" 1.0 2.00\n", "1 2")
    assert not _stdio_equal("1\n2", "1 2")
    assert not _stdio_equal("2 1", "1 2")


def test_future_imports_in_stdio_solution(tmp_path):
    task = livecode_task(
        {
            "question_id": "synthetic",
            "question_content": "Echo",
            "public_test_cases": '[{"input":"1","output":"1"}]',
        }
    )
    (tmp_path / "solution.py").write_text("from __future__ import annotations\nprint(input())\n")
    assert grade_livecode(task, NativeEnvironment(tmp_path), tmp_path).status == "passed"


def test_local_upstream_evaluator_in_subprocess(tmp_path):
    # Synthetic evaluator API fixture, never downloaded benchmark code.
    upstream = tmp_path / "upstream/lcb_runner/evaluation"
    upstream.mkdir(parents=True)
    (upstream / "testing_util.py").write_text(
        "def run_test(sample, test, debug, timeout):\n    return [True], {'synthetic':True}\n"
    )
    task = livecode_task(
        {
            "question_id": "synthetic",
            "question_content": "Echo",
            "public_test_cases": '[{"input":"1","output":"1"}]',
        }
    )
    (tmp_path / "solution.py").write_text("print(input())")
    grade = grade_livecode(
        task, NativeEnvironment(tmp_path), tmp_path, evaluator=tmp_path / "upstream"
    )
    assert grade.status == "passed"
    assert "official LiveCodeBench" in grade.details["evaluator"]
