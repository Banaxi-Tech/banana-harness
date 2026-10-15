from __future__ import annotations

import json

from banana_harness.types import HarnessError, Task


def swebench_task(row: dict) -> Task:
    image = row.get("image") or (
        "swebench/sweb.eval.x86_64."
        + row["instance_id"].lower().replace("__", "_1776_")
        + ":latest"
    )
    return Task(
        id=row["instance_id"],
        benchmark="swebench-verified",
        prompt=row["problem_statement"],
        repo="https://github.com/" + row["repo"] + ".git",
        base_commit=row["base_commit"],
        image=image,
        workdir="/testbed",
        kind="swebench",
        env={
            "PATH": "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "CONDA_DEFAULT_ENV": "testbed",
        },
        grading={"row": row},
        metadata={
            "repo": row["repo"],
            "version": row["version"],
            "evaluation_library": "swebench==4.1.0",
        },
    )


def make_spec(task: Task):
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as error:
        raise HarnessError(
            "Install banana-harness[swebench] (pins the 4.1.0 evaluator)."
        ) from error
    return make_test_spec(task.grading["row"], namespace="swebench")


def parse_grade(task: Task, log_path: str, patch: str):
    from swebench.harness.grading import get_eval_report

    report = get_eval_report(
        make_spec(task),
        {"instance_id": task.id, "model_name_or_path": "banana", "model_patch": patch},
        log_path,
        include_tests_status=True,
    )[task.id]
    return report


def required_tests(task: Task) -> dict[str, list[str]]:
    row = task.grading["row"]
    return {
        key: json.loads(row[key]) if isinstance(row[key], str) else row[key]
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS")
    }
