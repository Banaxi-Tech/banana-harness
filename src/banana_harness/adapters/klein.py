"""Klein-Code-Bench rows: complete projects built in empty native workspaces."""

from __future__ import annotations

import json

from banana_harness.types import HarnessError, Task

SUBSETS = ("easy", "intermediate", "hard", "max")


def klein_task(row: dict) -> Task:
    if row.get("protocol_version", "1.0") != "1.0":
        raise HarnessError("Unsupported Klein-Code-Bench protocol version")
    subset = row["subset"]
    if subset not in SUBSETS:
        raise HarnessError(f"Unknown Klein-Code-Bench subset: {subset}")
    tests = row["acceptance_tests"]
    if isinstance(tests, str):
        tests = json.loads(tests)
    if not isinstance(tests, list) or not tests:
        raise HarnessError("Klein-Code-Bench requires nonempty acceptance scenarios")
    for trace in tests:
        if not isinstance(trace, dict) or not trace.get("sessions"):
            raise HarnessError("Invalid Klein-Code-Bench scenario")
        for session in trace["sessions"]:
            if (
                not isinstance(session, dict)
                or not isinstance(session.get("requests"), list)
                or not isinstance(session.get("expected"), list)
                or not session["requests"]
                or len(session["requests"]) != len(session["expected"])
            ):
                raise HarnessError("Invalid Klein-Code-Bench session")
    timeout = row.get("timeout_seconds", 10)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 60:
        raise HarnessError("Klein-Code-Bench session timeout must be in (0, 60]")
    return Task(
        id=row["task_id"],
        benchmark="klein-code-bench",
        kind="klein",
        prompt=row["prompt"],
        workdir=".",
        metadata={
            "subset": subset,
            "family": row.get("family"),
            "category": row.get("category"),
            "target_model_parameters": row.get("target_model_parameters"),
            "protocol_version": row.get("protocol_version", "1.0"),
            "evaluation_library": "klein-code-bench==0.1.0",
            "metric": "resolve@1 (all acceptance tests pass)",
        },
        grading={"acceptance_tests": tests, "timeout_seconds": timeout},
    )
