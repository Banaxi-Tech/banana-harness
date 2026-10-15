from __future__ import annotations

import json
import math
import re
import shlex
import sys
import tempfile
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path

from banana_harness import lcb_official_worker, lcb_worker
from banana_harness.adapters.livecode import decode_cases
from banana_harness.adapters.swebench import make_spec, parse_grade
from banana_harness.environment import Environment, NativeEnvironment
from banana_harness.types import Grade, HarnessError, Task
from banana_harness.util import child_env, digest, run_process, write_json


def export_patch(environment: Environment, base: str, directory: Path) -> str:
    revision = environment.execute(
        "git rev-parse --verify --end-of-options " + shlex.quote(base + "^{commit}"),
        60,
    )
    base = revision.output.strip()
    if revision.returncode or not re.fullmatch(r"[0-9a-f]{40,64}", base):
        raise HarnessError(f"Cannot resolve submission base revision: {revision.output}")
    filename = ".banana-model-" + uuid.uuid4().hex + ".patch"
    # Staging includes newly created files; diff against the pinned base includes agent commits.
    command = (
        "git add -A && git -c core.fileMode=false diff --cached --binary "
        + shlex.quote(base)
        + " -- > "
        + shlex.quote(filename)
    )
    result = environment.execute(command, 120)
    if result.returncode:
        raise HarnessError(f"Patch export failed:\n{result.output}")
    patch = environment.read(filename)
    environment.execute("rm -- " + shlex.quote(filename), 30)
    (directory / "model.patch").write_text(patch)
    return patch


def apply_patch(environment: Environment, patch_file: Path) -> None:
    filename = ".banana-replay-" + uuid.uuid4().hex + ".patch"
    environment.put(patch_file, filename)
    result = environment.execute("git apply --binary " + shlex.quote(filename), 120)
    environment.execute("rm -- " + shlex.quote(filename), 30)
    if result.returncode:
        raise HarnessError(f"Patch replay failed:\n{result.output}")


def grade_swebench(task: Task, environment: Environment, directory: Path, patch: str) -> Grade:
    spec = make_spec(task)
    script = directory / "eval.sh"
    script.write_text(spec.eval_script)
    environment.put(script, "/eval.sh")
    result = environment.execute("bash /eval.sh", task.verifier_timeout, output_limit=None)
    log = directory / "test_output.txt"
    log.write_text(result.output)
    if result.timed_out:
        return Grade("verifier_timeout", details={"log": str(log)})
    report = parse_grade(task, str(log), patch)
    return Grade("passed" if report["resolved"] else "failed", float(report["resolved"]), report)


def grade_harbor(task: Task, environment: Environment, directory: Path) -> Grade:
    if isinstance(environment, NativeEnvironment):
        if not task.test_command:
            raise HarnessError(
                "Native Harbor grading needs test_command in banana.toml. "
                "Use rootfs to run the original /tests/test.sh verifier."
            )
        if task.path and (task.path / "tests").is_dir():
            environment.put(task.path / "tests", ".banana-tests")
        return grade_command(task, environment, directory)
    tests = task.path / "tests"
    # Only the verifier phase gets held-out files, replacing any stale /tests from the image.
    result = environment.execute("rm -rf -- /tests && mkdir -p /logs/verifier", 60)
    if result.returncode:
        raise HarnessError(f"Could not prepare verifier paths: {result.output}")
    environment.put(tests, "/tests")
    verifier = task.grading["verifier"]
    environment.env.update(verifier.get("env", {}))
    result = environment.execute("bash /tests/test.sh", task.verifier_timeout, output_limit=None)
    (directory / "test_output.txt").write_text(result.output)
    if result.timed_out:
        return Grade("verifier_timeout")
    reward_data = None
    reward = None
    for filename in ("reward.json", "reward.txt"):
        try:
            text = environment.read("/logs/verifier/" + filename)
            (directory / filename).write_text(text)
            if filename.endswith("json"):
                reward_data = json.loads(text)
                value = reward_data.get("reward")
                if value is None and len(reward_data) == 1:
                    value = next(iter(reward_data.values()))
                reward = float(value) if value is not None else None
            else:
                reward = float(text.strip())
            if reward is not None:
                break
        except FileNotFoundError:
            continue
        except (ValueError, TypeError, AttributeError) as error:
            return Grade("verifier_error", details={"reason": f"Invalid reward: {error}"})
    if reward is None or not math.isfinite(reward) or not 0 <= reward <= 1:
        return Grade(
            "verifier_error",
            details={"reason": "Missing or invalid reward", "returncode": result.returncode},
        )
    if result.returncode and reward > 0:
        return Grade(
            "verifier_error",
            details={
                "reason": "Positive reward with failed verifier",
                "returncode": result.returncode,
            },
        )
    return Grade("passed" if reward == 1 else "failed", reward, reward_data or {})


def grade_command(task: Task, environment: Environment, directory: Path) -> Grade:
    if not task.test_command:
        return Grade("ungraded", details={"reason": "No test_command supplied"})
    result = environment.execute(task.test_command, task.verifier_timeout, output_limit=None)
    (directory / "test_output.txt").write_text(result.output)
    if result.timed_out:
        return Grade("verifier_timeout")
    return Grade(
        "passed" if result.returncode == 0 else "failed",
        float(result.returncode == 0),
        {"returncode": result.returncode},
    )


def _stdio_equal(actual: str, expected: str) -> bool:
    # Preserve line boundaries, ignore trailing/leading whitespace, compare numeric lines exactly.
    left = [line.strip() for line in actual.strip().splitlines()]
    right = [line.strip() for line in expected.strip().splitlines()]
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if a == b:
            continue
        try:
            if [Decimal(x) for x in a.split()] == [Decimal(x) for x in b.split()]:
                continue
        except InvalidOperation:
            pass
        return False
    return True


def grade_livecode(
    task: Task, environment: NativeEnvironment, directory: Path, evaluator: Path | None = None
) -> Grade:
    try:
        code = environment.read("solution.py")
    except FileNotFoundError:
        return Grade("failed", 0.0, {"reason": "No solution.py produced"})
    (directory / "solution.py").write_text(code)
    cases = task.grading["public"] + decode_cases(task.grading["private_encoded"])
    if not cases:
        return Grade("verifier_error", details={"reason": "No test cases"})
    if evaluator is not None:
        return _grade_livecode_official(task, code, cases, directory, evaluator)
    reports = []
    timeout = float(task.grading.get("case_timeout", 6))
    function = task.grading.get("func_name")
    deadline = time.monotonic() + task.verifier_timeout
    with tempfile.TemporaryDirectory(prefix="banana-lcb-") as temporary:
        worker_directory = Path(temporary)
        script = worker_directory / "solution.py"
        # Contest evaluators expose standard imports for both functional and stdin tasks.
        # Separate exec calls preserve __future__ imports in submitted code.
        script.write_text(
            "from builtins import compile as _banana_compile\n"
            f"exec(_banana_compile({lcb_worker.PRELUDE!r}, '<prelude>', 'exec'))\n"
            f"exec(_banana_compile({code!r}, 'solution.py', 'exec'))\n"
        )
        for index, case in enumerate(cases):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return Grade("verifier_timeout", details={"completed_cases": index})
            case_timeout = min(timeout, remaining)
            kind = case.get("testtype", "functional" if function else "stdin")
            if kind not in {"stdin", "functional"}:
                raise HarnessError(f"Unknown LiveCodeBench test type: {kind}")
            if kind == "functional":
                if not function:
                    return Grade("verifier_error", details={"reason": "Missing func_name"})
                request, response = (
                    worker_directory / "request.json",
                    worker_directory / "result.json",
                )
                response.unlink(missing_ok=True)
                write_json(request, {"code": code, "func_name": function, "input": case["input"]})
                result = run_process(
                    [
                        sys.executable,
                        "-I",
                        str(Path(lcb_worker.__file__).resolve()),
                        str(request),
                        str(response),
                    ],
                    cwd=worker_directory,
                    env=child_env(),
                    timeout=case_timeout,
                )
                value = json.loads(response.read_text()) if response.exists() else {}
                expected = json.loads(case["output"])
                passed = result.returncode == 0 and "value" in value and value["value"] == expected
                error = value.get("error")
            else:
                result = run_process(
                    [sys.executable, "-I", str(script)],
                    cwd=worker_directory,
                    env=child_env(),
                    timeout=case_timeout,
                    input_data=case["input"].encode(),
                )
                passed = result.returncode == 0 and _stdio_equal(result.output, case["output"])
                error = None
            reports.append(
                {
                    "case": index,
                    "public": index < len(task.grading["public"]),
                    "passed": passed,
                    "timed_out": result.timed_out,
                    "returncode": result.returncode,
                    "error": error,
                }
            )
    write_json(directory / "test_cases.json", reports)
    passed = all(report["passed"] for report in reports)
    return Grade(
        "passed" if passed else "failed",
        float(passed),
        {
            "cases": len(reports),
            "passed": sum(r["passed"] for r in reports),
            "evaluator": "banana host evaluator; not official LiveCodeBench harness",
        },
    )


def _grade_livecode_official(
    task: Task, code: str, cases: list[dict], directory: Path, evaluator: Path
) -> Grade:
    module = evaluator.resolve() / "lcb_runner" / "evaluation" / "testing_util.py"
    if not module.is_file():
        raise HarnessError(f"LiveCodeBench evaluator not found: {module}")
    sample = {
        "input_output": json.dumps(
            {
                "inputs": [c["input"] for c in cases],
                "outputs": [c["output"] for c in cases],
                "fn_name": task.grading.get("func_name"),
            }
        )
    }
    timeout = float(task.grading.get("case_timeout", 6))
    with tempfile.TemporaryDirectory(prefix="banana-lcb-official-") as temporary:
        workdir = Path(temporary)
        request, response = workdir / "request.json", workdir / "result.json"
        write_json(
            request, {"module": str(module), "sample": sample, "code": code, "timeout": timeout}
        )
        result = run_process(
            [
                sys.executable,
                "-I",
                str(Path(lcb_official_worker.__file__).resolve()),
                str(request),
                str(response),
            ],
            cwd=workdir,
            env=child_env(),
            timeout=min(task.verifier_timeout, (timeout + 1) * len(cases) + 5),
        )
        (directory / "test_output.txt").write_text(result.output)
        if result.timed_out:
            return Grade("verifier_timeout")
        if result.returncode or not response.exists():
            return Grade("verifier_error", details={"reason": result.output})
        data = json.loads(response.read_text())
    write_json(directory / "test_cases.json", data)
    if "error" in data:
        return Grade("verifier_error", details=data)
    statuses = data["results"]
    passed = len(statuses) == len(cases) and all(
        isinstance(value, (bool, int, float)) and value > 0 for value in statuses
    )
    return Grade(
        "passed" if passed else "failed",
        float(passed),
        {
            "evaluator": "official LiveCodeBench testing_util.py (local checkout)",
            "evaluator_sha256": digest(module.read_bytes().hex()),
            "test_results": statuses,
            "metadata": data.get("metadata"),
        },
    )
