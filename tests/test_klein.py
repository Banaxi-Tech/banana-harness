import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from banana_harness.adapters import load_tasks
from banana_harness.adapters.klein import klein_task
from banana_harness.agent import FINISH, ScriptedModel
from banana_harness.runner import RunConfig, Runner
from banana_harness.types import HarnessError


def row():
    return {
        "task_id": "kcb-easy-synthetic",
        "subset": "easy",
        "family": "synthetic",
        "prompt": "Build main.py from scratch; echo every JSON input unchanged.",
        "acceptance_tests": json.dumps(
            [
                {
                    "name": "private-protocol",
                    "sessions": [
                        {
                            "requests": [{"secret_case": "evaluator-only"}, {"x": [1, None]}],
                            "expected": [{"secret_case": "evaluator-only"}, {"x": [1, None]}],
                        }
                    ],
                },
            ]
        ),
    }


def test_adapter_keeps_tests_out_of_agent_materials():
    task = klein_task(row())
    assert task.kind == "klein"
    assert task.repo is None and task.image is None
    assert "evaluator-only" not in task.prompt
    assert "evaluator-only" not in json.dumps(task.public())
    assert task.grading["acceptance_tests"][0]["name"] == "private-protocol"


@pytest.mark.parametrize("config", [None, "default", "easy", "intermediate", "hard", "max"])
def test_remote_subset_configuration_and_revision(monkeypatch, config):
    loader = Mock(return_value=[row()])
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=loader))
    tasks = list(
        load_tasks("klein-code-bench", allow_downloads=True, dataset_config=config, revision="pin")
    )
    loader.assert_called_once_with(
        "klein-ml/klein-code-bench",
        name=config or "default",
        split="test",
        revision="pin",
        streaming=True,
    )
    assert tasks[0].metadata["dataset_revision"] == "pin"


def test_local_rows_and_remote_opt_in(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(row()) + "\n")
    assert next(load_tasks("klein-code-bench", path)).kind == "klein"
    with pytest.raises(HarnessError, match="allow-downloads"):
        list(load_tasks("klein-code-bench"))


@pytest.mark.parametrize("options", [{"split": "train"}, {"dataset_config": "other"}])
def test_invalid_remote_configuration(monkeypatch, options):
    loader = Mock()
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=loader))
    with pytest.raises(HarnessError):
        list(load_tasks("klein-code-bench", allow_downloads=True, **options))
    loader.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"subset": "other"},
        {"acceptance_tests": "[]"},
        {"timeout_seconds": 0},
        {"timeout_seconds": True},
    ],
)
def test_reject_invalid_rows(change):
    with pytest.raises(HarnessError):
        klein_task(dict(row(), **change))


def test_rejects_container_backend(tmp_path):
    runner = Runner(lambda: ScriptedModel([]), RunConfig(output=tmp_path, backend="rootfs"))
    with pytest.raises(HarnessError, match="native"):
        runner._backend(klein_task(row()))


def test_missing_evaluator_has_actionable_error(monkeypatch, tmp_path):
    from banana_harness.grading import grade_klein

    monkeypatch.setitem(sys.modules, "klein_code_bench.runner", None)
    with pytest.raises(HarnessError, match=r"banana-harness\[klein\]"):
        grade_klein(klein_task(row()), SimpleNamespace(workspace=tmp_path), tmp_path)


def test_complete_multifile_project_fresh_verifier_and_fixed_denominator(tmp_path):
    pytest.importorskip("klein_code_bench")
    replies = [
        "```bash\n"
        "test ! -e main.py && test ! -e helper.py\n"
        "cat > helper.py <<'PYCODE'\nimport json\n"
        "def echo(line): return json.dumps(json.loads(line))\nPYCODE\n"
        "cat > main.py <<'PYCODE'\nimport sys\nfrom helper import echo\n"
        "for line in sys.stdin:\n print(echo(line),flush=True)\nPYCODE\n"
        "```",
        f"```bash\n{FINISH}\n```",
    ]
    runner = Runner(lambda: ScriptedModel(replies), RunConfig(output=tmp_path, use_venv=False))
    summary = runner.run([klein_task(row())])
    assert summary["passed"] == 1, summary
    assert summary["klein_code_bench"]["subsets"]["easy"]["resolution_rate"] == 1 / 65
    assert summary["klein_code_bench"]["resolution_rate"] == 1 / 260
    assert not summary["klein_code_bench"]["complete"]
    result = summary["results"][0]
    attempt = Path(result["attempt"])
    assert (attempt / "verify/workspace/helper.py").exists()
    assert json.loads((attempt / "klein_acceptance.json").read_text())["resolved"]
    assert "evaluator-only" not in (attempt / "task.json").read_text()
    assert "evaluator-only" not in (attempt / "trajectory.json").read_text()
    assert runner.run([klein_task(row())])["results"][0]["attempt"] == str(attempt)


def test_missing_project_fails_not_ungraded(tmp_path):
    pytest.importorskip("klein_code_bench")
    runner = Runner(
        lambda: ScriptedModel([f"```bash\n{FINISH}\n```"]),
        RunConfig(output=tmp_path, use_venv=False),
    )
    result = runner.run([klein_task(row())])["results"][0]
    assert result["status"] == "failed"
    assert result["reward"] == 0


def test_actual_max_task_survives_harness_patch_replay(tmp_path):
    bench = pytest.importorskip("klein_code_bench.runner")
    from klein_code_bench.reference.export import export_reference

    actual = next(bench.load_tasks("max"))
    reference = tmp_path / "trusted-reference"
    export_reference(actual["family"], "max", reference)
    source = (reference / "main.py").read_text()
    replies = [
        f"```bash\ncat > main.py <<'TRUSTED_REFERENCE'\n{source}\nTRUSTED_REFERENCE\n```",
        f"```bash\n{FINISH}\n```",
    ]
    result = Runner(
        lambda: ScriptedModel(replies), RunConfig(output=tmp_path / "run", use_venv=False)
    ).run([klein_task(actual)])
    assert result["passed"] == 1, result
    assert result["results"][0]["grading"]["assertions_total"] == actual["num_assertions"]
