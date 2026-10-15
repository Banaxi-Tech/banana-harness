import json
from pathlib import Path

import pytest

from banana_harness.agent import FINISH, AgentConfig, ScriptedModel
from banana_harness.cli import main, smoke
from banana_harness.runner import RunConfig, Runner
from banana_harness.types import Task


def test_native_patch_export_fresh_verifier_and_resume(tmp_path):
    config = RunConfig(output=tmp_path, backend="native", use_venv=False)
    replies = ["```bash\nprintf fixed > new-file.txt\n```", f"```bash\n{FINISH}\n```"]
    calls = []

    def factory():
        calls.append(True)
        return ScriptedModel(replies)

    task = Task("patch-task", "custom", "Fix", test_command="test -f new-file.txt")
    runner = Runner(factory, config)
    first = runner.run([task])
    assert first["passed"] == 1
    result = first["results"][0]
    assert "new-file.txt" in Path(result["patch_path"]).read_text()
    assert (Path(result["attempt"]) / "verify/workspace/new-file.txt").read_text() == "fixed"
    second = runner.run([task])
    assert second["results"][0]["attempt"] == result["attempt"]
    assert (
        len(
            list(
                (tmp_path / next(p.name for p in tmp_path.iterdir() if p.is_dir())).glob(
                    "attempt-*"
                )
            )
        )
        == 1
    )
    predictions = [json.loads(x) for x in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert predictions[0]["instance_id"] == "patch-task"
    task.prompt = "A different task"
    third = runner.run([task])
    assert third["results"][0]["attempt"] != result["attempt"]


def test_model_identity_affects_resume(tmp_path):
    task = Task("id", "custom", "Task", test_command="true")
    config = RunConfig(output=tmp_path, use_venv=False)

    def factory():
        return ScriptedModel([f"```bash\n{FINISH}\n```"])

    first = Runner(factory, config, model_identity="model-v1").run([task])
    second = Runner(factory, config, model_identity="model-v2").run([task])
    assert first["results"][0]["fingerprint"] != second["results"][0]["fingerprint"]


def test_grading_failure_and_verifier_timeout(tmp_path):
    def factory():
        return ScriptedModel([f"```bash\n{FINISH}\n```"])

    config = RunConfig(output=tmp_path, use_venv=False)
    tasks = [
        Task("fail", "custom", "Task", test_command="false"),
        Task("timeout", "custom", "Task", test_command="sleep 20", verifier_timeout=0.05),
    ]
    summary = Runner(factory, config).run(tasks)
    assert [r["status"] for r in summary["results"]] == ["failed", "verifier_timeout"]
    assert summary["passed"] == 0
    assert summary["graded"] == 1
    assert summary["pass_rate"] == 0


def test_model_error_is_recorded_and_retryable(tmp_path):
    config = RunConfig(output=tmp_path, use_venv=False, agent=AgentConfig(max_steps=1))
    task = Task("error", "custom", "Task", test_command="true")
    summary = Runner(lambda: ScriptedModel([]), config).run([task])
    assert summary["results"][0]["status"] == "error"
    assert "ran out" in summary["results"][0]["error"]


def test_smoke_and_doctor_no_api_key_or_dataset(tmp_path, capsys):
    assert smoke(tmp_path)["passed"] == 1
    assert main(["doctor", "--json"]) == 0
    assert '"downloads_attempted": false' in capsys.readouterr().out


def test_random_subset_matches_preview_across_three_runs(tmp_path, capsys):
    source = tmp_path / "tasks.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"task-{i}",
                    "prompt": "Finish",
                    "test_command": "true",
                }
            )
            for i in range(25)
        )
    )
    selection = ["--source", str(source), "--random-limit", "3", "--seed", "42"]
    assert main(["list", "custom", *selection]) == 0
    preview = [json.loads(line)["id"] for line in capsys.readouterr().out.splitlines()]
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps([f"```bash\n{FINISH}\n``` "]))
    for model in ("a", "b", "c"):
        output = tmp_path / model
        assert (
            main(
                [
                    "run",
                    "custom",
                    *selection,
                    "--replay",
                    str(replay),
                    "--no-venv",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        summary = json.loads((output / "summary.json").read_text())
        assert [result["task_id"] for result in summary["results"]] == preview
    capsys.readouterr()
    assert main(["list", "custom", "--source", str(source)]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 20


def test_cli_random_and_prefix_limits_are_exclusive():
    with pytest.raises(SystemExit) as error:
        main(
            [
                "list",
                "custom",
                "--source",
                "unused",
                "--limit",
                "3",
                "--random-limit",
                "3",
            ]
        )
    assert error.value.code == 2
