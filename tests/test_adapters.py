import base64
import json
import pickle
import random
import sys
import zlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from banana_harness.adapters import load_tasks, select_tasks
from banana_harness.adapters.livecode import decode_cases, livecode_task
from banana_harness.adapters.swebench import swebench_task
from banana_harness.download import fetch
from banana_harness.types import HarnessError, Task


def test_dataset_access_opt_in():
    with pytest.raises(HarnessError, match="allow-downloads"):
        list(load_tasks("swebench-verified"))


@pytest.mark.parametrize("revision", [None, "abc123", "refs/pr/7"])
def test_livecode_remote_loads_all_v6_files_without_dataset_script(monkeypatch, revision):
    load_dataset = Mock(
        return_value=[
            {
                "question_id": "one",
                "question_content": "Solve",
                "metadata": "{}",
                "private_test_cases": "private-encoded",
            }
        ]
    )
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    tasks = list(load_tasks("livecodebench-v6", allow_downloads=True, revision=revision))
    expected_revision = {None: "main", "abc123": "abc123", "refs/pr/7": "refs%2Fpr%2F7"}[revision]
    base = (
        "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/"
        + expected_revision
    )
    load_dataset.assert_called_once_with(
        "json",
        data_files={
            "test": [
                f"{base}/test.jsonl",
                f"{base}/test2.jsonl",
                f"{base}/test3.jsonl",
                f"{base}/test4.jsonl",
                f"{base}/test5.jsonl",
                f"{base}/test6.jsonl",
            ]
        },
        split="test",
        streaming=True,
    )
    assert tasks[0].metadata["dataset_revision"] == revision
    assert tasks[0].metadata["dataset_config"] == "release_v6"
    assert tasks[0].grading["private_encoded"] == "private-encoded"
    assert "private-encoded" not in tasks[0].prompt


@pytest.mark.parametrize("options", [{"split": "train"}, {"dataset_config": "release_v5"}])
def test_livecode_remote_rejects_other_releases_or_splits(monkeypatch, options):
    load_dataset = Mock()
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    with pytest.raises(HarnessError, match="only"):
        list(load_tasks("livecodebench-v6", allow_downloads=True, **options))
    load_dataset.assert_not_called()


def test_verified_remote_loading_preserves_revision_and_config(monkeypatch):
    load_dataset = Mock(return_value=[])
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    assert (
        list(
            load_tasks(
                "swebench-verified",
                allow_downloads=True,
                revision="pinned",
                dataset_config="default",
            )
        )
        == []
    )
    load_dataset.assert_called_once_with(
        "princeton-nlp/SWE-bench_Verified",
        name="default",
        split="test",
        revision="pinned",
        streaming=True,
    )


def test_fetch_opt_in(tmp_path):
    with pytest.raises(HarnessError, match="allow-downloads"):
        fetch("deepswe", tmp_path / "tasks")
    assert not (tmp_path / "tasks").exists()


def test_verified_image_and_hidden_inputs():
    task = swebench_task(
        {
            "instance_id": "org__repo-1",
            "repo": "org/repo",
            "version": "1",
            "base_commit": "a" * 40,
            "problem_statement": "fix bug",
            "test_patch": "hidden",
        }
    )
    assert task.image == "swebench/sweb.eval.x86_64.org_1776_repo-1:latest"
    assert "hidden" not in json.dumps(task.public())
    assert "hidden" not in task.prompt


def test_real_lcb_encoding_and_prompt_separation():
    cases = [{"input": "2", "output": "3", "testtype": "stdin"}]
    encoded = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()
    assert decode_cases(encoded) == cases
    task = livecode_task(
        {
            "question_id": "1",
            "question_content": "Solve",
            "metadata": "{}",
            "private_test_cases": encoded,
            "public_test_cases": "[]",
        }
    )
    assert encoded not in task.prompt
    assert encoded not in json.dumps(task.public())


def test_pickle_code_execution_rejected():
    class Dangerous:
        def __reduce__(self):
            return eval, ("1+1",)

    encoded = base64.b64encode(zlib.compress(pickle.dumps(Dangerous()))).decode()
    with pytest.raises(HarnessError, match="executable objects"):
        decode_cases(encoded)


def test_harbor_metadata_limits_and_separate_verifier(tmp_path):
    task = tmp_path / "synthetic"
    task.mkdir()
    (task / "task.toml").write_text("""
[environment]
docker_image = "synthetic:image"
memory_mb = 100
[agent]
timeout_sec = 23
[verifier]
environment_mode = "separate"
timeout_sec = 42
[[verifier.collect]]
command = "git diff > /logs/artifacts/model.patch"
""")
    (task / "instruction.md").write_text("Implement requested behavior")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("echo heldout")
    loaded = list(load_tasks("deepswe", tmp_path))[0]
    assert loaded.agent_timeout == 23
    assert loaded.verifier_timeout == 42
    assert loaded.grading["verifier"]["environment_mode"] == "separate"
    assert "heldout" not in loaded.prompt


def test_no_silent_dockerfile_execution(tmp_path):
    (tmp_path / "task.toml").write_text("[environment]\n")
    (tmp_path / "instruction.md").write_text("Task")
    with pytest.raises(HarnessError, match="Dockerfiles are not executed"):
        list(load_tasks("terminal-bench-2.1", tmp_path))


def test_selection_checks_missing_and_duplicates():
    with pytest.raises(HarnessError, match="not found"):
        list(select_tasks([Task("a", "custom", "prompt")], ids={"missing"}))
    with pytest.raises(HarnessError, match="Duplicate"):
        list(select_tasks([Task("a", "custom", "prompt"), Task("a", "custom", "prompt")]))


def test_random_subset_is_reproducible_independent_of_order_and_global_rng():
    tasks = [Task(str(i), "custom", "prompt") for i in range(50)]
    state = random.getstate()
    first = list(select_tasks(iter(tasks), random_limit=10, seed=42))
    second = list(select_tasks(reversed(tasks), random_limit=10, seed=42))
    assert random.getstate() == state
    assert [task.id for task in first] == [task.id for task in second]
    assert len({task.id for task in first}) == 10
    assert first != tasks[:10]
    assert first != list(select_tasks(tasks, random_limit=10, seed=43))


def test_random_subset_filters_ids_and_handles_short_or_empty_sources():
    tasks = [Task(str(i), "custom", "prompt") for i in range(10)]
    selected = list(select_tasks(tasks, ids={"1", "5", "9"}, random_limit=2))
    assert len(selected) == 2
    assert {task.id for task in selected} <= {"1", "5", "9"}
    assert {task.id for task in select_tasks(tasks, random_limit=100)} == {
        task.id for task in tasks
    }
    assert list(select_tasks([], random_limit=3)) == []


def test_random_subset_validates_limits_and_entire_source():
    tasks = [Task("a", "custom", "prompt"), Task("a", "custom", "duplicate")]
    with pytest.raises(HarnessError, match="mutually exclusive"):
        list(select_tasks([], limit=1, random_limit=1))
    with pytest.raises(HarnessError, match="positive"):
        list(select_tasks([], random_limit=0))
    with pytest.raises(HarnessError, match="Duplicate"):
        list(select_tasks(tasks, random_limit=1))
    with pytest.raises(HarnessError, match="not found"):
        list(select_tasks(tasks[:1], ids={"missing"}, random_limit=1))


def test_default_limit_still_selects_source_prefix():
    tasks = [Task(str(i), "custom", "prompt") for i in range(10)]
    assert list(select_tasks(tasks, limit=3)) == tasks[:3]
