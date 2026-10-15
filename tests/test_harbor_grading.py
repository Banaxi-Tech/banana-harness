from pathlib import Path

import pytest

from banana_harness.grading import grade_harbor
from banana_harness.types import CommandResult, Task


class SyntheticRootfs:
    def __init__(self, reward, returncode=0):
        self.reward = reward
        self.returncode = returncode
        self.env = {}
        self.files_uploaded = []

    def execute(self, command, timeout=60, output_limit=100_000):
        if command == "bash /tests/test.sh":
            return CommandResult("verifier output", self.returncode)
        return CommandResult("", 0)

    def put(self, source, destination):
        self.files_uploaded.append((source, destination))

    def read(self, path):
        if path.endswith("reward.json") and self.reward is not None:
            return self.reward
        raise FileNotFoundError(path)


@pytest.mark.parametrize(
    "reward,returncode,status",
    [
        ('{"reward": 1}', 0, "passed"),
        ('{"reward": 0}', 0, "failed"),
        ('{"reward": 1}', 1, "verifier_error"),
        ('{"reward": -1}', 0, "verifier_error"),
        ('{"reward": "nan"}', 0, "verifier_error"),
        (None, 0, "verifier_error"),
    ],
)
def test_rewards_are_required_and_validated(tmp_path, reward, returncode, status):
    source = tmp_path / "task"
    source.mkdir()
    (source / "tests").mkdir()
    task = Task(
        "synthetic",
        "deepswe",
        "Task",
        kind="harbor",
        source_dir=str(source),
        grading={"verifier": {}},
    )
    environment = SyntheticRootfs(reward, returncode)
    grade = grade_harbor(task, environment, tmp_path)
    assert grade.status == status
    assert environment.files_uploaded == [(Path(source / "tests"), "/tests")]
