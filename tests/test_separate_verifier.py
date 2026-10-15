import shutil

from banana_harness.agent import FINISH, ScriptedModel
from banana_harness.environment import NativeEnvironment
from banana_harness.runner import RunConfig, Runner
from banana_harness.types import Task
from banana_harness.util import checked


def test_separate_harbor_collection_and_fresh_verifier(tmp_path, monkeypatch):
    """Exercise the full collection flow using a synthetic filesystem, no image runtime."""
    repository = tmp_path / "repository"
    repository.mkdir()
    checked(["git", "init", "-q"], cwd=repository)
    (repository / "value.txt").write_text("base")
    checked(["git", "add", "."], cwd=repository)
    checked(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-qm", "base"],
        cwd=repository,
    )
    base = checked(["git", "rev-parse", "HEAD"], cwd=repository)
    source = tmp_path / "task"
    (source / "tests").mkdir(parents=True)
    (source / "tests/test.sh").write_text(
        "set -e\ncd /app\ngit apply /logs/artifacts/model.patch\n"
        'test "$(cat value.txt)" = fixed\n'
        "printf '{\"reward\":1}' > /logs/verifier/reward.json\n"
    )
    task = Task(
        "synthetic-separate",
        "deepswe",
        "Change value.txt to fixed",
        kind="harbor",
        image="synthetic:image",
        repo=str(repository),
        base_commit=base,
        source_dir=str(source),
        grading={
            "verifier": {
                "environment_mode": "separate",
                "collect": [
                    {
                        "command": f"cd /app && git diff --binary {base} HEAD "
                        "> /logs/artifacts/model.patch"
                    }
                ],
            }
        },
    )
    created = []

    class SyntheticEnvironment:
        def __init__(self, directory, logs):
            self.native = NativeEnvironment.create(task, directory, use_venv=False)
            self.root = directory.resolve()
            self.logs = logs
            self.env = self.native.env
            self.workdir = "/app"
            self.provenance = {"backend": "synthetic-rootfs"}
            self.was_pristine = (self.native.workspace / "value.txt").read_text() == "base"

        def _path(self, path):
            if path.startswith("/logs"):
                return self.logs / path.removeprefix("/logs/")
            if path.startswith("/tests"):
                return self.root / "tests" / path.removeprefix("/tests/")
            return self.native.workspace / path

        def execute(self, command, timeout=60, output_limit=100_000):
            command = (
                command.replace("/app", str(self.native.workspace))
                .replace("/logs", str(self.logs))
                .replace("/tests", str(self.root / "tests"))
            )
            # Translate only this synthetic test's own verifier script; real code is not copied.
            verifier = self.root / "tests/test.sh"
            if command.startswith("bash ") and verifier.is_file():
                translated = (
                    verifier.read_text()
                    .replace("/app", str(self.native.workspace))
                    .replace("/logs", str(self.logs))
                )
                verifier.write_text(translated)
            return self.native.execute(command, timeout, output_limit)

        def put(self, source, destination):
            target = self.root / "tests" if destination == "/tests" else self._path(destination)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

        def read(self, path):
            return self._path(path).read_text()

        def close(self):
            pass

    class SyntheticRunner(Runner):
        def _environment(self, task, backend, directory, logs=None):
            directory.mkdir(parents=True)
            environment = SyntheticEnvironment(directory, logs)
            created.append(environment)
            return environment

    monkeypatch.setattr("banana_harness.runner.inspect_cached_image", lambda *args: None)
    replies = ["```bash\nprintf fixed > value.txt\n```", f"```bash\n{FINISH}\n```"]
    runner = SyntheticRunner(
        lambda: ScriptedModel(replies), RunConfig(output=tmp_path / "job", use_venv=False)
    )
    summary = runner.run([task])
    assert summary["passed"] == 1, summary
    assert len(created) == 2
    assert all(environment.was_pristine for environment in created)
    assert not (created[0].root / "tests").exists()
    assert (created[1].root / "tests/test.sh").exists()
    assert "fixed" in (created[0].logs / "artifacts/model.patch").read_text()
