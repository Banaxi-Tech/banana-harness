import json
from pathlib import Path

import pytest

from banana_harness.environment import NativeEnvironment, RootfsEnvironment, prepare_repository
from banana_harness.types import CommandResult, HarnessError, Task
from banana_harness.util import child_env, run_process


def test_timeout_kills_process_group(tmp_path):
    marker = tmp_path / "should-not-exist"
    result = run_process(["bash", "-c", f"(sleep 0.3; touch {marker}) & wait"], timeout=0.05)
    assert result.timed_out
    assert result.returncode == 124
    run_process(["sleep", "0.4"])
    assert not marker.exists()


def test_background_children_are_cleaned(tmp_path):
    marker = tmp_path / "background"
    run_process(["bash", "-c", f"(sleep 0.3; touch {marker}) &"])
    run_process(["sleep", "0.4"])
    assert not marker.exists()


def test_large_output_and_stdin():
    result = run_process(["python", "-c", "print('a'*200000)"], output_limit=1000)
    assert len(result.output) < 1100
    assert "omitted" in result.output
    result = run_process(["cat"], input_data=b"stdin works")
    assert result.output == "stdin works"


def test_native_file_access_does_not_escape(tmp_path):
    environment = NativeEnvironment(tmp_path)
    with pytest.raises(HarnessError):
        environment.read("../outside")


def test_credentials_not_inherited(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    assert "OPENAI_API_KEY" not in child_env()
    assert "HF_TOKEN" not in child_env()


def test_remote_native_repo_requires_opt_in(tmp_path, monkeypatch):
    def no_process(*args, **kwargs):
        pytest.fail("Network-capable command should not be invoked")

    monkeypatch.setattr("banana_harness.environment.checked", no_process)
    with pytest.raises(HarnessError, match="allow-downloads"):
        prepare_repository(
            Task("id", "custom", "prompt", repo="https://example.org/repo"),
            tmp_path / "workspace",
            False,
        )


def fake_udocker(monkeypatch, cache, available=True):
    calls = []
    root = cache / "containers" / "synthetic" / "ROOT"
    (root / "app").mkdir(parents=True)

    def execute(argv, **kwargs):
        assert kwargs["env"]["UDOCKER_TARBALL"] == ""
        assert "OPENAI_API_KEY" not in kwargs["env"]
        calls.append(argv)
        if "inspect" in argv and "-p" in argv:
            return CommandResult(str(root) + "\n", 0)
        if "inspect" in argv:
            return CommandResult(
                json.dumps({"config": {"WorkingDir": "/app"}}), 0 if available else 1
            )
        return CommandResult("", 0)

    monkeypatch.setattr("banana_harness.environment.shutil.which", lambda _: "/fake/udocker")
    monkeypatch.setattr("banana_harness.environment.run_process", execute)
    return calls, root


def test_rootfs_uses_proot_and_never_auto_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls, root = fake_udocker(monkeypatch, tmp_path)
    environment = RootfsEnvironment("synthetic:image", tmp_path)
    environment.execute("ls")
    assert any("--execmode=P1" in argv for argv in calls)
    assert not any("pull" in argv or "install" in argv for argv in calls)
    assert not any("--hostenv" in argv or "--bindhome" in argv for argv in calls)
    source = tmp_path / "patch"
    source.write_text("patch")
    environment.put(source, "replay.patch")
    assert (root / "app" / "replay.patch").read_text() == "patch"
    environment.close()
    assert "rm" in calls[-1]


def test_missing_image_does_not_pull(tmp_path, monkeypatch):
    calls, _ = fake_udocker(monkeypatch, tmp_path, available=False)
    with pytest.raises(HarnessError, match="No download was attempted"):
        RootfsEnvironment("synthetic:image", tmp_path)
    assert not any("pull" in argv for argv in calls)


def test_rootfs_rejects_host_symlink_traversal(tmp_path, monkeypatch):
    _, root = fake_udocker(monkeypatch, tmp_path)
    environment = RootfsEnvironment("synthetic:image", tmp_path)
    (root / "escape").symlink_to("/tmp")
    with pytest.raises(HarnessError, match="symlink"):
        environment.put(Path(__file__), "/escape/file")
    environment.close()
