from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import uuid
import venv
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from banana_harness.types import CommandResult, HarnessError, Task
from banana_harness.util import checked, child_env, run_process


class Environment(Protocol):
    workdir: str

    def execute(
        self, command: str, timeout: float = 60, output_limit: int | None = 100_000
    ) -> CommandResult: ...
    def put(self, source: Path, destination: str) -> None: ...
    def read(self, path: str) -> str: ...
    def close(self) -> None: ...


def inspect_cached_image(image: str, cache: Path) -> Any | None:
    """Read image metadata without downloading images or udocker execution tools."""
    executable = shutil.which("udocker")
    if not executable:
        return None
    command = [executable, "--quiet"]
    if os.geteuid() == 0:
        command.append("--allow-root")
    command += ["inspect", image]
    result = run_process(
        command,
        env=child_env({"UDOCKER_DIR": str(cache.resolve()), "UDOCKER_TARBALL": ""}),
        timeout=60,
    )
    if result.returncode:
        return None
    try:
        return json.loads(result.output)
    except json.JSONDecodeError:
        return result.output.strip()


def prepare_repository(task: Task, target: Path, allow_downloads: bool) -> None:
    if task.repo:
        local = Path(task.repo).expanduser()
        if not local.exists() and not allow_downloads:
            raise HarnessError(
                "Repository is remote. Supply a local checkout or --allow-downloads."
            )
        checked(["git", "clone", "--no-hardlinks", "--", task.repo, str(target)], timeout=600)
        if task.base_commit:
            checked(["git", "checkout", "--detach", task.base_commit], cwd=target)
        # No future branches, remote refs, tags, or dangling commits in the agent workspace.
        refs = checked(["git", "for-each-ref", "--format=%(refname)"], cwd=target)
        for ref in refs.splitlines():
            checked(["git", "update-ref", "-d", ref], cwd=target)
        if "origin" in checked(["git", "remote"], cwd=target).splitlines():
            checked(["git", "remote", "remove", "origin"], cwd=target)
        checked(["git", "reflog", "expire", "--expire=now", "--all"], cwd=target)
        checked(["git", "gc", "--prune=now"], cwd=target, timeout=600)
    else:
        target.mkdir(parents=True)
        checked(["git", "init", "-q"], cwd=target)
        checked(
            [
                "git",
                "-c",
                "user.name=Banana",
                "-c",
                "user.email=banana@localhost",
                "commit",
                "--allow-empty",
                "-qm",
                "Initial workspace",
            ],
            cwd=target,
        )
    checked(["git", "config", "core.hooksPath", "/dev/null"], cwd=target)
    checked(["git", "config", "user.name", "Banana Agent"], cwd=target)
    checked(["git", "config", "user.email", "banana@localhost"], cwd=target)


class NativeEnvironment:
    """Independent bash commands in a private working directory on the host."""

    def __init__(self, workspace: Path, env: dict[str, str] | None = None):
        self.workspace = workspace.resolve()
        self.workdir = str(self.workspace)
        self.env = child_env(env)
        self.env["GIT_CONFIG_GLOBAL"] = str(self.workspace.parent / "banana-gitconfig")
        self.provenance: dict[str, Any] = {"backend": "native", "python": sys.version}

    @classmethod
    def create(
        cls,
        task: Task,
        directory: Path,
        *,
        allow_downloads: bool = False,
        use_venv: bool = True,
    ) -> NativeEnvironment:
        workspace = directory / "workspace"
        prepare_repository(task, workspace, allow_downloads)
        result = cls(workspace, task.env)
        if use_venv:
            prefix = directory / "venv"
            venv.EnvBuilder(with_pip=True).create(prefix)
            result.env["VIRTUAL_ENV"] = str(prefix.resolve())
            result.env["PATH"] = str(prefix.resolve() / "bin") + os.pathsep + result.env["PATH"]
        for command in task.setup:
            execution = result.execute(command, task.verifier_timeout)
            if execution.returncode:
                raise HarnessError(f"Native setup failed: {command}\n{execution.output}")
        return result

    def execute(
        self, command: str, timeout: float = 60, output_limit: int | None = 100_000
    ) -> CommandResult:
        return run_process(
            ["bash", "--noprofile", "--norc", "-c", command],
            cwd=self.workspace,
            env=self.env,
            timeout=timeout,
            output_limit=output_limit,
        )

    def _path(self, destination: str) -> Path:
        path = (self.workspace / destination).resolve()
        if not path.is_relative_to(self.workspace):
            raise HarnessError(f"File path escapes native workspace: {destination}")
        return path

    def put(self, source: Path, destination: str) -> None:
        target = self._path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)

    def read(self, path: str) -> str:
        return self._path(path).read_text()

    def close(self) -> None:
        pass

    def get_template_variables(self) -> dict[str, str]:
        return {"cwd": self.workdir, "environment": "native"}


class RootfsEnvironment:
    """udocker P1/P2: extracted image + PRoot, without Docker, root, or a daemon."""

    def __init__(
        self,
        image: str,
        cache: Path,
        *,
        workdir: str | None = "/app",
        env: dict[str, str] | None = None,
        mounts: dict[Path, str] | None = None,
        allow_downloads: bool = False,
        mode: str = "P1",
        setup_timeout: float = 1800,
    ):
        if not sys.platform.startswith("linux"):
            raise HarnessError("The rootfs backend requires Linux. Use native on other platforms.")
        if mode not in {"P1", "P2"}:
            raise HarnessError("Only daemon-free PRoot modes P1 and P2 are supported.")
        executable = shutil.which("udocker")
        if not executable:
            raise HarnessError(
                "Install banana-harness[rootfs], then run udocker install explicitly."
            )
        self.cli = [executable, "--quiet"]
        if os.geteuid() == 0:
            self.cli.append("--allow-root")
        self.cache = cache.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        # udocker otherwise auto-downloads its execution tools, even for read-only commands.
        # An empty tarball disables that implicit network path; install tools explicitly first.
        self.driver_env = child_env({"UDOCKER_DIR": str(self.cache), "UDOCKER_TARBALL": ""})
        self.env = dict(env or {})
        self.mounts = dict(mounts or {})
        self.workdir = workdir or "/app"
        self.container = "banana-" + uuid.uuid4().hex
        self._created = False
        self.provenance: dict[str, Any] = {"backend": "rootfs", "mode": mode, "image": image}
        try:
            # inspect checks the local image cache; it does not pull.
            available = self._call(["inspect", image], setup_timeout)
            if available.returncode:
                if not allow_downloads:
                    raise HarnessError(
                        f"Image {image} is not cached. Enable --allow-downloads to pull it "
                        "or pre-populate the udocker cache. No download was attempted."
                    )
                self._checked(["pull", image], setup_timeout)
                available = self._call(["inspect", image], setup_timeout)
            try:
                image_metadata = json.loads(available.output)
                self.provenance["image_metadata"] = image_metadata
                if workdir is None and isinstance(image_metadata, dict):
                    config = image_metadata.get("config", image_metadata.get("Config", {}))
                    self.workdir = config.get("WorkingDir") or "/app"
            except json.JSONDecodeError:
                self.provenance["image_metadata"] = available.output.strip()
            self._checked(["create", "--name=" + self.container, image], setup_timeout)
            self._created = True
            self._checked(["setup", "--execmode=" + mode, self.container], setup_timeout)
            root = self._checked(["inspect", "-p", self.container], 60).splitlines()[-1]
            self.root = Path(root).resolve()
            if not self.root.is_dir() or not self.root.is_relative_to(self.cache):
                raise HarnessError(f"Unexpected udocker root filesystem: {root}")
            for destination in self.mounts.values():
                self._root_path(destination).mkdir(parents=True, exist_ok=True)
            # Fail at provisioning, rather than report a benchmark failure, if PRoot is blocked.
            self._checked_run("true", 60)
        except BaseException:
            self.close()
            raise

    def _call(
        self, args: list[str], timeout: float, output_limit: int | None = 100_000
    ) -> CommandResult:
        return run_process(
            self.cli + args, env=self.driver_env, timeout=timeout, output_limit=output_limit
        )

    def _checked(self, args: list[str], timeout: float) -> str:
        result = self._call(args, timeout)
        if result.returncode:
            raise HarnessError(f"udocker failed: {' '.join(args)}\n{result.output}")
        return result.output.strip()

    def execute(
        self, command: str, timeout: float = 60, output_limit: int | None = 100_000
    ) -> CommandResult:
        args = ["run", "--nobanner", "--user=root", "--containerauth", "--workdir=" + self.workdir]
        for path, destination in self.mounts.items():
            args.append(f"--volume={path.resolve()}:{destination}")
        for key, value in self.env.items():
            args.append(f"--env={key}={value}")
        # Never --hostenv or --bindhome: API credentials stay in the model driver.
        args += [self.container, "/bin/bash", "--noprofile", "--norc", "-c", command]
        return self._call(args, timeout, output_limit)

    def _checked_run(self, command: str, timeout: float) -> None:
        result = self.execute(command, timeout)
        if result.returncode:
            raise HarnessError(f"Rootfs execution failed: {command}\n{result.output}")

    def _root_path(self, destination: str) -> Path:
        relative = PurePosixPath(destination)
        if not relative.is_absolute():
            relative = PurePosixPath(self.workdir) / relative
        if ".." in relative.parts:
            raise HarnessError(f"Invalid rootfs path: {destination}")
        # Image symlinks are resolved by PRoot, not by the host filesystem.
        parts = list(relative.parts)
        if parts and parts[0] == "/":
            parts.pop(0)
        path = self.root
        for part in parts:
            path = path / part
            if path.is_symlink():
                raise HarnessError(f"Refusing host file access through image symlink: {path}")
        return path

    def put(self, source: Path, destination: str) -> None:
        target = self._root_path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if target.exists():
                raise HarnessError(
                    f"Refusing to merge a directory into existing rootfs path: {target}"
                )
            shutil.copytree(source, target, symlinks=True)
        else:
            shutil.copy2(source, target)

    def read(self, path: str) -> str:
        # Reads inside the emulated filesystem correctly handle its absolute symlinks.
        result = self.execute("cat -- " + shlex.quote(path), 60, output_limit=None)
        if result.returncode:
            raise FileNotFoundError(path)
        return result.output

    def close(self) -> None:
        if self._created:
            self._call(["rm", self.container], 120)
            self._created = False

    def get_template_variables(self) -> dict[str, str]:
        return {"cwd": self.workdir, "environment": "rootfs"}
