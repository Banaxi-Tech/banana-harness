from __future__ import annotations

import fcntl
import json
import platform
import shlex
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from banana_harness.agent import AgentConfig, BashAgent, Model
from banana_harness.environment import (
    Environment,
    NativeEnvironment,
    RootfsEnvironment,
    inspect_cached_image,
)
from banana_harness.grading import (
    apply_patch,
    export_patch,
    grade_command,
    grade_harbor,
    grade_livecode,
    grade_swebench,
)
from banana_harness.types import HarnessError, Task
from banana_harness.util import checked, digest, slug, write_json


@dataclass
class RunConfig:
    output: Path = Path("jobs/default")
    backend: str = "auto"
    cache: Path = Path(".banana/udocker")
    allow_downloads: bool = False
    resume: bool = True
    use_venv: bool = True
    rootfs_mode: str = "P1"
    livecode_evaluator: Path | None = None
    agent: AgentConfig = field(default_factory=AgentConfig)

    def __post_init__(self) -> None:
        self.output, self.cache = Path(self.output), Path(self.cache)
        if self.livecode_evaluator is not None:
            self.livecode_evaluator = Path(self.livecode_evaluator)
        if self.backend not in {"auto", "native", "rootfs"}:
            raise ValueError("backend must be auto, native, or rootfs")


class Runner:
    def __init__(
        self,
        model_factory: Callable[[], Model],
        config: RunConfig | None = None,
        *,
        model_identity: str | None = None,
    ):
        self.model_factory = model_factory
        self.config = config or RunConfig()
        # Include provider options (temperature, endpoint, etc.) in CLI resume identities.
        self.model_identity = model_identity

    def _backend(self, task: Task) -> str:
        backend = self.config.backend
        if backend == "auto":
            backend = "rootfs" if task.image else "native"
        if task.kind == "livecode" and backend != "native":
            raise HarnessError("LiveCodeBench currently uses the native Python evaluator.")
        if task.kind == "swebench" and backend != "rootfs":
            raise HarnessError(
                "Verified's official evaluation scripts require the pinned rootfs environment. "
                "Use --backend rootfs, or a custom native Task with your setup/test command."
            )
        if (
            backend == "native"
            and task.kind == "harbor"
            and not task.metadata.get("native_profile")
        ):
            raise HarnessError("Native Harbor runs require an explicit banana.toml profile.")
        return backend

    def _environment(
        self, task: Task, backend: str, directory: Path, logs: Path | None = None
    ) -> Environment:
        directory.mkdir(parents=True, exist_ok=True)
        if backend == "native":
            return NativeEnvironment.create(
                task,
                directory,
                allow_downloads=self.config.allow_downloads,
                use_venv=self.config.use_venv,
            )
        if not task.image:
            raise HarnessError("Rootfs runs require an image reference.")
        workdir = task.workdir
        if task.kind == "harbor" and not task.metadata.get("explicit_workdir"):
            workdir = None
        mounts = {logs: "/logs"} if logs else {}
        return RootfsEnvironment(
            task.image,
            self.config.cache,
            workdir=workdir,
            env=task.env,
            mounts=mounts,
            allow_downloads=self.config.allow_downloads,
            mode=self.config.rootfs_mode,
        )

    def run(self, tasks) -> dict[str, Any]:
        """Serial by default: notebook memory and shared cache use remain predictable."""
        output = self.config.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        results = []
        seen = set()
        with (output / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise HarnessError(f"Another job is writing {output}") from error
            for task in tasks:
                if task.id in seen:
                    raise HarnessError(f"Duplicate task ID: {task.id}")
                seen.add(task.id)
                results.append(self.run_task(task))
                self._summary(results)
            return self._summary(results)

    def _summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(results)
        passed = sum(r["status"] == "passed" for r in results)
        graded = sum(r.get("reward") is not None for r in results)
        summary = {
            "schema_version": 1,
            "tasks": total,
            "passed": passed,
            "graded": graded,
            "pass_rate": passed / total if total else None,
            "mean_reward": (sum(r.get("reward") or 0 for r in results) / total) if total else None,
            "cost": sum(r.get("cost", 0) for r in results),
            "cost_known": all(r.get("cost_known", False) for r in results),
            "official_score": False,
            "results": results,
        }
        write_json(self.config.output / "summary.json", summary)
        with (self.config.output / "predictions.jsonl").open("w") as file:
            for result in results:
                patch_path = result.get("patch_path")
                if patch_path:
                    file.write(
                        json.dumps(
                            {
                                "instance_id": result["task_id"],
                                "model_name_or_path": result["model"],
                                "model_patch": Path(patch_path).read_text(),
                            }
                        )
                        + "\n"
                    )
        return summary

    def run_task(self, task: Task) -> dict[str, Any]:
        model = self.model_factory()
        backend = self._backend(task)
        local_revision = None
        if task.repo and Path(task.repo).is_dir() and not task.base_commit:
            local_revision = checked(["git", "rev-parse", "HEAD"], cwd=Path(task.repo))
        configuration_key = digest(
            {
                "task": asdict(task),
                "agent": asdict(self.config.agent),
                "model": self.model_identity or model.name,
                "backend": backend,
                "venv": self.config.use_venv,
                "mode": self.config.rootfs_mode,
                "package_version": "0.1.0",
                "local_repo_revision": local_revision,
                "livecode_evaluator": (
                    digest(
                        (
                            self.config.livecode_evaluator
                            / "lcb_runner"
                            / "evaluation"
                            / "testing_util.py"
                        )
                        .read_bytes()
                        .hex()
                    )
                    if self.config.livecode_evaluator is not None
                    else None
                ),
            }
        )
        cached_metadata = (
            inspect_cached_image(task.image, self.config.cache)
            if backend == "rootfs" and task.image
            else None
        )
        key = digest({"configuration": configuration_key, "image_metadata": cached_metadata})
        task_directory = self.config.output.resolve() / slug(task.id)
        task_directory.mkdir(parents=True, exist_ok=True)
        result_file = task_directory / "result.json"
        if self.config.resume and result_file.exists():
            previous = json.loads(result_file.read_text())
            if previous.get("fingerprint") == key and previous["status"] in {"passed", "failed"}:
                return previous
        attempt = task_directory / ("attempt-" + uuid.uuid4().hex[:12])
        attempt.mkdir()
        logs = attempt / "logs"
        (logs / "verifier").mkdir(parents=True)
        (logs / "artifacts").mkdir()
        write_json(attempt / "task.json", task.public())
        start = time.monotonic()
        result: dict[str, Any] = {
            "task_id": task.id,
            "benchmark": task.benchmark,
            "fingerprint": key,
            "model": model.name,
            "backend": backend,
            "status": "running",
            "reward": None,
            "attempt": str(attempt),
            "cost": 0.0,
            "cost_known": True,
            "official_score": False,
            "host": {"system": platform.system(), "architecture": platform.machine()},
            "runtime_differences": [
                "Host kernel and resources",
                "No network isolation",
                "No security sandbox",
            ],
        }
        write_json(result_file, result)
        agent_environment = verifier_environment = None
        agent = None
        try:
            agent_environment = self._environment(task, backend, attempt / "agent", logs)
            result["environment"] = agent_environment.provenance
            if backend == "rootfs":
                result["fingerprint"] = digest(
                    {
                        "configuration": configuration_key,
                        "image_metadata": agent_environment.provenance.get("image_metadata"),
                    }
                )
            if backend == "rootfs" and task.kind in {"swebench", "harbor"}:
                agent_environment.execute(
                    "git config --global --add safe.directory "
                    + shlex.quote(agent_environment.workdir),
                    60,
                )
            base = task.base_commit or task.metadata.get("base_commit_hash")
            if task.kind != "livecode":
                revision = agent_environment.execute("git rev-parse HEAD", 60)
                if revision.returncode == 0:
                    base = base or revision.output.strip()
                elif task.kind in {"swebench"} or task.benchmark == "deepswe":
                    raise HarnessError(f"Task repository missing: {revision.output}")
            agent = BashAgent(model, agent_environment, self.config.agent)
            trajectory = agent.run(
                task.prompt,
                timeout=task.agent_timeout,
                checkpoint=lambda value: write_json(attempt / "trajectory.json", value),
            )
            write_json(attempt / "trajectory.json", trajectory)
            result.update(
                agent_status=trajectory["status"],
                cost=trajectory["cost"],
                cost_known=trajectory["cost_known"],
            )
            patch = ""
            if base:
                patch = export_patch(agent_environment, base, attempt)
                result["patch_path"] = str(attempt / "model.patch")
            separate = task.grading.get("verifier", {}).get("environment_mode") == "separate"
            if task.kind == "livecode":
                grade = grade_livecode(
                    task, agent_environment, attempt, evaluator=self.config.livecode_evaluator
                )
            elif task.kind == "swebench":
                verifier_environment = self._environment(task, backend, attempt / "verify")
                if patch:
                    apply_patch(verifier_environment, attempt / "model.patch")
                grade = grade_swebench(task, verifier_environment, attempt, patch)
            elif task.kind == "harbor" and separate:
                if backend != "rootfs":
                    raise HarnessError("Separate Harbor verifiers currently require rootfs.")
                # Collect hooks require committed work. Include edits/new files before collection.
                commit = agent_environment.execute(
                    "git add -A && git -c user.name=Banana -c user.email=banana@localhost "
                    "-c core.hooksPath=/dev/null commit --allow-empty -qm 'Agent submission'",
                    120,
                )
                if commit.returncode:
                    raise HarnessError(f"Could not commit submission: {commit.output}")
                hooks = task.grading["verifier"].get("collect", [])
                for hook in hooks:
                    collected = agent_environment.execute(
                        hook["command"],
                        hook.get("timeout_sec", 300),
                        output_limit=None,
                    )
                    if collected.returncode:
                        raise HarnessError(f"Verifier collect hook failed: {collected.output}")
                verifier_environment = self._environment(task, backend, attempt / "verify", logs)
                # The official DeepSWE grader applies the collected model.patch to the fresh repo.
                grade = grade_harbor(task, verifier_environment, attempt)
            elif task.kind == "harbor":
                if backend == "native":
                    verifier_environment = self._environment(task, backend, attempt / "verify")
                    if patch:
                        apply_patch(verifier_environment, attempt / "model.patch")
                    grade = grade_harbor(task, verifier_environment, attempt)
                else:
                    grade = grade_harbor(task, agent_environment, attempt)
            else:
                verifier_environment = self._environment(task, backend, attempt / "verify")
                if patch:
                    apply_patch(verifier_environment, attempt / "model.patch")
                grade = grade_command(task, verifier_environment, attempt)
            result.update(status=grade.status, reward=grade.reward, grading=grade.details)
        except KeyboardInterrupt:
            result["status"] = "cancelled"
            raise
        except Exception as error:
            result.update(status="error", error=f"{type(error).__name__}: {error}")
        finally:
            if agent is not None:
                result.update(cost=agent.cost, cost_known=agent.cost_known)
                write_json(
                    attempt / "trajectory.json", agent.snapshot(result.get("agent_status", "error"))
                )
            for environment in (verifier_environment, agent_environment):
                if environment:
                    try:
                        environment.close()
                    except Exception as error:
                        result.setdefault("cleanup_errors", []).append(str(error))
            result["duration"] = time.monotonic() - start
            write_json(result_file, result)
        return result
