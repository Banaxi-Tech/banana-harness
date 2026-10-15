from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from banana_harness.types import HarnessError, Task
from banana_harness.util import digest, read_toml


def load_harbor(source: Path, benchmark: str) -> Iterator[Task]:
    if (source / "task.toml").is_file():
        files = [source / "task.toml"]
    else:
        files = sorted(source.glob("*/task.toml"))
    if not files:
        raise HarnessError(f"No Harbor tasks in {source}. Point --source at the tasks directory.")
    provenance = {}
    for ancestor in [source, *list(source.parents)[:3]]:
        manifest = ancestor / "banana-source.json"
        if manifest.is_file():
            provenance = json.loads(manifest.read_text())
            if provenance.get("benchmark") != benchmark:
                raise HarnessError("Fetched source manifest disagrees with selected benchmark.")
            break
    for file in files:
        directory = file.parent
        data = read_toml(file)
        environment = data.get("environment", {})
        verifier = data.get("verifier", {})
        metadata = data.get("metadata", {})
        profile_file = directory / "banana.toml"
        profile = read_toml(profile_file) if profile_file.exists() else {}
        dockerfile = directory / "environment" / "Dockerfile"
        workdir = "/app"
        explicit_workdir = False
        if dockerfile.exists():
            matches = re.findall(r"^WORKDIR\s+([^\n]+)", dockerfile.read_text(), re.M)
            if matches:
                workdir = matches[-1].strip()
                explicit_workdir = True
        image = environment.get("docker_image")
        if not image and not profile:
            raise HarnessError(
                f"{directory.name}: no prebuilt image or banana.toml native profile. "
                "Build-time Dockerfiles are not executed on the host."
            )
        prompt = (directory / "instruction.md").read_text()
        repository = profile.get("repo")
        if repository and not repository.startswith(("https://", "http://", "git@")):
            repository = str((directory / Path(repository).expanduser()).resolve())
        fingerprints = {}
        # Include held-out files in run identity without passing their content to the model.
        for path in sorted(directory.rglob("*")):
            if path.is_file() and ".git" not in path.parts and "solution" not in path.parts:
                fingerprints[str(path.relative_to(directory))] = digest(path.read_bytes().hex())
        yield Task(
            id=directory.name,
            benchmark=benchmark,
            prompt=prompt,
            kind="harbor",
            image=image,
            workdir=profile.get("workdir", workdir),
            source_dir=str(directory),
            repo=repository,
            base_commit=profile.get("base_commit"),
            setup=profile.get("setup", []),
            test_command=profile.get("test_command"),
            env={**environment.get("env", {}), **profile.get("env", {})},
            agent_timeout=data.get("agent", {}).get("timeout_sec", 3600),
            verifier_timeout=verifier.get("timeout_sec", 1800),
            metadata={
                **metadata,
                "task_file_hash": digest(fingerprints),
                "requested_resources": {
                    k: environment.get(k) for k in ("cpus", "memory_mb", "storage_mb", "gpus")
                },
                "network_mode": data.get("agent", {}).get(
                    "network_mode", environment.get("allow_internet")
                ),
                "native_profile": bool(profile),
                "explicit_workdir": explicit_workdir or "workdir" in profile,
                "source_provenance": provenance,
            },
            grading={"verifier": verifier, "native_profile": profile},
        )
