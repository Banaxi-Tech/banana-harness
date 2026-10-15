from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


class HarnessError(RuntimeError):
    """An actionable configuration, provisioning, or execution error."""


@dataclass
class Task:
    id: str
    benchmark: str
    prompt: str
    kind: str = "repository"
    repo: str | None = None
    base_commit: str | None = None
    image: str | None = None
    workdir: str = "/app"
    source_dir: str | None = None
    setup: list[str] = field(default_factory=list)
    test_command: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    agent_timeout: float = 3600
    verifier_timeout: float = 1800
    metadata: dict[str, Any] = field(default_factory=dict)
    # Private inputs are kept in the driver process, never in the agent prompt/workspace.
    grading: dict[str, Any] = field(default_factory=dict, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            item.name: getattr(self, item.name) for item in fields(self) if item.name != "grading"
        }

    @property
    def path(self) -> Path | None:
        return Path(self.source_dir) if self.source_dir else None


@dataclass
class CommandResult:
    output: str
    returncode: int
    timed_out: bool = False
    duration: float = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Grade:
    status: str
    reward: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
