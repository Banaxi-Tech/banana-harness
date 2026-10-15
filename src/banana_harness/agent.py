from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from banana_harness.environment import Environment
from banana_harness.types import HarnessError

FINISH = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SYSTEM_PROMPT = """You are a coding agent with one tool: bash.
Reply with exactly one fenced bash code block containing your next command.
You may briefly explain the action before the code block. Each command runs in a new shell:
directory changes and exports do not persist between actions; filesystem changes do.
Inspect the workspace, implement the requested changes, and test them.
When finished, submit with exactly this command in a bash block:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
Do not combine the submission command with another command.
"""


@dataclass
class ModelReply:
    content: str
    cost: float = 0.0
    usage: dict[str, Any] | None = None
    cost_known: bool = True


class Model(Protocol):
    name: str

    def query(self, messages: list[dict[str, str]]) -> ModelReply: ...


class ScriptedModel:
    """Offline replay model for smoke tests and deterministic command trajectories."""

    name = "scripted"

    def __init__(self, replies: Iterable[str]):
        self.replies = iter(replies)

    def query(self, messages: list[dict[str, str]]) -> ModelReply:
        try:
            return ModelReply(next(self.replies))
        except StopIteration as error:
            raise HarnessError("Scripted model ran out of responses before submission.") from error


class LiteLLMModel:
    def __init__(self, name: str, **kwargs: Any):
        try:
            import litellm
        except ImportError as error:
            raise HarnessError("Install banana-harness[models] to use API models.") from error
        self.client = litellm
        self.name = name
        self.kwargs = kwargs
        self.kwargs.setdefault("timeout", 120)
        self.kwargs.setdefault("num_retries", 2)

    def query(self, messages: list[dict[str, str]]) -> ModelReply:
        response = self.client.completion(model=self.name, messages=messages, **self.kwargs)
        cost_known = True
        try:
            cost = self.client.completion_cost(completion_response=response)
            if cost is None:
                cost_known = False
                cost = 0.0
        except Exception:
            cost_known = False
            cost = 0.0
        usage = response.usage.model_dump() if response.usage else {}
        return ModelReply(response.choices[0].message.content or "", cost, usage, cost_known)


@dataclass
class AgentConfig:
    max_steps: int = 100
    command_timeout: float = 120
    max_cost: float | None = None
    observation_limit: int = 10_000
    system_prompt: str = SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.command_timeout <= 0 or self.observation_limit < 100:
            raise ValueError("Positive step/timeout limits and observation_limit >= 100 required.")
        if self.max_cost is not None and self.max_cost <= 0:
            raise ValueError("max_cost must be positive or None")


class BashAgent:
    def __init__(self, model: Model, environment: Environment, config: AgentConfig | None = None):
        self.model, self.environment = model, environment
        self.config = config or AgentConfig()
        self.messages: list[dict[str, str]] = []
        self.steps: list[dict[str, Any]] = []
        self.cost = 0.0
        self.cost_known = True

    def snapshot(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "model": self.model.name,
            "config": asdict(self.config),
            "cost": self.cost,
            "cost_known": self.cost_known,
            "messages": self.messages,
            "steps": self.steps,
        }

    def run(
        self,
        prompt: str,
        *,
        timeout: float = 3600,
        checkpoint: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        self.messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": f"Workspace: {self.environment.workdir}\n\n{prompt}"},
        ]
        status = "step_limit"
        for _ in range(self.config.max_steps):
            if time.monotonic() - start >= timeout:
                status = "time_limit"
                break
            if self.config.max_cost is not None and self.cost >= self.config.max_cost:
                status = "cost_limit"
                break
            reply = self.model.query(self.messages)
            self.messages.append({"role": "assistant", "content": reply.content})
            self.cost += reply.cost
            self.cost_known &= reply.cost_known
            step: dict[str, Any] = {
                "response": reply.content,
                "usage": reply.usage,
                "cost": reply.cost,
                "cost_known": reply.cost_known,
            }
            self.steps.append(step)
            blocks = re.findall(r"```([^\n`]*)\n(.*?)```", reply.content, re.S)
            allowed = {"bash", "sh", "mswea_bash_command"}
            if self.config.max_cost is not None and not self.cost_known:
                status = "unknown_cost"
            elif self.config.max_cost is not None and self.cost >= self.config.max_cost:
                status = "cost_limit"
            elif time.monotonic() - start >= timeout:
                status = "time_limit"
            elif len(blocks) != 1 or blocks[0][0].strip() not in allowed:
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Format error: provide exactly one fenced bash code block.",
                    }
                )
                step["format_error"] = True
            else:
                command = blocks[0][1].strip()
                step["command"] = command
                if command == FINISH:
                    status = "submitted"
                else:
                    remaining = timeout - (time.monotonic() - start)
                    result = self.environment.execute(
                        command,
                        min(self.config.command_timeout, max(0.001, remaining)),
                    )
                    step["observation"] = result.as_dict()
                    output = result.output
                    limit = self.config.observation_limit
                    if len(output) > limit:
                        output = (
                            output[: limit // 2]
                            + "\n... output truncated ...\n"
                            + output[-limit // 2 :]
                        )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f"<returncode>{result.returncode}</returncode>\n"
                            f"<timed_out>{result.timed_out}</timed_out>\n"
                            f"<output>\n{output}\n</output>",
                        }
                    )
            if checkpoint:
                checkpoint(self.snapshot(status if status != "step_limit" else "running"))
            if status != "step_limit":
                break
        return self.snapshot(status)
