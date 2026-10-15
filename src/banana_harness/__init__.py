"""Host coding benchmark harness. Importing this package never downloads anything."""

from banana_harness.adapters import load_tasks, select_tasks
from banana_harness.agent import AgentConfig, BashAgent, LiteLLMModel, ScriptedModel
from banana_harness.environment import NativeEnvironment, RootfsEnvironment
from banana_harness.runner import RunConfig, Runner
from banana_harness.transformers_model import TransformersModel
from banana_harness.types import Task

__version__ = "0.1.0"
__all__ = [
    "AgentConfig",
    "BashAgent",
    "LiteLLMModel",
    "NativeEnvironment",
    "RootfsEnvironment",
    "RunConfig",
    "Runner",
    "ScriptedModel",
    "Task",
    "TransformersModel",
    "load_tasks",
    "select_tasks",
]
