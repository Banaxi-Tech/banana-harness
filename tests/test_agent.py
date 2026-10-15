from banana_harness.agent import FINISH, AgentConfig, BashAgent, ModelReply, ScriptedModel
from banana_harness.environment import NativeEnvironment


def test_linear_bash_history_and_stateless_shell(tmp_path):
    replies = [
        "```bash\nexport TASK_VAR=42; printf changed > file\n```",
        '```bash\ntest -z "${TASK_VAR:-}" && cat file\n```',
        f"```bash\n{FINISH}\n```",
    ]
    agent = BashAgent(ScriptedModel(replies), NativeEnvironment(tmp_path))
    result = agent.run("Fix it")
    assert result["status"] == "submitted"
    assert result["steps"][1]["observation"]["output"] == "changed"
    assert [m["role"] for m in result["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_format_recovery_and_submission_not_spoofed_by_output(tmp_path):
    replies = [
        "No command",
        "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo work\n```",
        f"```mswea_bash_command\n{FINISH}\n```",
    ]
    result = BashAgent(ScriptedModel(replies), NativeEnvironment(tmp_path)).run("Task")
    assert result["status"] == "submitted"
    assert len(result["steps"]) == 3
    assert result["steps"][0]["format_error"]
    assert "observation" in result["steps"][1]


def test_extra_code_blocks_rejected(tmp_path):
    replies = ["```python\nx=1\n```\n```bash\ntouch bad\n```"]
    result = BashAgent(
        ScriptedModel(replies), NativeEnvironment(tmp_path), AgentConfig(max_steps=1)
    ).run("Task")
    assert result["status"] == "step_limit"
    assert not (tmp_path / "bad").exists()


def test_budget_stops_before_action(tmp_path):
    class ExpensiveModel:
        name = "expensive"

        def query(self, messages):
            return ModelReply("```bash\ntouch bad\n```", cost=2)

    result = BashAgent(ExpensiveModel(), NativeEnvironment(tmp_path), AgentConfig(max_cost=1)).run(
        "Task"
    )
    assert result["status"] == "cost_limit"
    assert not (tmp_path / "bad").exists()


def test_unknown_cost_cannot_silently_bypass_budget(tmp_path):
    class UnknownModel:
        name = "unknown"

        def query(self, messages):
            return ModelReply("```bash\ntouch bad\n```", cost_known=False)

    result = BashAgent(UnknownModel(), NativeEnvironment(tmp_path), AgentConfig(max_cost=1)).run(
        "Task"
    )
    assert result["status"] == "unknown_cost"
    assert not (tmp_path / "bad").exists()
