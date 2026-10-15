import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from banana_harness import TransformersModel
from banana_harness.agent import FINISH, ModelReply
from banana_harness.cli import main
from banana_harness.types import HarnessError


@pytest.fixture
def runtime(monkeypatch):
    tokenizer = MagicMock(chat_template="template", pad_token_id=None)
    inputs = MagicMock()
    inputs.to.return_value = inputs
    inputs.__getitem__.return_value.shape = (1, 3)
    tokenizer.apply_chat_template.return_value = inputs
    tokenizer.decode.return_value = f"```bash\n{FINISH}\n```"
    model = MagicMock()
    model.device = "cpu"
    model.config.max_position_embeddings = 32
    model.generation_config.eos_token_id = [7, 8]
    model.generate.return_value.__getitem__.return_value = [4, 7]
    auto_tokenizer = MagicMock()
    auto_tokenizer.from_pretrained.return_value = tokenizer
    auto_model = MagicMock()
    auto_model.from_pretrained.return_value = model
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=auto_tokenizer,
            AutoModelForCausalLM=auto_model,
        ),
    )
    monkeypatch.setitem(sys.modules, "litellm", None)
    return SimpleNamespace(
        tokenizer=tokenizer,
        model=model,
        auto_tokenizer=auto_tokenizer,
        auto_model=auto_model,
    )


def test_local_generation_is_lazy_reused_and_excludes_prompt(runtime):
    adapter = TransformersModel("local/model", max_new_tokens=4)
    runtime.auto_model.from_pretrained.assert_not_called()
    messages = [{"role": "user", "content": "Fix it"}]
    reply = adapter.query(messages)
    adapter.query(messages)
    assert reply.content == f"```bash\n{FINISH}\n```"
    assert reply.usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert reply.cost == 0 and reply.cost_known
    runtime.auto_model.from_pretrained.assert_called_once_with(
        "local/model",
        dtype="auto",
        device_map=None,
        local_files_only=True,
        trust_remote_code=False,
    )
    runtime.auto_tokenizer.from_pretrained.assert_called_once_with(
        "local/model",
        local_files_only=True,
        trust_remote_code=False,
    )
    runtime.model.generate.return_value.__getitem__.assert_called_with((0, slice(3, None)))
    runtime.tokenizer.decode.assert_called_with([4, 7], skip_special_tokens=True)
    runtime.model.to.assert_called_once_with("cpu")
    assert runtime.model.generate.call_args.kwargs["pad_token_id"] == 7


def test_download_revision_remote_code_and_device_mapping(runtime):
    adapter = TransformersModel(
        "org/model",
        allow_downloads=True,
        trust_remote_code=True,
        revision="commit",
        device_map="auto",
        dtype="bfloat16",
        max_new_tokens=4,
        generation_kwargs={"do_sample": True, "temperature": 0.7},
        chat_template_kwargs={"enable_thinking": False},
    )
    adapter.query([{"role": "user", "content": "Task"}])
    for loader in (runtime.auto_model, runtime.auto_tokenizer):
        assert loader.from_pretrained.call_args.kwargs["trust_remote_code"] is True
        assert loader.from_pretrained.call_args.kwargs["local_files_only"] is False
        assert loader.from_pretrained.call_args.kwargs["revision"] == "commit"
    runtime.model.to.assert_not_called()
    assert runtime.model.generate.call_args.kwargs["temperature"] == 0.7
    assert runtime.tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"] is False


def test_context_overflow_fails_without_truncating_or_generating(runtime):
    with pytest.raises(HarnessError, match="exceeds model context"):
        TransformersModel("model", max_new_tokens=32).query([])
    runtime.model.generate.assert_not_called()


def test_missing_chat_template_is_actionable(runtime):
    runtime.tokenizer.chat_template = None
    with pytest.raises(HarnessError, match="chat template"):
        TransformersModel("model").query([])
    runtime.auto_model.from_pretrained.assert_not_called()


def test_missing_weights_explain_download_opt_in(runtime):
    runtime.auto_tokenizer.from_pretrained.side_effect = OSError("not cached")
    with pytest.raises(HarnessError, match="--allow-downloads"):
        TransformersModel("uncached").query([])


def test_missing_optional_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(HarnessError, match=r"banana-harness\[transformers\]"):
        TransformersModel("model").query([])


@pytest.mark.parametrize(
    "options",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"generation_kwargs": {"num_return_sequences": 2}},
        {"generation_kwargs": {"return_dict_in_generate": True}},
        {"chat_template_kwargs": {"truncation": True}},
    ],
)
def test_invalid_options_fail_before_loading(options):
    with pytest.raises(ValueError):
        TransformersModel("model", **options)


def test_cli_reuses_runtime_and_fingerprints_options(tmp_path, monkeypatch):
    created = []

    class LocalModel:
        name = "transformers/local"

        def __init__(self, name, **options):
            created.append((name, options))

        def query(self, messages):
            return ModelReply(f"```bash\n{FINISH}\n```")

    monkeypatch.setattr("banana_harness.cli.TransformersModel", LocalModel)
    monkeypatch.setitem(sys.modules, "litellm", None)
    source = tmp_path / "tasks.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"task-{i}",
                    "prompt": "Finish",
                    "test_command": "true",
                }
            )
            for i in range(2)
        )
    )
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"device": "cpu", "revision": "pinned"}))
    output = tmp_path / "job"
    argv = [
        "run",
        "custom",
        "--source",
        str(source),
        "--model",
        "local",
        "--model-runtime",
        "transformers",
        "--no-venv",
        "--output",
        str(output),
        "--model-options",
        str(options),
        "--trust-remote-code",
    ]
    assert main(argv) == 0
    assert created == [
        (
            "local",
            {
                "device": "cpu",
                "revision": "pinned",
                "trust_remote_code": True,
                "allow_downloads": False,
            },
        )
    ]
    first = json.loads((output / "summary.json").read_text())
    options.write_text(json.dumps({"device": "cpu", "revision": "different"}))
    assert main(argv) == 0
    second = json.loads((output / "summary.json").read_text())
    assert first["results"][0]["fingerprint"] != second["results"][0]["fingerprint"]


def test_real_transformers_checkpoint_offline(tmp_path, monkeypatch):
    """Runs when the optional extra is installed; uses original random tiny weights."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    monkeypatch.setitem(sys.modules, "litellm", None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    backend = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        chat_template=(
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        ),
    )
    tokenizer.save_pretrained(tmp_path)
    transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=3,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_positions=64,
            bos_token_id=1,
            eos_token_id=1,
        )
    ).save_pretrained(tmp_path)
    adapter = TransformersModel(str(tmp_path), max_new_tokens=4)
    reply = adapter.query([{"role": "user", "content": "hello"}])
    assert isinstance(reply.content, str)
    assert 1 <= reply.usage["completion_tokens"] <= 4
    assert reply.usage["total_tokens"] == (
        reply.usage["prompt_tokens"] + reply.usage["completion_tokens"]
    )
    assert reply.cost_known and reply.cost == 0
