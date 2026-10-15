from __future__ import annotations

from typing import Any

from banana_harness.agent import ModelReply
from banana_harness.types import HarnessError


class TransformersModel:
    """Local causal chat-model inference; weights are loaded only on the first query.

    Reuse this instance across tasks to keep one copy of the weights in memory.
    No LiteLLM imports or API server are involved.
    """

    def __init__(
        self,
        name: str,
        *,
        device: str = "cpu",
        device_map: str | dict[str, Any] | None = None,
        dtype: str = "auto",
        revision: str | None = None,
        allow_downloads: bool = False,
        trust_remote_code: bool = False,
        max_new_tokens: int = 1024,
        generation_kwargs: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError("max_new_tokens must be a positive integer")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self.model_path = name
        self.name = f"transformers/{name}"
        self.device, self.device_map, self.dtype = device, device_map, dtype
        self.revision, self.allow_downloads = revision, allow_downloads
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs = dict(generation_kwargs or {})
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        # One continuation per query; never truncate away task instructions/history.
        reserved = {
            "max_new_tokens",
            "max_length",
            "num_return_sequences",
            "return_dict_in_generate",
            "input_ids",
            "inputs",
            "attention_mask",
            "generation_config",
        }
        if reserved.intersection(self.generation_kwargs):
            raise ValueError(
                "generation_kwargs cannot override input, length, or output structure; "
                "use the top-level max_new_tokens option for the output limit"
            )
        reserved_chat = {
            "conversation",
            "tokenize",
            "add_generation_prompt",
            "return_tensors",
            "return_dict",
            "truncation",
            "max_length",
            "padding",
            "continue_final_message",
        }
        if reserved_chat.intersection(self.chat_template_kwargs):
            raise ValueError("chat_template_kwargs cannot override input/output formatting")
        self._model = self._tokenizer = self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise HarnessError(
                "Install banana-harness[transformers] to use local Transformers models."
            ) from error
        options = {
            "local_files_only": not self.allow_downloads,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.revision is not None:
            options["revision"] = self.revision
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, **options)
            if not tokenizer.chat_template:
                raise HarnessError(
                    "Transformers runtime requires a tokenizer with a chat template."
                )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=self.dtype, device_map=self.device_map, **options
            )
            if self.device_map is None:
                model.to(self.device)
            model.eval()
        except OSError as error:
            hint = (
                " Check the model path, revision, and Hugging Face access permissions."
                if self.allow_downloads
                else " Supply a local/cached model or explicitly enable allow_downloads "
                "(--allow-downloads in the CLI)."
            )
            raise HarnessError(
                f"Cannot load Transformers model {self.model_path}.{hint}"
            ) from error
        self._torch, self._tokenizer, self._model = torch, tokenizer, model

    def query(self, messages: list[dict[str, str]]) -> ModelReply:
        self._load()
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **self.chat_template_kwargs,
        ).to(self._model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        context_limit = getattr(self._model.config, "max_position_embeddings", None)
        if isinstance(context_limit, int) and prompt_tokens + self.max_new_tokens > context_limit:
            raise HarnessError(
                f"Prompt ({prompt_tokens} tokens) plus max_new_tokens ({self.max_new_tokens}) "
                f"exceeds model context ({context_limit}); reduce the output limit or history."
            )
        options = {"do_sample": False, **self.generation_kwargs}
        if self._tokenizer.pad_token_id is None:
            eos = self._model.generation_config.eos_token_id
            if eos is not None:
                options.setdefault("pad_token_id", eos[0] if isinstance(eos, list) else eos)
        with self._torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                **options,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                return_dict_in_generate=False,
            )
        continuation = outputs[0, prompt_tokens:]
        completion_tokens = len(continuation)
        return ModelReply(
            self._tokenizer.decode(continuation, skip_special_tokens=True),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )
