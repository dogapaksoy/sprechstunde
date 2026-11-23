"""Chat model backends.

Every agent turn must be a JSON action. Small models are unreliable at that (in the
belegt evaluation SmolLM3 misspelled 17 % of keys), so the local backend uses
grammar-constrained decoding: llama.cpp compiles the JSON schema into a grammar and the
model cannot emit anything that does not match it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class ChatBackend(Protocol):
    name: str

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int = 400
    ) -> Completion: ...


MODELS = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf"),
    "qwen3-4b": ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
}


class LlamaCppBackend:
    """Local model via llama.cpp (CPU is fine for 1-4B models)."""

    def __init__(self, model: str = "qwen3-1.7b", n_ctx: int = 8192, threads: int | None = None):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        repo, filename = MODELS.get(model, (None, model))
        path = hf_hub_download(repo, filename) if repo else filename
        self.name = model
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=threads, verbose=False)

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int = 400
    ) -> Completion:
        t0 = time.perf_counter()
        out = self.llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": schema},
            temperature=0.0,
            max_tokens=max_tokens,
        )
        usage = out["usage"]
        return Completion(
            text=out["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            seconds=time.perf_counter() - t0,
        )


class ScriptedBackend:
    """Replays fixed responses; for tests and demos without a model."""

    def __init__(self, responses: list[dict[str, Any] | str], name: str = "scripted") -> None:
        self.responses = list(responses)
        self.name = name
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int = 400
    ) -> Completion:
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            raise RuntimeError("ScriptedBackend ran out of responses")
        r = self.responses.pop(0)
        text = r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
        return Completion(text, sum(len(m["content"]) for m in messages) // 4, len(text) // 4, 0.0)
