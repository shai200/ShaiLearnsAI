"""Pluggable text client used by the labs that call a language model.

Selection order:

1. ``LLM_PROVIDER=openai|anthropic|huggingface|local`` forces a provider.
2. Otherwise an ``OPENAI_API_KEY`` selects OpenAI.
3. Otherwise an ``ANTHROPIC_API_KEY`` selects Anthropic.
4. Otherwise the local fallback runs, so the notebooks execute with no key
   and no network. ``LLM_PROVIDER=huggingface`` builds a tiny local GPT-2,
   or loads ``HF_MODEL`` when that variable names a public checkpoint.

The local fallback is not a language model. It is a deterministic stand-in
with the same ``complete(messages)`` shape, used so that prompting, structured
output, retrieval, and tool loops can be graded offline. Notebooks say this
explicitly wherever the fallback is on the path that produces an answer.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


Message = dict[str, str]


class LLMError(RuntimeError):
    pass


class LocalFallbackClient:
    """Deterministic offline client. See the module docstring."""

    name = "local-fallback"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        self.calls += 1
        text = "\n".join(m.get("content", "") for m in messages)
        mode = kwargs.get("mode")

        if mode == "json" or "RETURN_JSON" in text:
            return self._json(text)
        if mode == "tools" or "TOOL_LOOP" in text:
            return self._tools(text)
        if mode == "rag" or "RAG_ANSWER" in text:
            return self._rag(text)
        return self._prose(text)

    def _json(self, text: str) -> str:
        # The first structured-output call in the repair demo is intentionally
        # incomplete so the notebook's validation loop has something to fix.
        if "repair-demo" in text and self.calls == 1:
            return '{"city": "Paris"}'
        if "Paris" in text:
            return json.dumps({"city": "Paris", "country": "France"})
        if "Tokyo" in text:
            return json.dumps({"city": "Tokyo", "country": "Japan"})
        match = re.search(r"name=([A-Za-z]+).*?score=(\d+)", text)
        if match:
            return json.dumps({"name": match.group(1), "score": int(match.group(2))})
        return json.dumps({"status": "ok"})

    def _tools(self, text: str) -> str:
        """One tool decision from the transcript.

        The offline client looks for ``fee:``, ``calc:``, and ``convert:``
        markers, and it does not repeat a tool that already appears in the
        transcript. A hosted model receives the same transcript and is
        expected to return the same JSON shape.
        """

        used = re.findall(r'"action":\s*"(\w+)"', text)
        observations = re.findall(r"Observation \([^)]*\):\s*(.+)", text)

        fee = re.search(r"fee:\s*([a-z_]+)", text)
        if fee and "lookup_fee" not in used:
            return json.dumps({"action": "lookup_fee", "input": fee.group(1)})

        calc = re.search(r"calc:\s*([0-9+\-*/().{}a-z_ ]+)", text)
        if calc and "calculator" not in used:
            expr = calc.group(1).strip()
            if "{fee}" in expr:
                if not observations:
                    return json.dumps({"action": "final", "input": "missing fee observation"})
                expr = expr.replace("{fee}", observations[-1].strip().split()[0])
            return json.dumps({"action": "calculator", "input": expr})

        convert = re.search(r"convert:\s*([0-9.]+)\s*([A-Za-z]+)\s*to\s*([A-Za-z]+)", text)
        if convert and "convert" not in used:
            return json.dumps(
                {
                    "action": "convert",
                    "input": {
                        "value": float(convert.group(1)),
                        "src": convert.group(2).lower(),
                        "dst": convert.group(3).lower(),
                    },
                }
            )

        if observations:
            return json.dumps({"action": "final", "input": observations[-1].strip()})
        return json.dumps({"action": "final", "input": "I do not have a tool for that."})

    def _rag(self, text: str) -> str:
        # Extractive reader: quote the first context line and keep its citation.
        ctx = re.findall(r"\[(\d+)\]\s*(.+)", text)
        if not ctx:
            return "I do not know. [uncited]"
        cid, sentence = ctx[0]
        sentence = sentence.strip().split(". ")[0].strip(".")
        return f"{sentence}. [{cid}]"

    def _prose(self, text: str) -> str:
        if "capital of France" in text:
            if "step by step" in text.lower() or "chain of thought" in text.lower():
                return "France is a country in Europe. Its capital city is Paris. Answer: Paris"
            return "Paris"
        if "2 + 2" in text or "2+2" in text:
            return "4"
        return "Local fallback has no scripted answer for this prompt."


class HuggingFaceClient:
    """Local neural fallback with no API key.

    With ``HF_MODEL`` unset this builds a tiny GPT-2 from a config, so nothing
    is downloaded. That model is randomly initialized: it shows the provider
    plug, and it is not used to grade the labs. Set ``HF_MODEL`` to a public
    Hugging Face model id (for example an instruct model that fits the
    machine) to generate with a real checkpoint instead.
    """

    name = "huggingface"

    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
        except ImportError as exc:
            raise LLMError(
                "Install torch and transformers to use the Hugging Face provider."
            ) from exc
        self._torch = torch
        model_id = os.environ.get("HF_MODEL", "").strip()
        if model_id:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForCausalLM.from_pretrained(model_id)
            self.checkpoint = model_id
        else:
            self._tokenizer = None
            config = GPT2Config(
                vocab_size=128,
                n_positions=64,
                n_embd=32,
                n_layer=1,
                n_head=2,
                n_inner=64,
            )
            self._model = GPT2LMHeadModel(config)
            self.checkpoint = "untrained-tiny-gpt2"
        self._model.eval()
        if getattr(self._model.config, "pad_token_id", None) is None:
            self._model.config.pad_token_id = 0

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        prompt = "\n".join(m.get("content", "") for m in messages)[-200:]
        torch = self._torch
        if self._tokenizer is None:
            ids = [min(ord(ch), 127) for ch in prompt[-32:]] or [1]
            input_ids = torch.tensor([ids], dtype=torch.long)
        else:
            encoded = self._tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"][:, -32:]
        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self._model.config.pad_token_id,
            )
        if self._tokenizer is None:
            new_ids = output[0, input_ids.shape[1] :].tolist()
            text = "".join(chr(i) if 32 <= i < 127 else "?" for i in new_ids)
        else:
            text = self._tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
        return f"[{self.checkpoint}] {text}".strip()


class OpenAIClient:
    name = "openai"

    def __init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("Install the openai package to use the OpenAI provider.") from exc
        self._client = OpenAI()
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
        )
        return response.choices[0].message.content or ""


class AnthropicClient:
    name = "anthropic"

    def __init__(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("Install the anthropic package to use the Anthropic provider.") from exc
        self._client = anthropic.Anthropic()
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        user_messages = [m for m in messages if m.get("role") != "system"]
        kwargs_api: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(kwargs.get("max_tokens", 1024)),
            "temperature": kwargs.get("temperature", 0),
            "messages": user_messages or [{"role": "user", "content": ""}],
        }
        if system_parts:
            kwargs_api["system"] = "\n".join(system_parts)
        response = self._client.messages.create(**kwargs_api)
        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


def get_llm(provider: str | None = None):
    """Return a client. ``provider`` overrides the environment for one call."""

    choice = (provider or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if not choice:
        if os.environ.get("OPENAI_API_KEY"):
            choice = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        else:
            choice = "local"
    if choice in {"local", "fallback"}:
        return LocalFallbackClient()
    if choice == "openai":
        return OpenAIClient()
    if choice == "anthropic":
        return AnthropicClient()
    if choice in {"hf", "huggingface"}:
        return HuggingFaceClient()
    raise LLMError(
        f"Unknown LLM_PROVIDER {choice!r}. Use openai, anthropic, huggingface, or local."
    )
