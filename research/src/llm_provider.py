"""A tiny pluggable LLM provider used by the research notebooks.

One interface, three backends::

    from llm_provider import get_provider
    llm = get_provider()            # picks a backend automatically
    print(llm.name)                 # e.g. "hf:HuggingFaceTB/SmolLM2-135M-Instruct"
    llm.complete("Say hi", system="You are terse.", max_tokens=32)

Backend selection
-----------------
1. ``LAB_LLM_PROVIDER=openai|anthropic|hf`` if set (an explicit choice).
2. otherwise OpenAI if ``OPENAI_API_KEY`` is set,
3. otherwise Anthropic if ``ANTHROPIC_API_KEY`` is set,
4. otherwise the **local Hugging Face** fallback (the default; free, offline
   after the first download, runs on CPU).

If a remote backend is requested but its key or SDK is missing -- or a remote
call fails at runtime -- the provider prints a warning and transparently falls
back to the local HF model. A notebook never fails just because no API key is
configured.

Model overrides
---------------
LAB_HF_MODEL          default "HuggingFaceTB/SmolLM2-135M-Instruct" (~540 MB fp32, Apache-2.0).
                      For an ultra-light smoke test use "sshleifer/tiny-gpt2" (outputs gibberish).
LAB_OPENAI_MODEL      default "gpt-4o-mini"
LAB_ANTHROPIC_MODEL   default "claude-opus-5-5"

The remote SDKs (``openai``, ``anthropic``) are optional dependencies and are
only imported when that backend is actually used.
"""

from __future__ import annotations

import math
import os
import warnings
from abc import ABC, abstractmethod
from typing import Sequence

DEFAULT_HF_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


class LLMProvider(ABC):
    """Minimal text-in / text-out interface shared by every backend."""

    name: str = "abstract"

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Return the model's reply to a single user ``prompt``."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.name}>"


# --------------------------------------------------------------------------- #
# Local Hugging Face backend (default)
# --------------------------------------------------------------------------- #
class HFProvider(LLMProvider):
    """Local causal LM via ``transformers``. Uses the chat template when the
    tokenizer has one, otherwise a plain ``System/User/Assistant`` transcript."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name or os.environ.get("LAB_HF_MODEL", DEFAULT_HF_MODEL)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.name = f"hf:{self.model_name}"

    # -- prompt formatting -------------------------------------------------- #
    def format_prompt(self, prompt: str, system: str | None = None) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        head = f"System: {system}\n" if system else ""
        return f"{head}User: {prompt}\nAssistant:"

    def complete(self, prompt, system=None, max_tokens=256, temperature=0.0) -> str:
        import torch

        text = self.format_prompt(prompt, system)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.15,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        new_tokens = out[0, enc["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def continuation_logprob(self, prompt: str, continuation: str, system: str | None = None) -> float:
        """Sum of log P(continuation tokens | formatted prompt). Used for
        multiple-choice AI feedback (as in the Constitutional AI paper)."""
        import torch

        context = self.format_prompt(prompt, system)
        ctx_ids = self.tokenizer(context, return_tensors="pt")["input_ids"]
        full_ids = self.tokenizer(context + continuation, return_tensors="pt")["input_ids"]
        full_ids = full_ids.to(self.device)
        n_ctx = ctx_ids.shape[1]
        with torch.no_grad():
            logits = self.model(full_ids).logits
        logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = full_ids[0, 1:]
        token_lp = logprobs.gather(-1, targets[:, None]).squeeze(-1)
        return float(token_lp[n_ctx - 1 :].sum())


# --------------------------------------------------------------------------- #
# Remote backends (optional)
# --------------------------------------------------------------------------- #
class OpenAIProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        from openai import OpenAI  # optional dependency

        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.model = model or os.environ.get("LAB_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self.name = f"openai:{self.model}"

    def complete(self, prompt, system=None, max_tokens=256, temperature=0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return (resp.choices[0].message.content or "").strip()


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        import anthropic  # optional dependency

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("LAB_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.name = f"anthropic:{self.model}"

    def complete(self, prompt, system=None, max_tokens=256, temperature=0.0) -> str:
        # Current Claude models think adaptively and do not accept sampling
        # parameters, so `temperature` is intentionally not forwarded. We ask for
        # low effort (these are short lab prompts) and leave headroom in
        # max_tokens so thinking never starves the visible answer.
        kwargs = dict(
            model=self.model,
            max_tokens=max(int(max_tokens), 1024),
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
        )
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            return "[refused by the model's safety classifier]"
        return "".join(b.text for b in resp.content if b.type == "text").strip()


# --------------------------------------------------------------------------- #
# Fallback wrapper + factory
# --------------------------------------------------------------------------- #
class FallbackProvider(LLMProvider):
    """Wraps a remote provider; on any runtime error, warns once and uses HF."""

    def __init__(self, primary: LLMProvider):
        self.primary = primary
        self._fallback: HFProvider | None = None
        self.name = primary.name

    @property
    def fallback(self) -> HFProvider:
        if self._fallback is None:
            self._fallback = HFProvider()
        return self._fallback

    def complete(self, prompt, system=None, max_tokens=256, temperature=0.0) -> str:
        try:
            return self.primary.complete(prompt, system, max_tokens, temperature)
        except Exception as exc:  # network, auth, quota ...
            warnings.warn(f"{self.primary.name} failed ({exc!r}); falling back to local HF model.")
            self.name = self.fallback.name
            return self.fallback.complete(prompt, system, max_tokens, temperature)


_CACHE: dict[str, LLMProvider] = {}


def _try_remote(kind: str) -> LLMProvider | None:
    key_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[kind]
    if not os.environ.get(key_var):
        warnings.warn(f"LAB_LLM_PROVIDER={kind} but {key_var} is not set; using local HF model.")
        return None
    try:
        cls = OpenAIProvider if kind == "openai" else AnthropicProvider
        return FallbackProvider(cls())
    except ImportError:
        warnings.warn(f"The `{kind}` Python package is not installed (pip install {kind}); using local HF model.")
        return None


def get_provider(name: str | None = None) -> LLMProvider:
    """Return a (cached) provider. ``name`` overrides ``LAB_LLM_PROVIDER``."""
    choice = (name or os.environ.get("LAB_LLM_PROVIDER", "")).strip().lower()
    if not choice:
        if os.environ.get("OPENAI_API_KEY"):
            choice = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        else:
            choice = "hf"
    if choice not in {"openai", "anthropic", "hf"}:
        warnings.warn(f"Unknown LAB_LLM_PROVIDER={choice!r}; using local HF model.")
        choice = "hf"

    if choice in _CACHE:
        return _CACHE[choice]
    provider = _try_remote(choice) if choice != "hf" else None
    if provider is None:
        choice = "hf"
        provider = _CACHE.get("hf") or HFProvider()
    _CACHE[choice] = provider
    return provider


def complete(prompt: str, system: str | None = None, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """Module-level convenience: ``complete("...")`` with the default provider."""
    return get_provider().complete(prompt, system=system, max_tokens=max_tokens, temperature=temperature)


def choose(
    provider: LLMProvider,
    question: str,
    options: Sequence[str],
    system: str | None = None,
) -> tuple[int, list[float]]:
    """Ask the model a multiple-choice question; return (best_index, probs).

    * Local HF models: score each option label "(A)", "(B)", ... by its
      log-probability after the prompt and normalise -- the same trick the
      Constitutional AI paper uses to turn a feedback model into soft labels.
    * Remote models: ask for a single letter and parse it (probabilities are
      then one-hot).
    """
    letters = [chr(ord("A") + i) for i in range(len(options))]
    body = question.strip() + "\n\n" + "\n".join(f"({l}) {o}" for l, o in zip(letters, options))
    body += "\n\nAnswer with the letter of the best option."
    base = provider.fallback if isinstance(provider, FallbackProvider) and provider._fallback else provider
    if isinstance(base, HFProvider):
        scores = [base.continuation_logprob(body, f"The answer is ({l})", system) for l in letters]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps)
        probs = [e / z for e in exps]
        return int(max(range(len(probs)), key=probs.__getitem__)), probs
    reply = provider.complete(body, system=system, max_tokens=8).upper()
    idx = next((i for i, l in enumerate(letters) if f"({l})" in reply or reply.strip().startswith(l)), 0)
    return idx, [1.0 if i == idx else 0.0 for i in range(len(options))]


if __name__ == "__main__":
    llm = get_provider()
    print(llm.name)
    print(llm.complete("In one sentence, what is a language model?", max_tokens=48))
