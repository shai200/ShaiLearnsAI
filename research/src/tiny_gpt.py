"""A compact nanoGPT-style decoder-only transformer + training helpers.

Adapted from Andrej Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT,
model.py / train.py; MIT License, Copyright (c) 2022 Andrej Karpathy -- the
full notice is reproduced in research/THIRD_PARTY_NOTICES.md), condensed for
teaching and for CPU-sized experiments. Notebook 02 walks through this
architecture step by step; notebook 03 imports it to run a scaling sweep.

Differences from nanoGPT worth knowing about: evaluation at chosen steps
(``eval_at``), an optional constant-LR schedule, AdamW beta2 = 0.95 by default
(nanoGPT's shakespeare_char config uses 0.99), and ``num_params()`` excludes
both embedding tables by default.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64     # context length
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)  # q, k, v in one matmul
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.n_head, self.n_embd, self.dropout = cfg.n_head, cfg.n_embd, cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying (Press & Wolf, 2017)
        self.apply(self._init_weights)
        # GPT-2 style: scale residual projections by 1/sqrt(2 * n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count. Kaplan et al. (2020) scale with *non-embedding* N,
        so by default we subtract the token + position embeddings (the token
        embedding is tied to lm_head, so it is only counted once)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel() + self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx

    def configure_optimizer(self, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95)):
        """AdamW with decay on 2D weights only (matmuls, embeddings), as in nanoGPT."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class CharTokenizer:
    """Character-level tokenizer: the simplest possible vocabulary."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def train_val_split(data: torch.Tensor, frac: float = 0.9):
    n = int(frac * len(data))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu", generator=None):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def cosine_lr(step: int, max_steps: int, lr: float, warmup: int, min_lr_frac: float = 0.1) -> float:
    """Linear warmup then cosine decay to ``min_lr_frac * lr`` (nanoGPT schedule)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict, block_size: int, batch_size: int, eval_iters: int, device="cpu", seed=0):
    """Average loss over ``eval_iters`` fixed random batches per split."""
    model.eval()
    out = {}
    for name, data in splits.items():
        g = torch.Generator().manual_seed(seed)
        losses = [model(*get_batch(data, block_size, batch_size, device, g))[1].item() for _ in range(eval_iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def train(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    max_steps: int = 300,
    batch_size: int = 32,
    lr: float = 3e-3,
    warmup: int = 20,
    eval_every: int = 50,
    eval_iters: int = 10,
    device: str = "cpu",
    seed: int = 1337,
    eval_at: set[int] | None = None,
    schedule: str = "cosine",
    verbose: bool = True,
):
    """Minimal nanoGPT training loop. Returns a history dict.

    ``eval_at`` (optional) forces evaluation at specific steps -- handy for
    reading off loss-vs-tokens curves in the scaling-law notebook.
    ``schedule``: "cosine" (warmup + cosine decay, nanoGPT default) or
    "constant" (warmup then flat LR, so intermediate checkpoints are comparable).
    """
    block_size = model.cfg.block_size
    opt = model.configure_optimizer(lr)
    g = torch.Generator().manual_seed(seed)
    hist = {"step": [], "tokens": [], "train_loss": [], "val_loss": [], "batch_loss": []}
    splits = {"train": train_data, "val": val_data}
    t0 = time.time()
    model.train()
    for step in range(max_steps + 1):
        if step % eval_every == 0 or step == max_steps or (eval_at and step in eval_at):
            losses = estimate_loss(model, splits, block_size, batch_size, eval_iters, device)
            hist["step"].append(step)
            hist["tokens"].append(step * batch_size * block_size)
            hist["train_loss"].append(losses["train"])
            hist["val_loss"].append(losses["val"])
            if verbose:
                print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} | {time.time() - t0:5.1f}s")
        if step == max_steps:
            break
        for group in opt.param_groups:
            group["lr"] = cosine_lr(step, max_steps, lr, warmup) if schedule == "cosine" else lr * min(1.0, (step + 1) / warmup)
        x, y = get_batch(train_data, block_size, batch_size, device, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist["batch_loss"].append(loss.item())
    hist["seconds"] = time.time() - t0
    return hist
