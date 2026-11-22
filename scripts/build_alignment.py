"""LoRA, DPO, interpretability, RAG labs, and the prompting notebook."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nbutil import BOOT, colab, make_notebook  # noqa: E402


def check(cells):
    for kind, source in cells:
        if kind != "code":
            continue
        cleaned = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
        )
        ast.parse(cleaned)


def emit(path, cells, meta):
    check(cells)
    make_notebook(path, cells, meta)


def lora():
    cells = [
        ("md", """
# LoRA and PEFT

**Track:** Fine-tuning and alignment. **Level:** Advanced. **CPU:** yes.

## Goals

- Freeze a linear layer and train a low-rank update `x @ A @ B`.
- Check that merging the update into the base weight reproduces the forward pass.
- Attach Hugging Face PEFT to a randomly initialized GPT-2 that is not downloaded.

## Sources and license

The update is the factorization in Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021). The implementation is original. The second half calls Hugging Face PEFT (Apache-2.0) on a `GPT2LMHeadModel` built from a config, so the lab does not download a checkpoint. PEFT's own source is not copied.

## Concepts

Full fine-tuning updates every weight. LoRA leaves `W` frozen and learns `ΔW = (α / r) B A`, with `A` shaped `(in, r)` and `B` shaped `(r, out)` in this notebook's layout. `r` is the rank. If `r` is much smaller than `min(in, out)`, the number of trained numbers drops by about that factor. Merging adds `ΔW` into `W` so inference does not pay for the extra matmul. PEFT does the same attachment to the modules you name. Here that module is GPT-2's `c_attn`.
"""),
        ("code", colab(["torch", "transformers", "peft"])),
        ("code", BOOT + """
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from labs._shared.seed import seed_everything

seed_everything(0)
torch.manual_seed(0)
"""),
        ("md", """
## Guided exercise: LoRA from scratch

The base weight is frozen at zero, so the low-rank factors have to fit the target by themselves. The target map has rank at most 4, and `r` is 4, so a fit is possible. After training, `A.grad` exists and `base.weight.requires_grad` is false.
"""),
        ("code", r'''
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=8.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad_(False)
        self.base.bias.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(in_features, r) * 0.02)
        self.B = nn.Parameter(torch.zeros(r, out_features))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scale

    def merged_weight(self):
        # nn.Linear stores W as (out, in). x @ A @ B adds scale * (B.T @ A.T).
        return self.base.weight.detach() + self.scale * (self.B.detach().T @ self.A.detach().T)


torch.manual_seed(0)
x = torch.randn(256, 32)
# The target map has rank 4, which matches r below and is out of reach for r=1.
y = x @ (torch.randn(32, 4) @ torch.randn(4, 32))
layer = LoRALinear(32, 32, r=4, alpha=8.0)
with torch.no_grad():
    layer.base.weight.zero_()
    layer.base.bias.zero_()
opt = torch.optim.Adam([layer.A, layer.B], lr=0.08)
start = float(F.mse_loss(layer(x), y).detach())
for _ in range(120):
    loss = F.mse_loss(layer(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
print("mse", round(start, 3), "->", round(float(loss.detach()), 3))
assert float(loss.detach()) < start * 0.2
assert layer.base.weight.requires_grad is False
assert layer.A.grad is not None

merged = nn.Linear(32, 32)
with torch.no_grad():
    merged.weight.copy_(layer.merged_weight())
    merged.bias.copy_(layer.base.bias)
probe = torch.randn(8, 32)
assert torch.allclose(merged(probe), layer(probe), atol=1e-5)
trainable = layer.A.numel() + layer.B.numel()
full = layer.base.weight.numel()
print("lora params", trainable, "full params", full)
assert trainable < full
'''),
        ("md", """
## Guided exercise: PEFT on a tiny GPT-2

The model is constructed from `GPT2Config`. Nothing is downloaded. `target_modules=["c_attn"]` is the packed query/key/value projection. Trainable parameters must be a minority of the total. One optimizer step has to produce a finite loss. `bos_token_id` is set inside the vocab so the config warning stays quiet.
"""),
        ("code", r'''
config = GPT2Config(
    vocab_size=64,
    n_positions=32,
    n_embd=32,
    n_layer=2,
    n_head=2,
    n_inner=64,
    bos_token_id=0,
    eos_token_id=0,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
)
base = GPT2LMHeadModel(config)
peft_model = get_peft_model(
    base,
    LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
    ),
)
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in peft_model.parameters())
print(f"trainable {trainable} / {total}")
assert 0 < trainable < total * 0.5
ids = torch.randint(0, 64, (8, 16))
opt = torch.optim.AdamW((p for p in peft_model.parameters() if p.requires_grad), lr=1e-3)
peft_model.train()
out = peft_model(input_ids=ids, labels=ids)
out.loss.backward()
opt.step()
print("one-step loss", round(float(out.loss.detach()), 3))
assert torch.isfinite(out.loss).item()
print("LAB OK")
'''),
        ("md", """
## Your turn

Print the trainable parameter count of the from-scratch layer at `r=1` and at `r=8`. The count is `r * (in + out)` plus the frozen base, which does not change.

## Solution
"""),
        ("code", r'''
def trainable_count(rank: int) -> int:
    module = LoRALinear(16, 4, r=rank)
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

low, high = trainable_count(1), trainable_count(8)
print("r=1", low, "r=8", high)
assert high == 8 * low
'''),
        ("md", """
## Beyond the course

Fit the same target with `r=1` and with `r=4` for the same number of steps, from the same kind of initialization. `r=1` cannot represent a rank-4 map. Its final loss should be worse. This is the rank ablation on a problem where the answer is known.
"""),
        ("code", r'''
def fit(rank: int) -> float:
    torch.manual_seed(0)
    module = LoRALinear(32, 32, r=rank, alpha=float(rank))
    with torch.no_grad():
        module.base.weight.zero_()
        module.base.bias.zero_()
    optimizer = torch.optim.Adam([module.A, module.B], lr=0.08)
    last = None
    for _ in range(120):
        last = F.mse_loss(module(x), y)
        optimizer.zero_grad(set_to_none=True)
        last.backward()
        optimizer.step()
    return float(last.detach())

loss_r1, loss_r4 = fit(1), fit(4)
print("r=1", round(loss_r1, 3), "r=4", round(loss_r4, 3))
assert loss_r4 < loss_r1
'''),
        ("md", """
## Research questions

1. Merging is exact for this linear layer. What becomes inexact if the forward pass applied dropout to the LoRA branch during training?
2. PEFT trained `c_attn` only. Which GPT-2 tensors are still frozen, and which of them would you unfreeze for a new output vocabulary?
3. The rank ablation used a target of rank 4. How would you estimate the rank you need when the target is a language-model loss and you cannot see `W`?

## Further reading

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021).
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized Language Models* (2023).
- Hugging Face PEFT documentation (Apache-2.0). https://github.com/huggingface/peft
"""),
    ]
    emit(
        "labs/finetuning/01_lora/lora_peft.ipynb",
        cells,
        {"track": "finetuning", "lab": "lora", "needs_gpu": False, "needs_api_key": False},
    )


def dpo():
    cells = [
        ("md", """
# Reward models and DPO

**Track:** Fine-tuning and alignment. **Level:** Advanced. **CPU:** yes.

## Goals

- Fit a reward model with a Bradley-Terry loss on synthetic preferences.
- Implement the DPO loss and check that the preferred response gains probability.
- Compare a low `beta` with a high `beta` by the KL from the reference policy.

## Sources and license

Original notebook. Bradley-Terry preference modeling is the reward-model loss in Christiano et al., *Deep Reinforcement Learning from Human Preferences* (2017) and in Ouyang et al. (InstructGPT, 2022). The direct preference loss is Rafailov et al., *Direct Preference Optimization* (2023):

`-log sigmoid( beta * (log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l)) )`.

No human preference dataset is included. The pairs are drawn from a known linear reward so the accuracy has a right answer.

## Concepts

A reward model scores a response. Bradley-Terry says the preferred response should outscore the rejected one, with probability `sigmoid(r_w - r_l)`. DPO writes the same objective on the policy itself, using the closed form that relates the optimal policy to a reference policy and a reward. `beta` is how hard the policy is pulled toward the reference. At the optimum, a large `beta` stays closer to the reference. A small `beta` is willing to put nearly all mass on the preferred action. The KL measured at the end of training is the check.
"""),
        ("code", colab(["torch", "numpy"])),
        ("code", BOOT + """
import torch
import torch.nn as nn
import torch.nn.functional as F

from labs._shared.seed import seed_everything

seed_everything(0)
torch.manual_seed(0)
"""),
        ("md", """
## Guided exercise: Bradley-Terry reward model

Two actions. The preferred action is the one with the higher linear score `x · w_a`. The reward model is a linear map from `x` to two scores. Accuracy is how often the higher score matches the preference. The features are random and the rule is linear, so this should not be a close call.
"""),
        ("code", r'''
N, D = 900, 6
g = torch.Generator().manual_seed(0)
X = torch.randn(N, D, generator=g)
W = torch.randn(2, D, generator=g)
true_score = X @ W.T
prefer_second = true_score[:, 1] > true_score[:, 0]

reward = nn.Linear(D, 2)
opt = torch.optim.Adam(reward.parameters(), lr=0.08)
for _ in range(80):
    scores = reward(X)
    chosen = torch.where(prefer_second, scores[:, 1], scores[:, 0])
    rejected = torch.where(prefer_second, scores[:, 0], scores[:, 1])
    loss = -F.logsigmoid(chosen - rejected).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
with torch.no_grad():
    accuracy = float((reward(X).argmax(dim=-1) == prefer_second.long()).float().mean())
print("reward loss", round(float(loss.detach()), 3), "accuracy", round(accuracy, 3))
assert accuracy > 0.85
'''),
        ("md", """
## Guided exercise: DPO

The policy is a linear map to two logits. The reference is a frozen copy of the initialization. The loss is the expression above. The margin is the mean difference between the log-probability of the preferred action and the rejected action. It should rise.
"""),
        ("code", r'''
def dpo_loss(policy, reference, features, prefer_second, beta):
    logits = policy(features)
    with torch.no_grad():
        ref_logits = reference(features)
    logp = F.log_softmax(logits, dim=-1)
    logref = F.log_softmax(ref_logits, dim=-1)
    index_w = prefer_second.long()
    index_l = 1 - index_w
    pi_w = logp.gather(1, index_w[:, None]).squeeze(1)
    pi_l = logp.gather(1, index_l[:, None]).squeeze(1)
    ref_w = logref.gather(1, index_w[:, None]).squeeze(1)
    ref_l = logref.gather(1, index_l[:, None]).squeeze(1)
    return -F.logsigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l))).mean()

def margin(policy, features, prefer_second):
    logp = F.log_softmax(policy(features), dim=-1)
    index_w = prefer_second.long()
    index_l = 1 - index_w
    return float((logp.gather(1, index_w[:, None]) - logp.gather(1, index_l[:, None])).mean())

def kl_to_ref(policy, reference, features):
    logp = F.log_softmax(policy(features), dim=-1)
    logref = F.log_softmax(reference(features), dim=-1)
    probs = logp.exp()
    return float((probs * (logp - logref)).sum(dim=-1).mean())

def train_dpo(beta, steps=120, lr=0.08):
    torch.manual_seed(2)
    policy = nn.Linear(D, 2)
    reference = nn.Linear(D, 2)
    reference.load_state_dict(policy.state_dict())
    for p in reference.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    before = margin(policy, X, prefer_second)
    for _ in range(steps):
        loss = dpo_loss(policy, reference, X, prefer_second, beta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    after = margin(policy, X, prefer_second)
    return before, after, kl_to_ref(policy, reference, X), float(loss.detach())

before, after, kl_low, _loss = train_dpo(0.2)
print("margin", round(before, 3), "->", round(after, 3), "KL at beta 0.2", round(kl_low, 3))
assert after > before + 0.2
'''),
        ("md", """
## Your turn

Write down the loss at `beta = 1` when the policy and the reference assign the same probability to both answers. `logsigmoid(0)` is `-log(0.5)`. The loss should be that number, up to floating point.

## Solution
"""),
        ("code", r'''
torch.manual_seed(0)
policy = nn.Linear(D, 2)
reference = nn.Linear(D, 2)
reference.load_state_dict({k: v.detach().clone() for k, v in policy.state_dict().items()})
for p in reference.parameters():
    p.requires_grad_(False)
value = float(dpo_loss(policy, reference, X[:64], prefer_second[:64], beta=1.0))
print("loss at the reference", round(value, 4), "target", round(float(-torch.log(torch.tensor(0.5))), 4))
assert abs(value - float(-torch.log(torch.tensor(0.5)))) < 1e-4
'''),
        ("md", """
## Beyond the course

Train again at `beta = 5`. The preferred-answer margin should still increase, and the KL to the reference should be smaller than it was at `beta = 0.2`. That is the knob the DPO paper uses to stay near the reference policy. It is measured here, not assumed.
"""),
        ("code", r'''
before_hi, after_hi, kl_high, _ = train_dpo(5.0)
print("beta 5 margin", round(before_hi, 3), "->", round(after_hi, 3), "KL", round(kl_high, 3))
print("beta 0.2 KL", round(kl_low, 3))
assert after_hi > before_hi
assert kl_high < kl_low
'''),
        ("md", """
## Research questions

1. The preferences were noise-free. Where does Bradley-Terry accuracy stop being a measure of the reward model and start being a measure of label noise?
2. DPO never fits an explicit reward head. What question about the reward can you no longer ask after you throw the head away?
3. `beta` was constant. Gao et al. show reward over-optimization when the policy chases a learned reward too hard. Which quantity in this notebook would move first if the preferences were themselves the output of a fitted model?

## Further reading

- Christiano et al., *Deep Reinforcement Learning from Human Preferences* (2017).
- Ouyang et al., InstructGPT (OpenAI, 2022).
- Rafailov et al., *Direct Preference Optimization* (2023).
- Bai et al., *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback* (Anthropic, 2022).
- Gao, Schulman, and Hilton, *Scaling Laws for Reward Model Overoptimization* (OpenAI, 2022).
"""),
    ]
    emit(
        "labs/finetuning/02_dpo_reward_model/dpo_reward_model.ipynb",
        cells,
        {"track": "finetuning", "lab": "dpo", "needs_gpu": False, "needs_api_key": False},
    )


def interpret():
    cells = [
        ("md", """
# Probing and attention

**Track:** Frontier research. **Level:** Advanced. **CPU:** yes.

## Goals

- Train a one-layer attention model whose target at every position is the token in position 0.
- Measure how much attention mass lands on position 0.
- Compare that model with a probe that sees only the current token.
- Run the same tokens through TransformerLens `run_with_cache` when the package is installed, and name the cache tensors that correspond to the measurement.

## Sources and license

Original notebook. Linear probes are Alain and Bengio, *Understanding Intermediate Layers Using Linear Classifier Probes* (2016). The attention measurement is the kind of circuit check in Elhage et al., *A Mathematical Framework for Transformer Circuits* (Anthropic, 2021). The cache API is TransformerLens (Neel Nanda, MIT). This lab does not copy ARENA or TransformerLens tutorials. If `transformer_lens` is not installed the cache cell prints SKIP and the rest of the lab still runs. CPU CI does not require the package, because the measurement above is the experiment. The install line is in the further-reading cell as a comment, not as a hidden download.

## Concepts

A probe is a linear classifier on a frozen activation. It answers "is this fact linearly readable here?" It does not answer "does the model use it?". Attention mass answers a more specific question: of the residual update at this layer, which source positions is the head averaging? On this task the label is the token at position 0, and a causal head is allowed to look at position 0 from every later position. A model that only looks at the current token cannot do better than chance, because the current token is drawn independently. That control is what makes the attention number meaningful.
"""),
        ("code", colab(["torch", "numpy"])),
        ("code", BOOT + """
import torch
import torch.nn as nn
import torch.nn.functional as F

from labs._shared.seed import seed_everything

seed_everything(0)
torch.manual_seed(0)
V, T = 10, 6
"""),
        ("md", """
## Guided exercise: a copy-the-first-token attention model

The boolean attention mask uses `True` for positions that must not be attended, which is the PyTorch convention for `nn.MultiheadAttention`. The loss is next-token-free: every position is trained to predict token 0. Position 0 predicting itself is trivial. The accuracy that matters is on the later positions.
"""),
        ("code", r'''
class CopyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 32)
        self.pos = nn.Embedding(T, 32)
        self.attn = nn.MultiheadAttention(32, num_heads=1, batch_first=True)
        self.head = nn.Linear(32, V)

    def forward(self, idx):
        b, t = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(t, device=idx.device))
        banned = torch.triu(torch.ones(t, t, dtype=torch.bool, device=idx.device), diagonal=1)
        hidden, weights = self.attn(x, x, x, attn_mask=banned, need_weights=True)
        return self.head(hidden), weights


class CurrentTokenProbe(nn.Module):
    """Control: the label is not a function of the current token."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 32)
        self.head = nn.Linear(32, V)

    def forward(self, idx):
        return self.head(self.emb(idx))


def batch(n, generator):
    return torch.randint(0, V, (n, T), generator=generator)


def train(model, steps, current_only=False):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    last = None
    for _ in range(steps):
        idx = batch(128, g)
        target = idx[:, :1].expand(idx.size(0), T)
        if current_only:
            logits = model(idx)
        else:
            logits, _ = model(idx)
        loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    return last


copier = CopyAttention()
control = CurrentTokenProbe()
print("copier loss", round(train(copier, 80), 3), "control loss", round(train(control, 80, current_only=True), 3))

@torch.no_grad()
def evaluate(model, current_only=False):
    idx = batch(400, torch.Generator().manual_seed(1))
    target = idx[:, :1].expand_as(idx)
    if current_only:
        logits = model(idx)
        weights = None
    else:
        logits, weights = model(idx)
    pred = logits.argmax(dim=-1)
    acc_later = float((pred[:, 1:] == target[:, 1:]).float().mean())
    return acc_later, weights

acc_copy, weights = evaluate(copier)
acc_control, _ = evaluate(control, current_only=True)
# weights: (batch, query, key), averaged over heads. Mass on key 0 from queries 1:.
mass = float(weights[:, 1:, 0].mean())
print(f"attention mass on position 0 {mass:.3f}")
print(f"accuracy later positions, copier {acc_copy:.3f}, current-token probe {acc_control:.3f}")
assert mass > 0.45
assert acc_copy > acc_control + 0.25
print("LAB OK")
'''),
        ("md", """
## Your turn

Print the attention row at query position 5, averaged over the evaluation batch. Position 0 should be the largest entry.

## Solution
"""),
        ("code", r'''
row = weights[:, 5, :].mean(dim=0)
print("mean attention from position 5", row.round(decimals=3).tolist())
assert int(row.argmax()) == 0
'''),
        ("md", """
## Beyond the course

TransformerLens names the attention pattern `blocks.0.attn.hook_pattern` and the residual stream `blocks.0.hook_resid_post`. The cell builds a one-layer `HookedTransformer` from a config, with no download, and reads those tensors. If the package is absent the cell skips. The scientific claim of the lab does not depend on the import. The import is the bridge to the library the papers' code is usually written in.
"""),
        ("code", r'''
try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except ImportError:
    print("SKIP TransformerLens import. Measurements above used the same quantities.")
    print("hook names to look up: blocks.0.attn.hook_pattern, blocks.0.hook_resid_post")
else:
    cfg = HookedTransformerConfig(
        n_layers=1,
        n_heads=1,
        d_model=32,
        d_head=32,
        d_mlp=64,
        d_vocab=V,
        n_ctx=T,
        act_fn="relu",
        normalization_type=None,
        attn_only=False,
    )
    hooked = HookedTransformer(cfg)
    tokens = torch.randint(0, V, (4, T))
    logits, cache = hooked.run_with_cache(tokens)
    pattern = cache["blocks.0.attn.hook_pattern"]
    resid = cache["blocks.0.hook_resid_post"]
    print("logits", tuple(logits.shape), "pattern", tuple(pattern.shape), "resid", tuple(resid.shape))
    assert pattern.shape[0] == 4
    assert resid.shape[-1] == 32
'''),
        ("md", """
## Research questions

1. The attention mass on position 0 is high. What extra experiment would you need before saying the head's output is a copy of that token, rather than a mix that the linear head untangles?
2. The current-token probe is near chance. Alain and Bengio's probe can be high in a layer the model does not use. How is that different from this control?
3. Anthropic's induction-head papers study a two-layer circuit that copies a previous occurrence of the current token. What would you add to this one-layer task to make that circuit necessary?

## Further reading

- Alain and Bengio, *Understanding Intermediate Layers Using Linear Classifier Probes* (2016).
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (Anthropic, 2021).
- Elhage et al., *Toy Models of Superposition* (Anthropic, 2022).
- Olsson et al., *In-context Learning and Induction Heads* (Anthropic, 2022).
- Bricken et al., *Towards Monosemanticity* (Anthropic, 2023).
- Nanda and Bloom, TransformerLens (MIT). https://github.com/TransformerLensOrg/TransformerLens
- Conmy et al., *Towards Automated Circuit Discovery* (DeepMind, 2023).
- The ARENA mechanistic-interpretability curriculum covers the same measurements at greater length. Those notebooks are not copied. https://www.arena.education/
"""),
    ]
    emit(
        "labs/interpretability/01_probing_and_attention/probing_and_attention.ipynb",
        cells,
        {"track": "interpretability", "lab": "probing", "needs_gpu": False, "needs_api_key": False},
    )


def rag_fundamentals():
    cells = [
        ("md", """
# RAG fundamentals

**Track:** RAG. **Level:** Intermediate. **CPU:** yes. **Keys:** no.

## Goals

- Split a short document into sentences and into fixed character windows.
- Retrieve with TF-IDF over a numpy store.
- Answer by quoting a chunk and citing its id, or abstain when the query shares no terms.

## Sources and license

Original notebook. The outline follows the retrieval-augmented generation setup in Lewis et al. (2020) and the introductory RAG labs of the Full Stack LLM bootcamp. Those labs are not copied. TF-IDF is the scikit-learn vectorizer (BSD-3), implemented here directly so the weighting is visible. The portfolio RAG lab is the production-shaped version of this pipeline. This one is the smaller mechanism.

## Concepts

A chunk is the unit a citation can point at. A fixed window can cut a fact in half. A sentence window usually does not. Retrieval here is lexical: a query with no shared tokens scores zero, and zero is the abstention signal. That is the wrong behavior for a paraphrase and the right behavior for a question the corpus does not contain. The answer function is extractive. It does not call a language model. A generative reader belongs on top of a retriever you have already measured.
"""),
        ("code", colab(["numpy"])),
        ("code", BOOT + """
import math
import re
from collections import Counter

import numpy as np

from labs._shared.seed import seed_everything

seed_everything(0)
TOKEN = re.compile(r"[a-z0-9]+")

def tokenize(text):
    return TOKEN.findall(text.lower())
"""),
        ("md", """
## Guided exercise: chunk, embed, retrieve, abstain

The document is original. One fact, the price, sits in a single sentence. A 24-character window is short enough to separate the product name from the price. The sentence chunk is not.
"""),
        ("code", r'''
DOCUMENT = (
    "The storage service is called lighthouse. "
    "Its monthly price is 12 dollars per seat. "
    "Backups run every night and are kept for 30 days."
)

def fixed_windows(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]

def sentences(text):
    return [part.strip() for part in text.split(". ") if part.strip()]

windows = fixed_windows(DOCUMENT, 24)
sents = sentences(DOCUMENT)
print("windows", windows)
print("sentences", sents)
assert any("lighthouse" in chunk and "12" not in chunk for chunk in windows)
assert any("lighthouse" in chunk and "12" in chunk for chunk in [" ".join(sents)])

class TfidfIndex:
    def fit(self, ids, texts):
        self.ids = list(ids)
        self.texts = list(texts)
        docs = [tokenize(text) for text in texts]
        df = Counter()
        for doc in docs:
            for term in set(doc):
                df[term] += 1
        self.idf = {term: math.log((1 + len(docs)) / (1 + df[term])) + 1.0 for term in df}
        self.vocab = {term: i for i, term in enumerate(sorted(self.idf))}
        self.matrix = self.embed(texts)

    def embed(self, texts):
        matrix = np.zeros((len(texts), max(len(self.vocab), 1)))
        for row, text in enumerate(texts):
            for term, count in Counter(tokenize(text)).items():
                col = self.vocab.get(term)
                if col is not None:
                    matrix[row, col] = count * self.idf[term]
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix

    def rank(self, query, k=3, minimum=0.05):
        vector = self.embed([query])[0]
        if np.linalg.norm(vector) == 0:
            return []
        scores = self.matrix @ vector
        order = np.argsort(-scores)
        hits = []
        for i in order:
            if scores[i] < minimum:
                break
            hits.append((self.ids[int(i)], float(scores[i]), self.texts[int(i)]))
            if len(hits) == k:
                break
        return hits

index = TfidfIndex()
index.fit([f"s{i}" for i in range(len(sents))], sents)
price = index.rank("monthly price per seat")
unknown = index.rank("zebrafish genome assembly quality")
print("price", price)
print("unknown", unknown)
assert price and "12" in price[0][2]
assert unknown == []

def answer(query):
    hits = index.rank(query)
    if not hits:
        return "I do not know."
    doc_id, _score, text = hits[0]
    return f"{text.strip('.')} [{doc_id}]"

print(answer("monthly price per seat"))
print(answer("zebrafish genome assembly quality"))
assert "[s" in answer("monthly price per seat")
assert "12" in answer("monthly price per seat")
assert answer("zebrafish genome assembly quality") == "I do not know."
print("LAB OK")
'''),
        ("md", """
## Your turn

Retrieve `nightly backups`. The hit should be the backups sentence, not the price sentence.

## Solution
"""),
        ("code", r'''
hits = index.rank("nightly backups")
print(hits)
assert hits and "30" in hits[0][2]
'''),
        ("md", """
## Beyond the course

Build the index on the fixed windows instead of the sentences and ask for the price using the word `lighthouse`. Report whether any returned window contains both the name and the price. The sentence index did. This is the chunking ablation in one query.
"""),
        ("code", r'''
window_index = TfidfIndex()
window_index.fit([f"w{i}" for i in range(len(windows))], windows)
window_hits = window_index.rank("lighthouse price", k=4, minimum=0.0)
print(window_hits)
both = [hit for hit in window_hits if "lighthouse" in hit[2] and "12" in hit[2]]
print("windows containing both facts", len(both))
assert both == []
'''),
        ("md", """
## Research questions

1. Abstention used a shared-token threshold. What relevant question does that rule refuse?
2. The quoted answer can be the wrong sentence with a legal citation. Which check from the portfolio RAG lab would catch it?
3. TF-IDF was refit on three sentences. What changes when the idf statistics come from a million documents that do not include this product?

## Further reading

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020).
- scikit-learn TF-IDF user guide (BSD-3).
- Full Stack LLM bootcamp, retrieval section. Not copied.
- The portfolio lab `labs/portfolio/01_production_rag` adds hybrid search, stores, and a citation check.
"""),
    ]
    emit(
        "labs/rag/01_rag_fundamentals/rag_fundamentals.ipynb",
        cells,
        {"track": "rag", "lab": "fundamentals", "needs_gpu": False, "needs_api_key": False},
    )


def advanced_rag():
    cells = [
        ("md", """
# Advanced RAG: hybrid search and an eval harness

**Track:** RAG. **Level:** Intermediate. **CPU:** yes.

## Goals

- Compute hit rate and mean reciprocal rank on a hand-checked example before using them on a corpus.
- Fuse BM25 and dense rankings with reciprocal rank fusion.
- Score a quoted answer with a lexical faithfulness check, and show a paraphrase the check rejects.

## Sources and license

Original notebook. Hit rate and MRR are the standard ranked-retrieval metrics used in the information-retrieval textbooks and in the BEIR benchmark writeups. Reciprocal rank fusion is Cormack, Clarke, and Buettcher (2009). The faithfulness check is a lexical baseline, the same idea as the portfolio lab, on a smaller corpus. LangChain and LlamaIndex cookbooks (Apache-2.0) describe this pipeline. Their notebooks are not copied.

## Concepts

Hit rate at `k` is 1 when any gold id appears in the first `k` rows. MRR is `1 / rank` of the first gold id, and 0 when it is missing. A single query with the gold item in position 2 has hit rate at 1 equal to 0, hit rate at 2 equal to 1, and MRR equal to 0.5. Fusion by rank ignores the fact that one retriever's scores are not on the other's scale. The faithfulness check asks whether the claim's content words and numbers appear in the cited chunk. It accepts a supported quote and rejects a true sentence that was paraphrased. Both outcomes are part of the result.
"""),
        ("code", colab(["numpy"])),
        ("code", BOOT + """
import math
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")

def tokenize(text):
    return TOKEN.findall(text.lower())

def hit_rate(ranked, gold, k):
    return int(any(doc_id in gold for doc_id in ranked[:k]))

def mrr(ranked, gold):
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0

assert hit_rate(["b", "a"], {"a"}, 1) == 0
assert hit_rate(["b", "a"], {"a"}, 2) == 1
assert mrr(["b", "a", "c"], {"a"}) == 0.5
assert mrr(["c"], {"a"}) == 0.0
print("hand example: gold a is second, MRR", mrr(["b", "a", "c"], {"a"}))
"""),
        ("md", """
## Guided exercise: fuse two rankings

BM25 is term frequency with an idf penalty. The dense vector is L2-normalized TF-IDF. One query is written with `yield` where the document says `interest`. BM25 on the raw query misses. The expanded query, and the hybrid list, hit.
"""),
        ("code", r'''
DOCS = {
    "savings": "The savings account yields 3 percent annual interest and has no monthly fee.",
    "card": "The credit card has an annual fee of 49 dollars.",
    "wire": "An international wire costs 15 dollars.",
    "loan": "Personal loans carry a fixed interest rate of 7 percent.",
}
GLOSSARY = {"yield": "interest", "yearly": "annual"}

def expand(query):
    extra = [GLOSSARY[tok] for tok in tokenize(query) if tok in GLOSSARY]
    return query if not extra else query + " " + " ".join(extra)

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.ids = list(docs)
        self.tokenized = [tokenize(docs[i]) for i in self.ids]
        self.df = Counter(term for doc in self.tokenized for term in set(doc))
        self.tf = [Counter(doc) for doc in self.tokenized]
        self.lengths = [len(doc) or 1 for doc in self.tokenized]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        self.k1, self.b = k1, b

    def rank(self, query, k=4):
        scored = []
        q = tokenize(query)
        for i, doc_id in enumerate(self.ids):
            score = 0.0
            for term in q:
                freq = self.tf[i].get(term, 0)
                if not freq:
                    continue
                idf = math.log(1 + (len(self.ids) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                score += idf * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda item: -item[1])
        return scored[:k]

class Dense:
    def __init__(self, docs):
        self.ids = list(docs)
        self.texts = [docs[i] for i in self.ids]
        tokenized = [tokenize(text) for text in self.texts]
        df = Counter(term for doc in tokenized for term in set(doc))
        self.idf = {t: math.log((1 + len(self.ids)) / (1 + df[t])) + 1 for t in df}
        self.vocab = {t: i for i, t in enumerate(sorted(self.idf))}
        self.matrix = self._embed(self.texts)

    def _embed(self, texts):
        m = np.zeros((len(texts), len(self.vocab)))
        for r, text in enumerate(texts):
            for term, count in Counter(tokenize(text)).items():
                j = self.vocab.get(term)
                if j is not None:
                    m[r, j] = count * self.idf[term]
            n = np.linalg.norm(m[r])
            if n:
                m[r] /= n
        return m

    def rank(self, query, k=4):
        v = self._embed([query])[0]
        if np.linalg.norm(v) == 0:
            return []
        scores = self.matrix @ v
        order = np.argsort(-scores)
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0][:k]

def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: -item[1])]

bm25, dense = BM25(DOCS), Dense(DOCS)
QUESTIONS = [
    ("savings account annual interest", {"savings"}),
    ("credit card annual fee", {"card"}),
    ("international wire cost", {"wire"}),
    ("yearly yield", {"savings"}),
]
rows = []
for query, gold in QUESTIONS:
    expanded = expand(query)
    sparse = [doc for doc, _ in bm25.rank(expanded)]
    neural = [doc for doc, _ in dense.rank(expanded)]
    fused = rrf([sparse, neural])
    raw = [doc for doc, _ in bm25.rank(query)]
    rows.append((query, hit_rate(raw, gold, 3), hit_rate(fused, gold, 3), mrr(fused, gold), fused[:2]))
for row in rows:
    print(row)
assert rows[0][2] == 1 and rows[3][1] == 0 and rows[3][2] == 1
mean_mrr = sum(row[3] for row in rows) / len(rows)
print("mean hybrid MRR", round(mean_mrr, 3))
assert mean_mrr > 0.5
'''),
        ("md", """
## Your turn

The synonym query is the last row. Confirm the raw BM25 hit rate is 0 and the fused hit rate is 1.

## Solution
"""),
        ("code", r'''
query, raw_hit, fused_hit, reciprocal, top = rows[-1]
print(query, "raw", raw_hit, "fused", fused_hit, "mrr", reciprocal, top)
assert raw_hit == 0 and fused_hit == 1
'''),
        ("md", """
## Beyond the course

Quote the savings sentence and cite it. The lexical check passes. A paraphrase that preserves the 3 percent fact and shares almost no content words fails the same check. Report both. The failure is the reason a later faithfulness model has to understand the sentence rather than its tokens.

The check also rejects a claim whose number is not in the chunk.
"""),
        ("code", r'''
STOP = {"the", "a", "an", "of", "and", "to", "is", "in", "on", "has", "no"}

def supported(claim, source, minimum=0.4):
    if set(re.findall(r"\d+", claim)) - set(re.findall(r"\d+", source)):
        return False
    claim_terms = set(tokenize(claim)) - STOP
    source_terms = set(tokenize(source)) - STOP
    if not claim_terms:
        return False
    return len(claim_terms & source_terms) / len(claim_terms) >= minimum

source = DOCS["savings"]
quote = "The savings account yields 3 percent annual interest."
paraphrase = "Cash in that product earns three percent a year."
wrong_number = "The savings account yields 9 percent annual interest."
print("quote", supported(quote, source), "paraphrase", supported(paraphrase, source), "wrong number", supported(wrong_number, source))
assert supported(quote, source)
assert not supported(paraphrase, source)
assert not supported(wrong_number, source)
print("LAB OK")
'''),
        ("md", """
## Research questions

1. MRR treats the second position as half a point. For a user who only reads the top hit, which of hit rate at 1 or MRR matches the product?
2. Rank fusion recovered the synonym query because of the glossary, not because of the dense geometry. What would you change in the embedder before removing the glossary?
3. The paraphrase was true and was rejected. Construct a false sentence this check would accept. The portfolio lab asks the same question of a larger corpus.

## Further reading

- Cormack, Clarke, and Buettcher, *Reciprocal Rank Fusion* (SIGIR 2009).
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval* (2021).
- Lewis et al., RAG (2020). Nogueira and Cho, passage reranking with BERT (2019).
- LangChain and LlamaIndex retrieval docs (Apache-2.0). Not copied.
"""),
    ]
    emit(
        "labs/rag/02_advanced_rag/advanced_rag.ipynb",
        cells,
        {"track": "rag", "lab": "advanced", "needs_gpu": False, "needs_api_key": False},
    )


def prompting():
    cells = [
        ("md", """
# Prompting, structured output, and tools

**Track:** GenAI applications. **Level:** Intermediate. **CPU:** yes. **Keys:** optional.

## Goals

- Run a zero-shot prompt, a few-shot prompt, and a chain-of-thought prompt through one client.
- Validate a JSON object against required keys and repair it when a key is missing.
- Run a tool loop whose only tools are a calculator and a unit converter.

## Sources and license

Original notebook. The prompt patterns are the ones taught in the Hugging Face LLM course and the Full Stack LLM bootcamp. Neither course notebook is copied. Tool use follows the action/observation loop in Yao et al., *ReAct* (2022). The client is `labs/_shared/llm.py`. The default provider is the local fallback so the notebook is reproducible. Set `LAB_LLM=openai` or `LAB_LLM=anthropic` to send the same messages to a hosted model, or `LAB_LLM=huggingface` to sample a local Transformers model. The repair demo forces the local provider, because the first missing field has to be deterministic.

## Concepts

A prompt is an input to a function with an unstable output. The harness around it is the part you can test: the template, the schema, and the tool allow-list. Structured output means parsing JSON and checking keys, then sending the error back as the next message. A tool loop alternates model actions and tool observations and stops on `final` or on a step limit. The local provider is not a language model. It returns scripted strings so the harness can be graded without a key. The Hugging Face provider in this lab is an untrained tiny GPT-2 unless `HF_MODEL` names a checkpoint. Its text is not graded.
"""),
        ("code", colab(["torch", "transformers"])),
        ("code", BOOT + """
import json
import os

from labs._shared.llm import get_llm

provider = os.environ.get("LAB_LLM", "local")
llm = get_llm(provider)
print("provider", llm.name)
"""),
        ("md", """
## Guided exercise: three prompts, one client

The local provider answers the France question with `Paris`, and with a short derivation when the prompt says `step by step`. A hosted model receives the same strings. The assertions run only for the local provider. A hosted run prints the text and checks that it is non-empty.
"""),
        ("code", r'''
zero = llm.complete([{"role": "user", "content": "What is the capital of France? Answer in one word."}])
few = llm.complete([{
    "role": "user",
    "content": "Italy: Rome\nSpain: Madrid\nWhat is the capital of France?",
}])
cot = llm.complete([{
    "role": "user",
    "content": "What is the capital of France? Reason step by step, then give the answer.",
}])
print("zero-shot:", zero)
print("few-shot:", few)
print("chain-of-thought:", cot)
if llm.name == "local-fallback":
    assert "Paris" in zero and "Paris" in few and "Paris" in cot
    assert "Answer:" in cot
else:
    assert zero and few and cot
'''),
        ("md", """
## Guided exercise: schema and repair

Required keys are `city` and `country`. The fresh local client omits `country` on the first call when the prompt contains `repair-demo`, then returns both keys. The loop records the missing-key error. That error is the thing under test.
"""),
        ("code", r'''
def complete_json(client, prompt, keys, retries=3):
    messages = [{"role": "user", "content": prompt}]
    errors = []
    for _ in range(retries):
        raw = client.complete(messages, mode="json")
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": prompt + " RETURN_JSON"})
            continue
        missing = [key for key in keys if key not in obj]
        if missing:
            errors.append("missing " + ",".join(missing))
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": prompt + " RETURN_JSON include " + ",".join(keys),
            })
            continue
        return obj, errors
    raise RuntimeError(errors)

repair_client = get_llm("local")
obj, errors = complete_json(
    repair_client,
    "repair-demo RETURN_JSON Paris is the city. Name the city and the country.",
    ["city", "country"],
)
print(obj, errors)
assert errors and "country" in errors[0]
assert obj == {"city": "Paris", "country": "France"}
'''),
        ("md", """
## Guided exercise: a tool loop

The calculator accepts arithmetic. The converter accepts a fixed table of units. Anything else raises, and the loop records the error instead of calling `eval` on it. The transcript uses the markers the local provider understands (`calc:`, `convert:`). A hosted model would be given the same JSON action schema in the system message. This cell stays on the local provider so the trace is stable.
"""),
        ("code", r'''
def calculator(expr: str) -> str:
    import re
    if not re.fullmatch(r"[0-9+*/(). -]+", expr):
        raise ValueError("rejected")
    return str(eval(expr, {"__builtins__": {}}, {}))

def convert(spec: dict) -> str:
    factors = {("km", "m"): 1000.0, ("m", "km"): 0.001, ("kg", "g"): 1000.0, ("g", "kg"): 0.001}
    factor = factors[(spec["src"], spec["dst"])]
    value = float(spec["value"]) * factor
    return str(int(value)) if float(value).is_integer() else str(value)

TOOLS = {"calculator": calculator, "convert": convert}

def run_tools(prompt, client, max_steps=4):
    messages = [{"role": "user", "content": prompt + "\nTOOL_LOOP"}]
    trace = []
    for _ in range(max_steps):
        action = json.loads(client.complete(messages, mode="tools"))
        trace.append(action["action"])
        if action["action"] == "final":
            return trace, str(action["input"])
        observation = TOOLS[action["action"]](action["input"])
        messages.append({"role": "assistant", "content": json.dumps(action)})
        messages.append({"role": "user", "content": f"Observation ({action['action']}): {observation}\nTOOL_LOOP"})
    return trace, ""

agent = get_llm("local")
trace, answer = run_tools("calc: (12 + 5) * 3", agent)
print(trace, answer)
assert trace == ["calculator", "final"] and answer == "51"
trace, answer = run_tools("convert: 1 km to m", agent)
print(trace, answer)
assert trace == ["convert", "final"] and answer == "1000"
try:
    calculator("__import__('os')")
    raise AssertionError("sandbox failed")
except ValueError:
    print("sandbox rejected a non-arithmetic expression")
'''),
        ("md", """
## Your turn

Ask the local client to convert 2 kg to g. The trace should be `convert`, then `final`, and the answer should be 2000.

## Solution
"""),
        ("code", r'''
trace, answer = run_tools("convert: 2 kg to g", get_llm("local"))
print(trace, answer)
assert trace == ["convert", "final"] and answer == "2000"
'''),
        ("md", """
## Beyond the course

Call the Hugging Face provider. With no `HF_MODEL` set, this builds an untrained tiny GPT-2 and samples eight characters. The string is prefixed with the checkpoint name. Assert that prefix, and do not assert that the sample means anything. Setting `HF_MODEL` to a public instruct model is how the same function calls a real open model. That download is outside CI.
"""),
        ("code", r'''
hf = get_llm("huggingface")
text = hf.complete([{"role": "user", "content": "Say hello in one short sentence."}])
print(text)
assert text.startswith("[untrained-tiny-gpt2]") or text.startswith("[" + hf.checkpoint + "]")
print("LAB OK")
'''),
        ("md", """
## Research questions

1. The repair loop trusts the second JSON object once the keys exist. What can a model put in `country` that still passes, and that you would refuse to store?
2. The tool allow-list is two functions. Where in `run_tools` would a third tool name returned by a hosted model be stopped?
3. The chain-of-thought string and the one-word string can name the same city. Which one do you want when you are grading the answer, and which one do you want when you are grading the trace?

## Further reading

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022).
- Hugging Face LLM course, the prompting and agents chapters (Apache-2.0 code). Not copied. https://huggingface.co/learn/llm-course
- Full Stack LLM bootcamp, the prompting and evaluation labs. Not copied.
- OpenAI and Anthropic tool-use guides, for the JSON action shape a hosted model is asked to emit.
"""),
    ]
    emit(
        "labs/genai_apps/01_prompting_and_tools/prompting_and_tools.ipynb",
        cells,
        {"track": "genai_apps", "lab": "prompting", "needs_gpu": False, "needs_api_key": False},
    )


if __name__ == "__main__":
    lora()
    dpo()
    interpret()
    rag_fundamentals()
    advanced_rag()
    prompting()
