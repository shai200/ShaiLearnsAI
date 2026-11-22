"""Tokenization, transformers, fine-tuning, interpretability, RAG, and tool use."""

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


TINY = r'''
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        mixed = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(mixed)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """A nanoGPT-style decoder. Original code, same block layout as nanoGPT (MIT)."""

    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=32):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, loss_mask=None):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            per = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).view(b, t)
            if loss_mask is None:
                loss = per.mean()
            else:
                loss = (per * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8):
        self.eval()
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.block_size :])
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def train_lm(data, vocab_size, steps, n_embd=64, n_layer=2, n_head=4, block_size=32, batch_size=32, lr=3e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = TinyGPT(vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=g)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return model, losses
'''


def bpe():
    cells = [
        ("md", """
# Byte-pair encoding

**Track:** Training LLMs. **Level:** Intermediate. **CPU:** yes.

## Goals

- Train a byte-level BPE merge table on a small corpus.
- Encode and decode back to the same Unicode string, including a Hebrew sentence.
- Compare the token count with the byte count on repeated text.

## Sources and license

Original implementation of the algorithm in Sennrich, Haddow, and Birch, *Neural Machine Translation of Rare Words with Subword Units* (2016). The training loop follows the design of Karpathy's minbpe (MIT). That repository is not copied. Byte-level BPE is the scheme used by GPT-2.

## Concepts

The corpus is a sequence of bytes, so any Unicode string is in vocabulary from the start. Each merge replaces the most frequent adjacent pair with a new id. Encoding applies the learned merges from earliest to latest, which is the lowest id among the pairs that still occur. Decoding walks the merge table backwards into bytes. Compression comes from merges that repeat. A language with multi-byte characters starts further from one token per character, which is a property of UTF-8, not of the merge rule.
"""),
        ("code", colab(["numpy"])),
        ("code", BOOT + """
from collections import Counter

from labs._shared.seed import seed_everything

seed_everything(0)
"""),
        ("md", """
## Guided exercise: train, encode, decode
"""),
        ("code", r'''
def pair_counts(seqs):
    counts = Counter()
    for seq in seqs:
        for left, right in zip(seq, seq[1:]):
            counts[(left, right)] += 1
    return counts


def apply_merge(seq, pair, new_id):
    out = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_bpe(text: str, num_merges: int) -> dict:
    seqs = [list(text.encode("utf-8"))]
    merges = {}
    for i in range(num_merges):
        stats = pair_counts(seqs)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        new_id = 256 + i
        seqs = [apply_merge(seq, pair, new_id) for seq in seqs]
        merges[pair] = new_id
    return merges


def encode(text: str, merges: dict) -> list[int]:
    seq = list(text.encode("utf-8"))
    while True:
        stats = pair_counts([seq])
        present = [pair for pair in stats if pair in merges]
        if not present:
            break
        pair = min(present, key=lambda item: merges[item])
        seq = apply_merge(seq, pair, merges[pair])
    return seq


def decode(ids: list[int], merges: dict) -> str:
    vocab = {i: bytes([i]) for i in range(256)}
    for (left, right), new_id in sorted(merges.items(), key=lambda item: item[1]):
        vocab[new_id] = vocab[left] + vocab[right]
    return b"".join(vocab[i] for i in ids).decode("utf-8")


corpus = ("the cat sat on the mat. the cat sat on the hat. " * 40)
merges = train_bpe(corpus, 30)
roundtrip = decode(encode(corpus, merges), merges)
print("merges", len(merges), "bytes", len(corpus.encode()), "tokens", len(encode(corpus, merges)))
print("first merges", list(merges.items())[:5])
assert roundtrip == corpus
assert len(encode(corpus, merges)) < len(corpus.encode())

hebrew = "חשבון החיסכון מניב ריבית שנתית"
hebrew_merges = train_bpe(hebrew * 6, 15)
assert decode(encode(hebrew, hebrew_merges), hebrew_merges) == hebrew
assert len(hebrew.encode("utf-8")) > len(hebrew)
print("LAB OK")
'''),
        ("md", """
## Your turn

Encode the English sentence and the Hebrew sentence with a merge table trained on the concatenated parallel text. Both must round-trip. Print tokens per whitespace-separated word. That ratio is called fertility in the tokenization literature.

## Solution
"""),
        ("code", r'''
parallel = "the savings account yields interest. חשבון החיסכון מניב ריבית. " * 10
merges_p = train_bpe(parallel, 40)
en = "the savings account yields interest."
he = "חשבון החיסכון מניב ריבית."
for label, sentence in (("en", en), ("he", he)):
    ids = encode(sentence, merges_p)
    words = sentence.split()
    print(label, "tokens", len(ids), "words", len(words), "fertility", round(len(ids) / len(words), 2))
    assert decode(ids, merges_p) == sentence
'''),
        ("md", """
## Beyond the course

Train two merge tables, one on English only and one on the parallel text, and encode the Hebrew sentence with the English-only table. It still round-trips, because unknown bytes stay bytes. The token count is the thing that changes. A tokenizer that never saw Hebrew spends an id per byte.
"""),
        ("code", r'''
english_only = train_bpe("the savings account yields interest. " * 30, 40)
ids_en_table = encode(he, english_only)
ids_bi_table = encode(he, merges_p)
print("hebrew tokens, english merges", len(ids_en_table), "parallel merges", len(ids_bi_table))
assert decode(ids_en_table, english_only) == he
assert len(ids_en_table) >= len(ids_bi_table)
'''),
        ("md", """
## Research questions

1. Fertility was higher for one of the two sentences. Is that the language, the script, or the fact that the parallel corpus contained more English repeats?
2. GPT-2's byte-level BPE has a regex pre-tokenizer that keeps most punctuation separate. What bug appears in this lab's encoder because it does not?
3. A merge learned from one domain will happily segment another domain into bytes. Where does that show up in a multilingual model's loss, as opposed to in the round-trip test?

## Further reading

- Sennrich, Haddow, and Birch, *Neural Machine Translation of Rare Words with Subword Units* (ACL 2016).
- Karpathy, minbpe (MIT). https://github.com/karpathy/minbpe. Design reference, not copied.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019), on byte-level BPE.
"""),
    ]
    emit(
        "labs/training_llms/01_tokenization_bpe/tokenization_bpe.ipynb",
        cells,
        {"track": "training_llms", "lab": "bpe", "needs_gpu": False, "needs_api_key": False},
    )


def attention():
    cells = [
        ("md", """
# Embeddings and attention

**Track:** Training LLMs. **Level:** Intermediate. **CPU:** yes.

## Goals

- Build word vectors from a co-occurrence matrix and check a nearest-neighbor relation you can see in the corpus.
- Implement scaled dot-product attention, a causal mask, and a padding mask.
- See what a temperature does to the attention weights.

## Sources and license

Original notebook. The embedding section follows the co-occurrence / PPMI / SVD line of Levy and Goldberg and of the word2vec papers, which is also how Stanford CS224N introduces distributional vectors. CS224N assignment code is not copied. Attention is the scaled dot-product from Vaswani et al., *Attention Is All You Need* (2017). The formulas are implemented from the paper, not from a course repository.

## Concepts

A word vector here is a low-rank view of "which words appear nearby." PPMI up-weights pairs that occur more than independence would predict and clips negative values at zero. SVD keeps the directions with the largest singular values. Attention is a weighted average of value vectors. The weights are a softmax of query-key dot products, divided by `sqrt(d)` so the variance of the scores does not grow with the key dimension. A causal mask sets future scores to `-inf` before the softmax. A padding mask does the same for positions that are not real tokens.
"""),
        ("code", colab(["numpy", "torch"])),
        ("code", BOOT + """
import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from labs._shared.seed import seed_everything

seed_everything(0)
"""),
        ("md", """
## Guided exercise: co-occurrence embeddings

The corpus repeats two topics. `river` should land nearer `water` than `money`. `loan` should land nearer `interest` than `fish`. If either comparison fails, the window or the corpus is wrong, not the assertion.
"""),
        ("code", r'''
sentences = [
    "river water bank fish",
    "river water flow fish",
    "bank money account loan",
    "bank money interest account",
    "fish swim river water",
    "loan interest money account",
] * 8
vocab = sorted({word for sentence in sentences for word in sentence.split()})
index = {word: i for i, word in enumerate(vocab)}
counts = np.zeros((len(vocab), len(vocab)))
for sentence in sentences:
    tokens = sentence.split()
    for i, word in enumerate(tokens):
        for j in range(max(0, i - 2), min(len(tokens), i + 3)):
            if i != j:
                counts[index[word], index[tokens[j]]] += 1
row = counts.sum(axis=1, keepdims=True)
col = counts.sum(axis=0, keepdims=True)
total = counts.sum()
expected = row @ col / total
ppmi = np.zeros_like(counts)
mask = counts > 0
ppmi[mask] = np.maximum(np.log(counts[mask] / expected[mask]), 0.0)
u, s, _vt = np.linalg.svd(ppmi, full_matrices=False)
vectors = u[:, :4] * s[:4]

def cosine(a, b):
    va, vb = vectors[index[a]], vectors[index[b]]
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))

print("river~water", round(cosine("river", "water"), 3), "river~money", round(cosine("river", "money"), 3))
print("loan~interest", round(cosine("loan", "interest"), 3), "loan~fish", round(cosine("loan", "fish"), 3))
assert cosine("river", "water") > cosine("river", "money")
assert cosine("loan", "interest") > cosine("loan", "fish")
'''),
        ("md", """
## Guided exercise: scaled dot-product attention

`Q`, `K`, and `V` are shaped `(batch, time, dim)`. The causal mask is a lower triangle. Each row of the weight matrix sums to 1, and position 0 puts no mass on position 2.
"""),
        ("code", r'''
def attention(query, key, value, causal=False, key_padding=None):
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if causal:
        t = query.shape[-2]
        causal_mask = torch.tril(torch.ones(t, t, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    if key_padding is not None:
        # key_padding True means "this key position is padding".
        scores = scores.masked_fill(key_padding[:, None, :], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ value, weights

torch.manual_seed(0)
q = torch.randn(1, 3, 4)
k = torch.randn(1, 3, 4)
v = torch.randn(1, 3, 4)
out, weights = attention(q, k, v, causal=True)
print(weights[0].round(decimals=3))
assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 3), atol=1e-5)
assert float(weights[0, 0, 2]) == 0.0
'''),
        ("md", """
## Your turn

Pass a padding mask that marks the last key as padding. Its column of attention weights must be zero, and the remaining weights in each row must still sum to 1.

## Solution
"""),
        ("code", r'''
padding = torch.tensor([[False, False, True]])
_out, padded = attention(q, k, v, causal=False, key_padding=padding)
print(padded[0].round(decimals=3))
assert torch.allclose(padded[0, :, 2], torch.zeros(3), atol=1e-6)
assert torch.allclose(padded.sum(dim=-1), torch.ones(1, 3), atol=1e-5)
'''),
        ("md", """
## Beyond the course

Divide the scores by a temperature before the softmax. A larger temperature flattens the distribution, so the maximum weight falls. This is the same knob as the temperature in a language-model sampler. It is not a new parameter of the architecture.
"""),
        ("code", r'''
def max_weight(temperature):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return float(torch.softmax(scores / temperature, dim=-1).max())

cool, hot = max_weight(1.0), max_weight(4.0)
print("max weight T=1", round(cool, 3), "T=4", round(hot, 3))
assert cool > hot
print("LAB OK")
'''),
        ("md", """
## Research questions

1. `bank` occurred in both topics. Which neighbor did it actually get, and what would a sense-specific vector have to change in the count matrix?
2. The causal mask and the padding mask both write `-inf` into `scores`. What happens to a row whose keys are all masked?
3. Vaswani et al. scale by `sqrt(d)`. If you remove the scale and increase `d` from 4 to 64, what happens to the softmax before any training?

## Further reading

- Vaswani et al., *Attention Is All You Need* (2017).
- Levy and Goldberg, *Neural Word Embedding as Implicit Matrix Factorization* (2014).
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (2013).
- Stanford CS224N lecture notes on distributional semantics and on Transformers. The assignment code is not copied. https://web.stanford.edu/class/cs224n/
"""),
    ]
    emit(
        "labs/training_llms/02_embeddings_and_attention/embeddings_and_attention.ipynb",
        cells,
        {"track": "training_llms", "lab": "attention", "needs_gpu": False, "needs_api_key": False},
    )


def tiny_gpt():
    cells = [
        ("md", """
# A tiny GPT, trained from scratch

**Track:** Training LLMs. **Level:** Intermediate. **CPU:** yes.

## Goals

- Read a decoder-only Transformer: token embedding, position embedding, causal self-attention, MLP block, tied loss.
- Train it for a few dozen steps on a repeated sentence.
- Sample from the trained model and check that the loss moved.

## Sources and license

Original implementation of the nanoGPT layout (Karpathy, MIT): pre-norm residual blocks, causal multi-head attention, and a next-token cross-entropy. nanoGPT and llm.c are not copied. The corpus is original and repetitive so a CPU run of well under a minute can reduce the loss. Tiny Shakespeare, the usual nanoGPT demo corpus, is public domain. It is not vendored here.

## Concepts

The model defines a distribution over the next character given the previous characters, inside a window of `block_size`. The causal mask is what makes that distribution not depend on future characters. The loss is the mean cross-entropy of those next-character predictions. Generation draws from the softmax of the last position and appends the draw. A drop in loss on this corpus means the network assigned more probability to the characters that actually repeat. It does not mean the network has learned English.
"""),
        ("code", colab(["torch", "numpy", "matplotlib"])),
        ("code", BOOT + """
%matplotlib inline
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from labs._shared.seed import seed_everything

seed_everything(0)
torch.manual_seed(0)
""" + TINY),
        ("md", """
## Guided exercise: overfit a repeated pattern

The vocabulary is the set of characters in the text. Training uses random windows. The assertion compares the mean loss of the first five steps with the mean of the last five.
"""),
        ("code", r'''
text = ("the cat sat on the mat. the cat sat on the hat. the dog sat on the log. " * 50)
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
print("chars", len(text), "vocab", len(chars))
model, losses = train_lm(
    data, vocab_size=len(chars), steps=80, n_embd=64, n_layer=2, n_head=4, block_size=32, batch_size=32, lr=3e-3,
)
print("loss start", round(float(np.mean(losses[:5])), 3), "loss end", round(float(np.mean(losses[-5:])), 3))
assert np.mean(losses[-5:]) < 0.75 * np.mean(losses[:5])
start = torch.tensor([[stoi["t"]]])
sample = model.generate(start, n_new=60, temperature=0.7)[0].tolist()
print("".join(itos[i] for i in sample))

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(losses)
ax.set_xlabel("step")
ax.set_ylabel("cross-entropy")
ax.set_title("Tiny GPT on a repeated sentence")
plt.tight_layout()
plt.show()
print("LAB OK")
'''),
        ("md", """
## Your turn

Count parameters. For this configuration the count is fixed. Print it and check that the token embedding and the output head are separate tensors. nanoGPT often ties them. This lab does not, so you can see the two shapes.

## Solution
"""),
        ("code", r'''
n_params = sum(p.numel() for p in model.parameters())
print("parameters", n_params)
print("token embedding", tuple(model.tok.weight.shape), "lm head", tuple(model.head.weight.shape))
assert model.tok.weight.data_ptr() != model.head.weight.data_ptr()
assert n_params > 10_000
'''),
        ("md", """
## Beyond the course

Train the same data for the same number of steps with one layer and with two layers. Both should reduce the loss. The comparison is which one finishes lower at this step budget, not which architecture is better in general. A deeper model has more compute per step, so a lower loss is not a free win.
"""),
        ("code", r'''
_, one = train_lm(data, len(chars), steps=40, n_embd=64, n_layer=1, n_head=4, block_size=32, seed=1)
_, two = train_lm(data, len(chars), steps=40, n_embd=64, n_layer=2, n_head=4, block_size=32, seed=1)
print("1 layer", round(float(np.mean(one[-5:])), 3), "2 layers", round(float(np.mean(two[-5:])), 3))
assert np.mean(one[-5:]) < np.mean(one[:5])
assert np.mean(two[-5:]) < np.mean(two[:5])
'''),
        ("md", """
## Research questions

1. The sample is still mostly the training sentence. Which failure would tell you the model is copying the window rather than estimating a next-character distribution?
2. Temperature 0.7 was used for the sample and 1 is implicit in the loss. Why can the loss fall while a high-temperature sample still looks random?
3. What would you have to log, besides the scalar loss, before claiming the attention heads specialized?

## Further reading

- Karpathy, nanoGPT (MIT). https://github.com/karpathy/nanoGPT. Layout reference, not copied.
- Karpathy, llm.c (MIT), the same model in C.
- Vaswani et al., *Attention Is All You Need* (2017).
- Radford et al., GPT-2 technical report (2019).
"""),
    ]
    emit(
        "labs/training_llms/03_tiny_gpt/tiny_gpt.ipynb",
        cells,
        {"track": "training_llms", "lab": "tiny_gpt", "needs_gpu": False, "needs_api_key": False},
    )


def scaling():
    cells = [
        ("md", """
# A small scaling experiment

**Track:** Training LLMs. **Level:** Intermediate. **CPU:** yes.

## Goals

- Train the tiny GPT at three widths on the same data and the same step count.
- Train one width on a quarter of the data.
- Fit a line to log loss versus log parameter count, and say what the line is not.

## Sources and license

Original experiment. The questions come from Kaplan et al., *Scaling Laws for Neural Language Models* (OpenAI, 2020) and Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, DeepMind, 2022). No figure or code from those papers is reproduced. The model is the tiny GPT from the previous lab.

## Concepts

A scaling law, in those papers, is a power-law fit of loss against model size, data, or compute, measured at convergence or at a compute-optimal stop. This lab is smaller than that in every way. The step count is fixed, so a wider model also uses more compute per step. The data is a repeated sentence, so "more data" means more copies of the same pattern, not more diversity. The plot is still the right shape of experiment: change one axis, record the loss, and do not claim a law the design cannot support.
"""),
        ("code", colab(["torch", "numpy", "matplotlib"])),
        ("code", BOOT + """
%matplotlib inline
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from labs._shared.seed import seed_everything

seed_everything(0)
""" + TINY + """
text = ("the cat sat on the mat. the cat sat on the hat. the dog sat on the log. " * 50)
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
full = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
quarter = full[: max(200, len(full) // 4)]
print("tokens", len(full), "quarter", len(quarter), "vocab", len(chars))
"""),
        ("md", """
## Guided exercise: width and data

Parameter count is measured, not assumed. The compute proxy is `steps * layers * width^2`. It is not FLOPs. It is enough to show that the widest run did more work per step.
"""),
        ("code", r'''
def final_loss(data, n_embd, n_layer, steps=40, seed=0):
    model, losses = train_lm(
        data,
        vocab_size=len(chars),
        steps=steps,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=4,
        block_size=32,
        batch_size=32,
        lr=3e-3,
        seed=seed,
    )
    n_params = sum(p.numel() for p in model.parameters())
    return float(np.mean(losses[-5:])), n_params, float(np.mean(losses[:5]))

rows = []
for n_embd, n_layer in ((32, 1), (64, 2), (96, 2)):
    end, n_params, start = final_loss(full, n_embd, n_layer, seed=0)
    rows.append({
        "width": n_embd,
        "layers": n_layer,
        "params": n_params,
        "loss": end,
        "start": start,
        "data": len(full),
        "compute_proxy": 40 * n_layer * n_embd ** 2,
    })
end_q, params_q, start_q = final_loss(quarter, 64, 2, seed=0)
rows.append({
    "width": 64, "layers": 2, "params": params_q, "loss": end_q, "start": start_q,
    "data": len(quarter), "compute_proxy": 40 * 2 * 64 ** 2,
})
for row in rows:
    print(
        f"width {row['width']:3d} layers {row['layers']} data {row['data']:4d} "
        f"params {row['params']:7d} loss {row['loss']:.3f}"
    )
size_rows = [row for row in rows if row["data"] == len(full)]
assert size_rows[-1]["loss"] < size_rows[0]["loss"]
assert size_rows[-1]["compute_proxy"] > size_rows[0]["compute_proxy"]
full_loss = next(row["loss"] for row in rows if row["width"] == 64 and row["data"] == len(full))
assert full_loss <= end_q + 0.05

log_params = np.log(np.array([row["params"] for row in size_rows], dtype=float))
log_loss = np.log(np.array([row["loss"] for row in size_rows], dtype=float))
slope, intercept = np.polyfit(log_params, log_loss, 1)
print(f"log(loss) ~ {slope:.3f} * log(params) + {intercept:.3f}")

fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(log_params, log_loss)
xs = np.linspace(log_params.min(), log_params.max(), 20)
ax.plot(xs, slope * xs + intercept)
ax.set_xlabel("log parameters")
ax.set_ylabel("log final loss")
ax.set_title("Fixed steps, repeated sentence")
plt.tight_layout()
plt.show()
print("LAB OK")
'''),
        ("md", """
## Your turn

The quarter-data run uses the medium model. Print its loss next to the full-data medium model and the ratio of their token counts.

## Solution
"""),
        ("code", r'''
print("full", round(full_loss, 3), "quarter", round(end_q, 3), "token ratio", round(len(full) / len(quarter), 2))
assert len(full) > len(quarter)
'''),
        ("md", """
## Beyond the course

Chinchilla's claim is about the loss at the compute-optimal allocation, not at a shared step count. Recompute the proxy and state the comparison this lab cannot make: we did not match compute across widths, and we did not train to a loss plateau. The slope above is a description of these three runs.
"""),
        ("code", r'''
print("compute proxy by width", [(row["width"], row["compute_proxy"]) for row in size_rows])
print("not a Chinchilla frontier: step count is shared, so proxy grows with width**2")
assert all(row["loss"] < row["start"] for row in size_rows)
'''),
        ("md", """
## Research questions

1. The widest model finished lower. How much of that is extra parameters, and how much is the extra compute in the proxy column?
2. The quarter corpus is a prefix of the same repeated sentence. What would "less data" mean if the corpus were a real mixture of documents?
3. Kaplan et al. fit loss against compute with a power law on models that are orders of magnitude larger. Which assumption of that fit is visibly false on three points?

## Further reading

- Kaplan et al., *Scaling Laws for Neural Language Models* (OpenAI, 2020).
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (DeepMind, 2022).
- Bahri et al., *Explaining Neural Scaling Laws* (2021).
"""),
    ]
    emit(
        "labs/training_llms/04_scaling_laws/scaling_laws.ipynb",
        cells,
        {"track": "training_llms", "lab": "scaling", "needs_gpu": False, "needs_api_key": False},
    )


def sft():
    cells = [
        ("md", """
# Supervised fine-tuning

**Track:** Training LLMs. **Level:** Intermediate. **CPU:** yes.

## Goals

- Format a prompt and a response as one token sequence.
- Mask the loss so it is computed only on the response tokens.
- Overfit a handful of pairs, then try a color the model never trained on.

## Sources and license

Original notebook. The loss mask is the supervised-fine-tuning objective in Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT, OpenAI, 2022): predict the answer tokens, not the copied prompt. No InstructGPT data is used. The pairs below were written for this lab.

## Concepts

A causal language model scores every next token. If the prompt is included in the loss, the model is rewarded for reproducing the user's text. SFT sets that reward to zero with a mask. The numerical value of the loss is then the cross-entropy of the answer tokens only. Memorizing eight pairs is a check that the mask and the optimizer are wired up. It is not instruction following. A held-out word that shares a meaning with a training word is the test that separates memorization from the behavior the papers are about.
"""),
        ("code", colab(["torch", "numpy"])),
        ("code", BOOT + """
import math

import torch
import torch.nn as nn

from labs._shared.seed import seed_everything

seed_everything(0)
torch.manual_seed(0)
""" + TINY),
        ("md", """
## Guided exercise: mask and memorization

Each example is `user : <color> assistant : <response>`. The mask is 1 only on the response token. Greedy decoding reads the logits at the last prompt position.
"""),
        ("code", r'''
PAIRS = [
    ("red", "stop"),
    ("green", "go"),
    ("yellow", "wait"),
    ("blue", "sea"),
    ("cat", "meow"),
    ("dog", "bark"),
    ("bird", "sing"),
    ("cow", "moo"),
]
words = ["user", ":", "assistant"] + [w for pair in PAIRS for w in pair] + ["crimson"]
stoi = {word: i for i, word in enumerate(words)}
itos = {i: word for word, i in stoi.items()}

def example(src, dst):
    ids = [stoi["user"], stoi[":"], stoi[src], stoi["assistant"], stoi[":"], stoi[dst]]
    x = torch.tensor(ids[:-1])
    y = torch.tensor(ids[1:])
    mask = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    return x, y, mask

batch = [example(src, dst) for src, dst in PAIRS]
X = torch.stack([row[0] for row in batch])
Y = torch.stack([row[1] for row in batch])
M = torch.stack([row[2] for row in batch])
assert int(M.sum()) == len(PAIRS)
assert int(M[:, :-1].sum()) == 0

model = TinyGPT(len(stoi), n_embd=32, n_head=2, n_layer=1, block_size=8)
opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
losses = []
for _ in range(250):
    _, loss = model(X, Y, loss_mask=M)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
print("masked loss", round(losses[0], 3), "->", round(losses[-1], 3))
assert losses[-1] < losses[0] * 0.5

@torch.no_grad()
def predict(src: str) -> str:
    ids = torch.tensor([[stoi["user"], stoi[":"], stoi[src], stoi["assistant"], stoi[":"]]])
    logits, _ = model(ids)
    return itos[int(logits[0, -1].argmax())]

correct = sum(predict(src) == dst for src, dst in PAIRS) / len(PAIRS)
print("train exact match", correct, {src: predict(src) for src, _dst in PAIRS})
assert correct >= 0.75
print("LAB OK")
'''),
        ("md", """
## Your turn

Run the same batch with a mask of all ones, for 30 steps, from a fresh model. The loss should fall. That run is ordinary language-model training on the concatenated string. It is not the SFT objective, because the prompt tokens contribute.

## Solution
"""),
        ("code", r'''
fresh = TinyGPT(len(stoi), n_embd=32, n_head=2, n_layer=1, block_size=8)
opt2 = torch.optim.AdamW(fresh.parameters(), lr=5e-3)
unmasked = []
ones = torch.ones_like(M)
for _ in range(30):
    _, loss = fresh(X, Y, loss_mask=ones)
    opt2.zero_grad(set_to_none=True)
    loss.backward()
    opt2.step()
    unmasked.append(float(loss.detach()))
print("unmasked", round(unmasked[0], 3), "->", round(unmasked[-1], 3))
assert unmasked[-1] < unmasked[0]
'''),
        ("md", """
## Beyond the course

`crimson` is in the vocabulary and was never a training prompt. The training colors that mean "stop" are only `red`. Report the greedy prediction for `crimson` and the training exact-match rate. A gap is the expected result of this objective on eight memorized pairs.
"""),
        ("code", r'''
heldout = predict("crimson")
print("crimson ->", heldout, "train exact match", correct)
assert heldout in itos.values()
'''),
        ("md", """
## Research questions

1. The mask throws away the gradient on the prompt. When would you want a small amount of loss on the prompt anyway?
2. Exact match on eight pairs can be 1 while `crimson` is wrong. What data would you add before treating "color to action" as a learned rule?
3. InstructGPT combines this loss with a preference model. Which failure of the mask does the preference model not fix?

## Further reading

- Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback* (OpenAI, 2022).
- Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN, Google, 2021).
- The SFT loss in Hugging Face TRL (Apache-2.0) is the production version of this mask. Not copied.
"""),
    ]
    emit(
        "labs/training_llms/05_supervised_finetuning/supervised_finetuning.ipynb",
        cells,
        {"track": "training_llms", "lab": "sft", "needs_gpu": False, "needs_api_key": False},
    )


if __name__ == "__main__":
    bpe()
    attention()
    tiny_gpt()
    scaling()
    sft()
