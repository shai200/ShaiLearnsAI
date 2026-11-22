# Training LLMs & Frontier Research

*A hands-on track of eight runnable notebooks covering how frontier language models are built, aligned and
interpreted: tokenizers, pre-training, scaling laws, SFT, RLHF/DPO, Constitutional AI, induction heads and sparse
autoencoders. Everything runs on a laptop CPU in minutes.*

Part of **ShaiLearnsAI**, a software engineer learning generative-AI research in public. Each notebook takes one
landmark idea (usually one or two papers), rebuilds the core of it at toy scale, and is explicit about what toy scale
can and cannot tell you.

## The notebooks

| # | Notebook | What you build | Key papers | CPU runtime* | Needs |
|---|---|---|---|---|---|
| 01 | [BPE tokenizer](notebooks/01_bpe_tokenizer.ipynb) | Byte-level BPE from scratch (training, regex pre-tokenization, special tokens, vocab-size sweep), compared with GPT-2 | Sennrich et al. 2016; Radford et al. 2019 | ~10 s | — |
| 02 | [Tiny GPT pre-training](notebooks/02_tiny_gpt_pretrain.ipynb) | Causal self-attention derived by hand, a nanoGPT-style ~0.8M-param char model trained on Shakespeare, sampling | Vaswani et al. 2017; GPT-2 | ~1 min | — |
| 03 | [Scaling laws](notebooks/03_scaling_laws.ipynb) | A 7-model size × data sweep; Kaplan-style L(N) fit, Chinchilla L(N, D) fit, compute-optimal allocation, and an honest look at why toy fits mislead | Kaplan et al. 2020; Hoffmann et al. 2022 | ~1.5 min | — |
| 04 | [SFT with TRL](notebooks/04_sft_trl.ipynb) | LoRA supervised fine-tuning of SmolLM2-135M on 40 hand-written examples, completion-only loss, before/after evaluation | Ouyang et al. 2022; Zhou et al. 2023 (LIMA); Hu et al. 2021 (LoRA) | ~2 min | — |
| 05 | [Reward model + DPO](notebooks/05_reward_model_dpo.ipynb) | Bradley–Terry reward model (`RewardTrainer`), DPO derived and checked by hand, `DPOTrainer`, RM-scored generations | Ouyang et al. 2022; Rafailov et al. 2023 | ~2 min | — |
| 06 | [Constitutional AI](notebooks/06_constitutional_ai.ipynb) | A written constitution, a critique → revision loop, multiple-choice AI feedback with position-bias correction, export of RLAIF preference pairs | Bai et al. 2022 (Anthropic) | ~1 min | API key optional |
| 07 | [Induction heads](notebooks/07_induction_heads.ipynb) | Find the previous-token → induction-head circuit in a 2-layer model with TransformerLens: attention scores, ablation, activation patching, K-composition | Elhage et al. 2021; Olsson et al. 2022 (Anthropic) | ~10 s | — |
| 08 | [Sparse autoencoders](notebooks/08_sparse_autoencoder.ipynb) | ReLU+L1 and TopK SAEs on a 1-layer model's MLP; L0 / FVE / dead latents / CE-loss-recovered; max-activating examples | Bricken et al. 2023, Templeton et al. 2024 (Anthropic); Gao et al. 2024 (OpenAI); Rajamanoharan et al. 2024 (DeepMind) | ~2.5 min | — |

\*Measured on an 8-thread CPU with no GPU, after the models are cached (see
[`outputs/execution_report.md`](outputs/execution_report.md)). The first run also downloads ~1 GB of small models (SmolLM2-135M base + instruct, two TransformerLens toy models, the GPT-2 tokenizer).
**Whole track: about 9 minutes.**

Every notebook follows the same layout: a hook, **source & license credit**, learning goals, background with paper
links, guided code, **"Your turn"** challenges, a **"Beyond the paper"** mini research question, and references.

## Setup

Requires Python ≥ 3.10 (tested on 3.13).

```bash
git clone https://github.com/shai200/ShaiLearnsAI.git
cd ShaiLearnsAI/research
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel, avoids multi-GB CUDA downloads
pip install -r requirements.txt
jupyter lab notebooks/
```

**Google Colab:** open a notebook, then run this in a first cell and restart the runtime:

```python
!git clone https://github.com/shai200/ShaiLearnsAI.git
%cd ShaiLearnsAI/research/notebooks
!pip install -q -r ../requirements.txt
```

Each notebook's first code cell finds `research/src/` automatically, so the notebooks work from `research/`,
`research/notebooks/` or a Colab clone.

## Configuration

All notebooks read their settings from [`src/lab_config.py`](src/lab_config.py):

| Variable | Default | Effect |
|---|---|---|
| `LAB_FULL_SIZE` | `0` | `1` switches every notebook from tiny CPU settings to larger, closer-to-the-paper settings (bigger models/grids, more steps) and enables cells marked **heavy**. You can also set `FULL_SIZE = True` in a notebook's setup cell. |
| `LAB_SEED` | `1337` | Global seed (Python, NumPy, PyTorch). |
| `LAB_DEVICE` | auto | `cuda` if available, else `cpu`. |
| `LAB_NUM_THREADS` | `min(8, cpus)` | PyTorch CPU threads. |

**Heavy / GPU cells** are clearly marked and call `heavy_cell(...)`, which prints `[skipped] …` unless
`LAB_FULL_SIZE=1` (and, where noted, a GPU is present). By default nothing heavy runs.

| Notebook | Skipped by default | Needs GPU? |
|---|---|---|
| 02 | Full nanoGPT `shakespeare_char` run with `torch.compile` | yes |
| 03 | Wider scaling grid (up to ~10M params) | recommended |
| 04 | SFT of SmolLM2-360M on a smoltalk slice | yes |
| 06 | SL-CAI fine-tuning on the revisions | no (slow) |
| 07 | Induction-score heatmap for GPT-2 small | no (~500 MB) |
| 08 | TopK sparsity sweep | no (slow) |

## LLM provider (notebook 06, and reusable elsewhere)

[`src/llm_provider.py`](src/llm_provider.py) exposes one interface, `complete(prompt, system=None, max_tokens=...)`,
with three backends:

1. **OpenAI**, if `OPENAI_API_KEY` is set (model: `LAB_OPENAI_MODEL`, default `gpt-4o-mini`);
2. **Anthropic**, if `ANTHROPIC_API_KEY` is set (model: `LAB_ANTHROPIC_MODEL`, default `claude-opus-5-5`);
3. **Local Hugging Face** fallback (the default): `LAB_HF_MODEL`, default
   `HuggingFaceTB/SmolLM2-135M-Instruct` (~540 MB, CPU-friendly).

Force a backend with `LAB_LLM_PROVIDER=openai|anthropic|hf`. If a key or SDK is missing, or a remote call fails, the
provider warns and falls back to the local model, so **no notebook ever needs an API key**. The remote SDKs are
optional (`pip install openai` / `pip install anthropic`). Never commit keys; use environment variables.

## Running everything (CI-style)

```bash
python scripts/run_all_notebooks.py            # all 8, CPU defaults; exits non-zero on any failure
python scripts/run_all_notebooks.py 03 07      # a subset (name filters)
python scripts/run_all_notebooks.py --report   # also rewrite outputs/execution_report.md
```

Executed copies (with outputs) go to `outputs/executed/`. That folder, model checkpoints, LoRA adapters and caches
are git-ignored. The committed notebooks are kept output-free.

## Layout

```
research/
├── notebooks/            # the eight labs (01-08)
├── src/
│   ├── lab_config.py     # FULL_SIZE flag, device, seeds, paths, heavy_cell()
│   ├── llm_provider.py   # pluggable OpenAI / Anthropic / local-HF LLM
│   └── tiny_gpt.py       # nanoGPT-style model + training loop (notebooks 02-03)
├── data/                 # tiny vendored datasets (public-domain or original)
├── scripts/run_all_notebooks.py
├── outputs/              # generated artifacts (git-ignored) + execution_report.md
├── requirements.txt
└── THIRD_PARTY_NOTICES.md
```

## Honesty notes

* These are **toy-scale reproductions of methods**, not of results. Where a number is computed (fitted exponents,
  accuracies, recovered loss) the notebooks print it and discuss how far it can be trusted. Nothing is claimed
  that the notebook doesn't compute.
* The small models (135M parameters and below) are weak. That is a feature for learning: you see the algorithm, and
  you see its failure modes.
* Licenses and sources for everything reused are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and at the
  top of each notebook.
