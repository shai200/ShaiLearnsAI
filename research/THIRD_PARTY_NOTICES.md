# Third-party notices: Training LLMs & Frontier Research track

This track reuses, adapts, or depends on the projects, models and texts below. Code that was **adapted** is
marked as such in the file or notebook where it appears. Everything not listed here (notebook prose, the tiny
datasets in `data/` other than the Shakespeare excerpt, the constitution, and the helper modules except where
noted) was written for this repository.

Licenses were re-checked on 2026-11-23 against each project's repository (LICENSE file, README, and GitHub's license
metadata) and each Hugging Face model or dataset card. Where a source states no license, this file says so plainly
instead of guessing, and nothing from that source is copied into this repository. This is a good-faith record, not
legal advice.

Model weights and tokenizer files are **never vendored**. They are downloaded at runtime into your Hugging Face cache
(`~/.cache/huggingface` by default), subject to their own terms.

## Code adapted or closely followed

| Project | License | Used in | What was reused |
|---|---|---|---|
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | MIT | `src/tiny_gpt.py`, notebooks 02–03 | GPT model structure (causal self-attention, MLP, pre-LN block, weight tying, GPT-2 init, AdamW param groups), training-loop structure and the `shakespeare_char` hyperparameters. Condensed and modified (constant-LR option, `eval_at`, non-embedding param count). |
| [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) | MIT per the README ("License: MIT"); the repository has no LICENSE file or copyright line | notebook 02, section 2 | **Idea only, no code.** The teaching order "average the past → masked softmax → data-dependent weights" is inspired by the lecture. The cell is written from the attention equation and checked against `F.scaled_dot_product_attention`. |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | MIT | notebook 01 | BPE algorithm structure (`get_stats` / `merge` / `train` / `encode` / `decode`), special-token handling design, GPT-2/GPT-4 split regex constants. Rewritten with a chunk-frequency training speed-up. |
| [openai/tiktoken](https://github.com/openai/tiktoken) | MIT | notebook 01 | GPT-4 (`cl100k_base`) pre-tokenization regex; `allowed_special` safety design. |
| [openai/gpt-2](https://github.com/openai/gpt-2) | "Modified MIT License" (MIT terms plus a note that the notice need not accompany model outputs) | notebook 01 | GPT-2 pre-tokenization regex. |

Copyright notices, as they appear in each project's LICENSE file: *Copyright (c) 2022 Andrej Karpathy* (nanoGPT),
*Copyright (c) 2024 Andrej* (minbpe), *Copyright (c) 2022 OpenAI, Shantanu Jain* (tiktoken),
*Software Copyright (c) 2019 OpenAI* (gpt-2). The MIT License requires the copyright notice and permission notice to
be included with substantial portions of the software:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
> documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
> rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright
> notice and this permission notice shall be included in all copies or substantial portions of the Software.
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
> WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
> COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
> OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Libraries (dependencies, not vendored)

| Project | License | Used in |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | all |
| [Hugging Face transformers](https://github.com/huggingface/transformers) | Apache-2.0 | 01, 04, 05, 06, `src/llm_provider.py` |
| [Hugging Face TRL](https://github.com/huggingface/trl) | Apache-2.0 | 04 (`SFTTrainer`), 05 (`RewardTrainer`, `DPOTrainer`), 06 (optional SL-CAI cell) |
| [Hugging Face PEFT](https://github.com/huggingface/peft) | Apache-2.0 | 04, 05, 06 (LoRA) |
| [Hugging Face datasets](https://github.com/huggingface/datasets) | Apache-2.0 | 04, 05, 06 |
| [Hugging Face accelerate](https://github.com/huggingface/accelerate) | Apache-2.0 | 04, 05 (via TRL / transformers) |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | MIT | 07, 08 (`TransformerBridge`, hooks, activation cache) |
| [NumPy](https://numpy.org) / [SciPy](https://scipy.org) / [pandas](https://pandas.pydata.org) | BSD-3-Clause | 03, 05, 06, 08 |
| [Matplotlib](https://matplotlib.org) | Matplotlib License (PSF-style) | all |
| [regex](https://github.com/mrabarnett/mrab-regex) | Apache-2.0 AND CNRI-Python (per the package metadata) | 01 |
| [Jupyter](https://jupyter.org) (nbclient, nbformat, ipykernel) | BSD-3-Clause | `scripts/run_all_notebooks.py` |
| [openai-python](https://github.com/openai/openai-python) (optional) | Apache-2.0 | `src/llm_provider.py`, only if `OPENAI_API_KEY` is set |
| [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) (optional) | MIT | `src/llm_provider.py`, only if `ANTHROPIC_API_KEY` is set |

## Models and tokenizers (downloaded at runtime, not redistributed)

| Model | License (per Hugging Face card / repo metadata) | Used in |
|---|---|---|
| [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | Apache-2.0 | 04 (SFT base) |
| [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Apache-2.0 | 05, 06, default local LLM in `src/llm_provider.py` |
| [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) / [-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Apache-2.0 | 04, 05 (`FULL_SIZE` only) |
| [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) (tokenizer; full model in an optional cell) | MIT | 01 (tokenizer comparison), 07 (optional GPT-2 small cell) |
| [NeelNanda/Attn_Only_2L512W_C4_Code](https://huggingface.co/NeelNanda/Attn_Only_2L512W_C4_Code) | **No license stated** (the repository has no model card and no license metadata) | 07. Fetched at runtime by TransformerLens (~210 MB). |
| [NeelNanda/GELU_1L512W_C4_Code](https://huggingface.co/NeelNanda/GELU_1L512W_C4_Code) | MIT (card metadata and `LICENSE.txt`) | 08. Fetched at runtime (~210 MB). |
| [NeelNanda/gpt-neox-tokenizer-digits](https://huggingface.co/NeelNanda/gpt-neox-tokenizer-digits) (tokenizer for both toy models) | **No license stated.** Its card describes it as a fork of the GPT-NeoX-20B tokenizer with multi-digit tokens removed | 07, 08. Fetched at runtime by TransformerLens (~2 MB). |
| [sshleifer/tiny-gpt2](https://huggingface.co/sshleifer/tiny-gpt2) (optional) | **No license stated** (no model card) | Never downloaded by the notebooks. Only if you set `LAB_HF_MODEL=sshleifer/tiny-gpt2` yourself for a smoke test. |

**About the three "no license stated" entries.** With no license, the default is that the author keeps all rights, so
we do not copy, modify or redistribute these files: the notebooks only ask the Hugging Face Hub to download them to
your machine at run time, which is how TransformerLens (the MIT-licensed library their author wrote, and which loads
these toy models by name) uses them. Whether that is enough for *your* use is your call. If you need explicit terms,
for example for redistribution or commercial work, ask the author, or switch to a model with a stated license:
notebook 07 has a GPT-2 small cell (MIT), and notebook 08's model is MIT (only its tokenizer is unlicensed).

## Datasets

| Data | Source / license | Notes |
|---|---|---|
| `data/tiny_shakespeare.txt` (299,868 bytes) | Text: plays of William Shakespeare (d. 1616), **public domain**. Compilation: the `tinyshakespeare/input.txt` file from [karpathy/char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), whose README states "License: MIT" (there is no LICENSE file or copyright line). | See "Provenance of the Shakespeare excerpt" below. |
| `data/sft_tiny.jsonl`, `data/sft_eval_prompts.json` | Original to this repository | 40 instruction/response pairs + 8 held-out prompts. |
| `data/preferences_tiny.jsonl` | Original to this repository | 36 synthetic preference pairs. |
| `data/constitution.json`, `data/red_team_prompts.json` | Original to this repository | Format inspired by Bai et al. (2022); not an official Anthropic document. |
| [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), `everyday-conversations` subset (optional) | The smoltalk card licenses its newly created subsets under Apache-2.0 and refers to the original datasets for the rest; this subset's source, [everyday-conversations-llama3.1-2k](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k), is tagged Apache-2.0 | Only downloaded by the optional `FULL_SIZE` + GPU cell in notebook 04. Not vendored. |

### Provenance of the Shakespeare excerpt

* **What the file is.** `data/tiny_shakespeare.txt` is byte-for-byte the first 299,868 bytes of char-rnn's
  `data/tinyshakespeare/input.txt` (1,115,394 bytes); this was verified against the upstream file on 2026-11-23. It
  contains play text only: no Project Gutenberg header, license text or other boilerplate.
* **The text.** Shakespeare's plays are in the public domain everywhere.
* **The edition.** The compilation does not say which edition it was taken from, and we could not establish that
  independently. The modernised spelling and the `Speaker:` layout are typical of the nineteenth-century editions
  that free online Shakespeare collections use, which are themselves out of copyright, but treat that as our
  reading rather than a documented fact.
* **The compilation.** Selecting and concatenating the plays was done by Andrej Karpathy for char-rnn, whose README
  says "License: MIT". That grant has no accompanying LICENSE file or copyright notice to reproduce, so we credit
  the source here; notebooks 01 and 02 repeat the credit, and notebooks 03 and 08 point to this file.

## Educational sources linked, not copied

* **ARENA** curriculum (Callum McDougall et al.):
  [github.com/ARENA-education/ARENA_materials](https://github.com/ARENA-education/ARENA_materials) (formerly
  `callummcdougall/ARENA_3.0`). The repository has no LICENSE file and its README states no license, so **no ARENA
  material (code, exercises, text or figures) is copied**. Notebooks 07–08 cover overlapping topics (induction heads,
  SAEs) because those come from the cited papers; the code and prose are original, and ARENA is linked for depth.
* **SAELens** ([github.com/decoderesearch/SAELens](https://github.com/decoderesearch/SAELens), MIT; formerly
  `jbloomAus/SAELens`) and **Gemma Scope** ([huggingface.co/google/gemma-scope](https://huggingface.co/google/gemma-scope),
  CC-BY-4.0): linked as production-grade references in notebook 08; not used or copied.
* **TRL documentation** (Apache-2.0): notebooks 04–06 follow its documented usage of `SFTTrainer`, `RewardTrainer` and
  `DPOTrainer`; no dataset or prompt text is taken from it.
* **Anthropic's published constitutions** (2023 and 2026) are linked from notebook 06 for context. The demo
  constitution in `data/constitution.json` was written for this repository and copies no text from them.
* Papers are cited in each notebook with arXiv IDs or official blog/transformer-circuits URLs.
