# Third-party notices: Training LLMs & Frontier Research track

This track reuses, adapts, or depends on the projects, models and texts below. Code that was **adapted** is
marked as such in the file or notebook where it appears. Everything not listed here (notebook prose, the tiny
datasets in `data/` other than the Shakespeare excerpt, the constitution, and the helper modules except where
noted) was written for this repository.

Licenses were checked against each project's repository or Hugging Face model card when this track was written
(November 2026). Model weights are **never vendored**. They are downloaded at runtime into your Hugging Face cache
(`~/.cache/huggingface` by default), subject to their own licenses.

## Code adapted or closely followed

| Project | License | Used in | What was reused |
|---|---|---|---|
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | MIT | `src/tiny_gpt.py`, notebooks 02–03 | GPT model structure (causal self-attention, MLP, pre-LN block, weight tying, GPT-2 init, AdamW param groups), training-loop structure and the `shakespeare_char` hyperparameters. Condensed and modified (constant-LR option, `eval_at`, non-embedding param count). |
| [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) | MIT | notebook 02 | Structure of the "average the past → masked softmax → attention" derivation (code rewritten). |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | MIT | notebook 01 | BPE algorithm structure (`get_stats` / `merge` / `train` / `encode` / `decode`), special-token handling design, GPT-2/GPT-4 split regex constants. Rewritten with a chunk-frequency training speed-up. |
| [openai/tiktoken](https://github.com/openai/tiktoken) | MIT | notebook 01 | GPT-4 (`cl100k_base`) pre-tokenization regex; `allowed_special` safety design. |
| [openai/gpt-2](https://github.com/openai/gpt-2) | Modified MIT | notebook 01 | GPT-2 pre-tokenization regex. |

Copyright notices for the MIT-licensed code above: *Copyright (c) 2022 Andrej Karpathy* (nanoGPT),
*Copyright (c) 2024 Andrej Karpathy* (minbpe), *Copyright (c) 2022 OpenAI, Shantanu Jain* (tiktoken),
*Copyright (c) 2019 OpenAI* (gpt-2). The MIT License requires the copyright notice and permission notice to be
included with substantial portions of the software:

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
| [regex](https://github.com/mrabarnett/mrab-regex) | Apache-2.0 | 01 |
| [Jupyter](https://jupyter.org) (nbclient, nbformat, ipykernel) | BSD-3-Clause | `scripts/run_all_notebooks.py` |
| [openai-python](https://github.com/openai/openai-python) (optional) | Apache-2.0 | `src/llm_provider.py`, only if `OPENAI_API_KEY` is set |
| [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) (optional) | MIT | `src/llm_provider.py`, only if `ANTHROPIC_API_KEY` is set |

## Models (downloaded at runtime, not redistributed)

| Model | License (per model card) | Used in |
|---|---|---|
| [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | Apache-2.0 | 04 (SFT base) |
| [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Apache-2.0 | 05, 06, default local LLM in `src/llm_provider.py` |
| [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) / [-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Apache-2.0 | 04, 05 (`FULL_SIZE` only) |
| [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) (tokenizer; full model in an optional cell) | MIT | 01 (tokenizer comparison), 07 (optional GPT-2 small cell) |
| [NeelNanda/Attn_Only_2L512W_C4_Code](https://huggingface.co/NeelNanda/Attn_Only_2L512W_C4_Code) | No license stated on the model card | 07. Loaded at runtime via TransformerLens for research/education; not redistributed. |
| [NeelNanda/GELU_1L512W_C4_Code](https://huggingface.co/NeelNanda/GELU_1L512W_C4_Code) | MIT | 08 |
| [sshleifer/tiny-gpt2](https://huggingface.co/sshleifer/tiny-gpt2) (optional) | No license stated on the model card | Only if you set `LAB_HF_MODEL=sshleifer/tiny-gpt2` for smoke tests. |

## Datasets

| Data | Source / license | Notes |
|---|---|---|
| `data/tiny_shakespeare.txt` (~300 KB) | Works of William Shakespeare: **public domain**. Excerpt (first ~300 KB) of the `tinyshakespeare` file distributed with [karpathy/char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare). | The char-rnn repository does not state a license. We rely on the public-domain status of the underlying text, and credit the compilation. |
| `data/sft_tiny.jsonl`, `data/sft_eval_prompts.json` | Original to this repository | 40 instruction/response pairs + 8 held-out prompts. |
| `data/preferences_tiny.jsonl` | Original to this repository | 36 synthetic preference pairs. |
| `data/constitution.json`, `data/red_team_prompts.json` | Original to this repository | Format inspired by Bai et al. (2022); not an official Anthropic document. |
| [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (optional) | See dataset card | Only downloaded by the optional `FULL_SIZE` + GPU cell in notebook 04. |

## Educational sources linked, not copied

* **ARENA** curriculum (Callum McDougall et al.): [github.com/callummcdougall/ARENA_3.0](https://github.com/callummcdougall/ARENA_3.0).
  The repository does not state a license, so **no ARENA material is copied**. Notebooks 07–08 follow a similar
  syllabus with original code and link to ARENA for depth.
* **SAELens** ([github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)) and **Gemma Scope**: linked as
  production-grade references in notebook 08; not used or copied.
* Papers are cited in each notebook with arXiv IDs or official blog/transformer-circuits URLs.
