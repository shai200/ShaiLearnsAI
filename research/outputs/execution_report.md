# Execution report: Training LLMs & Frontier Research track

* Generated: 2026-11-23 07:07 by `research/scripts/run_all_notebooks.py`
* Mode: `LAB_FULL_SIZE=0` (CPU tiny defaults)
* Python 3.13.5 on Linux x86_64, 8 CPU threads, no GPU required
* Packages: torch 2.15.0+cpu, transformers 5.24.0, trl 1.18.2, peft 0.22.0, datasets 5.2.0, transformer-lens 4.4.0
* LLM provider: Anthropic (ANTHROPIC_API_KEY set)
* Total wall-clock: 129.8 min

| Notebook | Result | Runtime | Notes |
|---|---|---|---|
| `01_bpe_tokenizer.ipynb` | ✅ ran clean | 16 s | — |
| `02_tiny_gpt_pretrain.ipynb` | ✅ ran clean | 82 s | 1 heavy/GPU cell(s) skipped by design |
| `03_scaling_laws.ipynb` | ✅ ran clean | 68 s | 1 heavy/GPU cell(s) skipped by design |
| `04_sft_trl.ipynb` | ✅ ran clean | 100 s | 1 heavy/GPU cell(s) skipped by design |
| `05_reward_model_dpo.ipynb` | ✅ ran clean | 105 s | — |
| `06_constitutional_ai.ipynb` | ✅ ran clean | 30 s | 1 heavy/GPU cell(s) skipped by design |
| `07_induction_heads.ipynb` | ✅ ran clean | 10 s | 1 heavy/GPU cell(s) skipped by design |
| `08_sparse_autoencoder.ipynb` | ✅ ran clean | 156 s | 1 heavy/GPU cell(s) skipped by design |

Skipped heavy cells (run with `LAB_FULL_SIZE=1`, some also need a GPU):

* `02_tiny_gpt_pretrain.ipynb`: Full nanoGPT shakespeare_char run with torch.compile -- set LAB_FULL_SIZE=1 to run this cell.
* `03_scaling_laws.ipynb`: Wider scaling grid (up to ~10M params, 4k steps each) -- set LAB_FULL_SIZE=1 to run this cell.
* `04_sft_trl.ipynb`: SFT of SmolLM2-360M on a 2k-example smoltalk slice -- set LAB_FULL_SIZE=1 to run this cell.
* `06_constitutional_ai.ipynb`: SL-CAI: supervised fine-tuning on critique-revised responses -- set LAB_FULL_SIZE=1 to run this cell.
* `07_induction_heads.ipynb`: Induction-score heatmap for GPT-2 small (124M) -- set LAB_FULL_SIZE=1 to run this cell.
* `08_sparse_autoencoder.ipynb`: TopK sweep k in {8, 16, 32, 64, 128} to trace the L0-vs-FVE frontier -- set LAB_FULL_SIZE=1 to run this cell.

<!-- Hand-written notes below this line are preserved by --report. Update them when results change. -->

## Notes per notebook (numbers from the run in the table above)

Numbers are copied from the executed notebooks (`outputs/executed/`, git-ignored). Everything is seeded. On this
machine, repeated runs of notebooks 04, 05 and 06 now give identical text outputs (checked with extra back-to-back
runs), the numbers quoted below for 01, 03 and 07 were identical in every full run, and 02's final loss agreed to
three decimal places. That was not true before this review: the SmolLM2 models were being loaded in bfloat16 and
trained under TRL's default bf16 autocast, which on CPU is slow and made the results of 04, 05 and 06 drift from run
to run. Other machines, thread counts and library versions can still give
different numbers. Notebook 08 reads this repo's own `src/*.py` as its code sample, so its numbers shift slightly
whenever those files are edited.

* **01 BPE:** 512 merges trained in ~5 s; 2.47 bytes/token on the training text. Tokens for four one-line samples
  (ours vs. GPT-2): Shakespeare 31 vs. 20, modern English 36 vs. 13, Python 38 vs. 22, Hebrew 64 vs. 44 (ours falls
  back to one token per byte).
* **02 Tiny GPT:** 812K-parameter character model, 1,200 steps; final validation loss 1.54 nats (2.22 bits/char);
  samples have the right play format and little meaning.
* **03 Scaling laws:** 7 models (3K–788K non-embedding parameters). Pure power-law fit L ∝ N^-0.055. Chinchilla-form
  fit: N_opt ∝ C^0.45. **This is not a Chinchilla match:** the same fit pins E at the edge of its search box (0.050
  vs. Chinchilla's 1.69), asks for ~223× more training tokens than any run saw, and implies tens of thousands of tokens
  per parameter (Chinchilla: ~20). The envelope estimate from the same runs is C^0.89, and it is biased toward 1
  because every size shares one training horizon. The notebook prints a three-item checklist; all three items fail.
  (The previous version of this report quoted an envelope exponent of 0.96. That estimator dropped finished runs from
  the comparison, so its "best" loss could go *up* with compute; it was replaced.)
* **04 SFT (TRL + LoRA, SmolLM2-135M):** 4.9M trainable parameters (3.5%). On 8 held-out prompts: stops by itself
  12% → 88%; uses the trained `TL;DR:` format 0% → 88%. Factual errors remain, now in the tidy format: Paris "located
  in the south of France", "10% of 50 is 0.10 × 50 = 0.50", a list's length being "its length minus one".
* **05 Reward model + DPO:** trained RM pairwise accuracy: train 96%, held-out 88% on 8 pairs. **The untrained score
  head also scored 88% on those 8 pairs** (68% on the training pairs), and a no-model "longer answer is better"
  baseline scores 79% train / 75% held-out, so the held-out number is not evidence of learning. The one held-out pair
  the trained RM gets wrong is the "concision" pair, where the worse answer is longer. DPO held-out implicit-reward
  accuracy 88%. The RM preferred the DPO generation on 8 of 8 prompts, and **two of those wins are a wrong answer or
  a non-answer beating a correct one** ("How many days are in a week?", "How many legs does a spider have?"). Several
  generations from both policies are unsafe or incorrect.
* **06 Constitutional AI (local SmolLM2-135M-Instruct, no API keys):** the critique → revision → AI-feedback loop runs
  end to end. Debiased P(revised ≻ original) = 0.482, position bias P(first) = 0.632; 2 of 4 pairs passed the export
  threshold. **The 135M model is a weak critic and a weak judge:** critiques miss the point, revisions are often no
  better than the originals, and the position bias is larger than any preference for the revision. The result shows
  that the plumbing works, not that the method does.
* **07 Induction heads (TransformerLens 4, attention-only 2L):** loss on repeated random tokens 14.63 → 3.04 nats
  between first and second copy. Previous-token head L0H3 (score 0.51), induction head L1H6 (0.66). Patching L1H6 alone
  recovers 92% of clean performance (L0H3 alone: 2%). K-composition L1H6←L0H3 = 0.128 vs. a 0.044 baseline.
* **08 SAE (GELU-1L MLP, 4,096 latents, 600 steps, one seed):** ReLU+L1: L0 29, FVE 0.62, CE recovered 95.5%.
  TopK k=32: FVE 0.75, CE recovered 97.9%. Dead latents: 9.4% (ReLU) vs. 4.2% (TopK) on the 6k held-out tokens, but
  only 0.2% vs. 2.2% when recounted on the 57k training tokens, so this run does **not** show TopK having fewer dead
  latents. The best single neuron separates code from prose better than any latent (gap 0.88 vs. 0.35); the notebook
  explains why that is not a fair test of an SAE.

## What was and was not exercised

* **Default path (`LAB_FULL_SIZE=0`):** all eight notebooks, top to bottom, in the run tabulated above.
* **Gated heavy cells:** not part of the table. Each was smoke-tested once on this CPU-only machine by forcing
  `heavy_cell()` open while keeping the tiny default sizes, to check that the code in the cell runs. The 06 cell was
  re-tested after the float32 / bf16 change described above; the 04 cell's one-argument change (`dtype=`) was not.
  * 03 (prints the grid in use), 06 (SL-CAI: 50 LoRA steps on the four revisions) and 07 (GPT-2 small induction
    heatmap; top heads L5H5, L6H9, L7H10, L5H1, L7H2) ran as written.
  * 08 (TopK sweep) ran with the step count cut to 60 per SAE.
  * 02 ran with 30 steps. `torch.compile` could not be used here (no C++ compiler), which crashed the cell as
    originally written; it now falls back to the uncompiled model. The compiled path itself was only checked with
    `backend="eager"`.
  * 04 was checked in a reduced form (same calls; SmolLM2-135M instead of 360M, 16 rows of the smoltalk subset,
    2 steps, CPU).
* **Not run at all:** the `LAB_FULL_SIZE=1` experiment sizes of any notebook, anything on a GPU, and any successful
  call to the OpenAI or Anthropic APIs. The provider's fallback paths were checked with missing keys, invalid keys
  (HTTP 401 → local model) and stubbed clients.

## Environment adaptations

* **TransformerLens 4.x** removed `HookedTransformer`. Notebooks 07–08 use `TransformerBridge.boot_tl_legacy(...)`
  for Neel Nanda's toy models (and `TransformerBridge.boot_transformers("gpt2")` in the optional cell), with the same
  hook names (`blocks.L.attn.hook_z`, `hook_pattern`, `mlp.hook_post`).
* **transformers 5 / TRL 1.x precision defaults.** Models are loaded with `dtype=torch.float32` and the TRL configs
  set `bf16=False` when no GPU is present (see the note at the top of this section).
* No API keys were present, so `llm_provider` used the local Hugging Face fallback (`SmolLM2-135M-Instruct`).
* No GPU; all cells marked heavy were skipped by `heavy_cell()` in the tabulated run, as listed above.
* Runtimes vary with machine load: across the full runs made during this review notebook 08 took between 156 s and
  251 s, and the whole track between about 8.5 and 11 minutes.
