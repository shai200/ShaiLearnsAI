# Execution report: Training LLMs & Frontier Research track

* Generated: 2026-11-23 01:21 by `research/scripts/run_all_notebooks.py`
* Mode: `LAB_FULL_SIZE=0` (CPU tiny defaults)
* Python 3.13.5 on Linux x86_64, 8 CPU threads, no GPU required
* LLM provider: local Hugging Face fallback (no API keys set)
* Total wall-clock: 8.8 min

| Notebook | Result | Runtime | Notes |
|---|---|---|---|
| `01_bpe_tokenizer.ipynb` | ✅ ran clean | 10 s | — |
| `02_tiny_gpt_pretrain.ipynb` | ✅ ran clean | 49 s | 1 heavy/GPU cell(s) skipped by design |
| `03_scaling_laws.ipynb` | ✅ ran clean | 70 s | 1 heavy/GPU cell(s) skipped by design |
| `04_sft_trl.ipynb` | ✅ ran clean | 101 s | 1 heavy/GPU cell(s) skipped by design |
| `05_reward_model_dpo.ipynb` | ✅ ran clean | 86 s | — |
| `06_constitutional_ai.ipynb` | ✅ ran clean | 57 s | 1 heavy/GPU cell(s) skipped by design |
| `07_induction_heads.ipynb` | ✅ ran clean | 8 s | 1 heavy/GPU cell(s) skipped by design |
| `08_sparse_autoencoder.ipynb` | ✅ ran clean | 145 s | 1 heavy/GPU cell(s) skipped by design |

Skipped heavy cells (run with `LAB_FULL_SIZE=1`, some also need a GPU):

* `02_tiny_gpt_pretrain.ipynb`: Full nanoGPT shakespeare_char run with torch.compile -- set LAB_FULL_SIZE=1 to run this cell.
* `03_scaling_laws.ipynb`: Wider scaling grid (up to ~10M params, 4k steps each) -- set LAB_FULL_SIZE=1 to run this cell.
* `04_sft_trl.ipynb`: SFT of SmolLM2-360M on a 2k-example smoltalk slice -- set LAB_FULL_SIZE=1 to run this cell.
* `06_constitutional_ai.ipynb`: SL-CAI: supervised fine-tuning on critique-revised responses -- set LAB_FULL_SIZE=1 to run this cell.
* `07_induction_heads.ipynb`: Induction-score heatmap for GPT-2 small (124M) -- set LAB_FULL_SIZE=1 to run this cell.
* `08_sparse_autoencoder.ipynb`: TopK sweep k in {8, 16, 32, 64, 128} to trace the L0-vs-FVE frontier -- set LAB_FULL_SIZE=1 to run this cell.

## Notes per notebook (computed in this run)

Numbers below are copied from the executed notebooks (`outputs/executed/`, git-ignored). CPU runs are seeded but not
bit-exact across machines and thread counts, so expect small differences (e.g. reward-model scores in 05 varied by a few
tenths between two runs on this machine; qualitative conclusions were unchanged).

* **01 BPE:** 512 merges trained in ~5 s; 2.47 bytes/token on the training text; Hebrew falls back to 1 token per byte.
* **02 Tiny GPT:** 812K-param char model, 1,200 steps; final val loss 1.54 nats (2.22 bits/char); samples have the
  right play format.
* **03 Scaling laws:** 7 models (3K–788K non-embedding params). Pure power-law fit L ∝ N^-0.055; Chinchilla-form fit
  gives N_opt ∝ C^0.45, but the empirical envelope gives C^0.96 and the irreducible term collapses to ~0. The
  notebook discusses why the toy fit is under-determined.
* **04 SFT (TRL + LoRA, SmolLM2-135M):** held-out prompts that stop by themselves 25% → 100%; adoption of the trained
  `TL;DR:` format 0% → 88%. Factual errors remain (by design, the notebook discusses this).
* **05 RM + DPO:** RM held-out pairwise accuracy 88% on 8 pairs (an *untrained* score head also scored 88%, a caveat
  shown in the notebook); DPO held-out implicit-reward accuracy 88%; RM prefers DPO generations on 6/8 prompts (7/8 in a re-run of 05 alone), while
  one DPO answer that the RM rewarded is factually wrong (the "proxy reward" lesson).
* **06 Constitutional AI (local SmolLM2-135M-Instruct; no API keys):** full critique → revision → AI-feedback loop
  ran; debiased P(revised ≻ original) = 0.48, position bias P(first) = 0.58. As expected, the 135M model is too weak
  to be a reliable critic or judge; the algorithm and prompts are the deliverable here.
* **07 Induction heads (TransformerLens 4, attn-only 2L):** previous-token head L0H3 (score 0.51), induction head
  L1H6 (0.66). Patching L1H6 alone recovers 92% of clean performance; K-composition L1H6←L0H3 = 0.128 vs. 0.044
  baseline.
* **08 SAE (GELU-1L MLP, 4,096 latents):** ReLU+L1: L0 29, FVE 0.62, 10% dead, CE recovered 95%; TopK k=32: FVE
  0.75, 4% dead, CE recovered 97.5%. Max-activating examples show clearly specific code and prose latents.

## Environment adaptations

* **TransformerLens 4.x** removed `HookedTransformer`. Notebooks 07–08 use `TransformerBridge.boot_tl_legacy(...)`
  for Neel Nanda's toy models and the same hook names (`blocks.L.attn.hook_z`, `hook_pattern`, `mlp.hook_post`).
* No API keys were present, so `llm_provider` used the local Hugging Face fallback (`SmolLM2-135M-Instruct`).
* No GPU; all cells marked heavy were skipped by `heavy_cell()` as listed above.
