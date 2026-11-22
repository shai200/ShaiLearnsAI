# ShaiLearnsAI

Notebooks for learning generative AI by implementing the methods and measuring them. The repository is aimed at an engineer who already ships software and wants the training stack, the retrieval stack, and the research literature to be concrete.

The first track is a set of portfolio labs: production retrieval, agent evaluation, reward design, semantic caching, inference measurement, a backtest, and vision. The later tracks follow public courses and papers. Each notebook is original code, names the source it follows, and ends with a **Beyond the course** section that the source does not contain.

No API key is required to run the suite. No GPU is required. Cells that would use a GPU are marked and skipped unless you set `RUN_GPU=1`.

## Run the labs

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m ipykernel install --user --name python3
pytest -q
python scripts/execute_notebooks.py
```

Each lab directory also has a `requirements.txt` for the packages that notebook imports. The root file is the union, and it is what CI installs.

Open a notebook from a clone of this repository. The setup cell finds the repo root and imports `labs._shared`. On Colab, that cell clones the repository if it is not already on the machine. The badge on each row opens the notebook on `main`.

### Language-model calls

`labs/_shared/llm.py` is the only place a notebook asks for text.

| `LAB_LLM` | What runs |
| --- | --- |
| unset or `local` | A deterministic fallback. The notebooks default to this so a run is reproducible and does not spend a key. It is not a language model. The notebooks say so where the fallback writes the answer. |
| `openai` | The OpenAI API. Requires `OPENAI_API_KEY`. Optional package: `openai`. |
| `anthropic` | The Anthropic API. Requires `ANTHROPIC_API_KEY`. Optional package: `anthropic`. |
| `huggingface` | A tiny GPT-2 built from a config, with no download. Set `HF_MODEL` to a public model id to load a real checkpoint instead. |

`get_llm()` with no argument uses an OpenAI key if one is set, then an Anthropic key, then the local fallback. The graded notebooks pass `local` explicitly so a key sitting in the environment does not change the assertions.

## Portfolio labs

These are the labs to read first. They are the production-shaped versions of the skills the later tracks take apart.

| Lab | Track | Level | Source | Colab |
| --- | --- | --- | --- | --- |
| [Production RAG](labs/portfolio/01_production_rag/production_rag.ipynb) | Portfolio | Practitioner | Original. Hybrid retrieval following the Full Stack LLM bootcamp topic list (not copied) and the LlamaIndex / LangChain retriever shape (MIT, not copied). BM25 and reciprocal rank fusion. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/portfolio/01_production_rag/production_rag.ipynb) |
| [Agent evaluation](labs/portfolio/02_agent_evaluation/agent_evaluation.ipynb) | Portfolio | Practitioner | Original. Task harness in the spirit of OpenAI Evals (MIT, not copied). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/portfolio/02_agent_evaluation/agent_evaluation.ipynb) |
| [Reward design and next-best offer](labs/portfolio/03_reward_design_nbo/reward_design_nbo.ipynb) | Portfolio | Practitioner | Original. LinUCB (Li et al., 2010) and doubly robust evaluation (Dudík et al., 2011). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/portfolio/03_reward_design_nbo/reward_design_nbo.ipynb) |
| [Semantic cache](labs/portfolio/04_semantic_cache/semantic_cache.ipynb) | Portfolio | Practitioner | Original. Threshold cache following GPTCache (Apache-2.0, not copied). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/portfolio/04_semantic_cache/semantic_cache.ipynb) |
| [Inference optimization](labs/portfolio/05_inference_optimization/inference_optimization.ipynb) | Portfolio | Practitioner | Original. PyTorch dynamic quantization and a CPU batching measurement. GPU cell is skipped in CI. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/portfolio/05_inference_optimization/inference_optimization.ipynb) |
| [Trading backtest](labs/portfolio/06_trading_backtest/trading_backtest.ipynb) | Portfolio | Practitioner | Original educational backtest on synthetic prices. Moody and Saffell, 2001, for the reward. Not investment advice. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/portfolio/06_trading_backtest/trading_backtest.ipynb) |
| [Computer vision](labs/portfolio/07_computer_vision/computer_vision.ipynb) | Portfolio | Practitioner | Original. OpenCV (Apache-2.0) and a NumPy port of Hu moments. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/portfolio/07_computer_vision/computer_vision.ipynb) |

## Learning path

Read these after the portfolio labs, or start here if you want the mechanisms before the systems.

| Lab | Track | Level | Source | Colab |
| --- | --- | --- | --- | --- |
| [Supervised learning](labs/foundations/01_supervised_learning/supervised_learning.ipynb) | Foundations | Foundations | scikit-learn supervised tutorials (BSD-3). Original notebook. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/foundations/01_supervised_learning/supervised_learning.ipynb) |
| [Unsupervised learning](labs/foundations/02_unsupervised_learning/unsupervised_learning.ipynb) | Foundations | Foundations | scikit-learn clustering, PCA, and outlier tutorials (BSD-3). UMAP library (BSD-3). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/foundations/02_unsupervised_learning/unsupervised_learning.ipynb) |
| [Neural nets from scratch](labs/training/01_neural_nets_from_scratch/neural_nets_from_scratch.ipynb) | Training | Foundations | Karpathy micrograd and Zero to Hero (MIT). Original implementation. PyTorch training loop. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/training/01_neural_nets_from_scratch/neural_nets_from_scratch.ipynb) |
| [Byte-pair encoding](labs/training_llms/01_tokenization_bpe/tokenization_bpe.ipynb) | Training LLMs | Intermediate | Sennrich et al., 2016. Karpathy minbpe (MIT) as the design. Original code. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/training_llms/01_tokenization_bpe/tokenization_bpe.ipynb) |
| [Embeddings and attention](labs/training_llms/02_embeddings_and_attention/embeddings_and_attention.ipynb) | Training LLMs | Intermediate | CS224N distributional-semantics syllabus (assignment code not copied). Vaswani et al., 2017. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/training_llms/02_embeddings_and_attention/embeddings_and_attention.ipynb) |
| [Tiny GPT](labs/training_llms/03_tiny_gpt/tiny_gpt.ipynb) | Training LLMs | Intermediate | Karpathy nanoGPT (MIT). Original small implementation of that layout. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/training_llms/03_tiny_gpt/tiny_gpt.ipynb) |
| [Scaling](labs/training_llms/04_scaling_laws/scaling_laws.ipynb) | Training LLMs | Intermediate | Kaplan et al. (OpenAI, 2020) and Hoffmann et al., Chinchilla (DeepMind, 2022). A small fixed-step experiment, not a reproduction of those fits. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/training_llms/04_scaling_laws/scaling_laws.ipynb) |
| [Supervised fine-tuning](labs/training_llms/05_supervised_finetuning/supervised_finetuning.ipynb) | Training LLMs | Intermediate | Ouyang et al., InstructGPT (OpenAI, 2022). Original toy pairs and a response-token mask. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/training_llms/05_supervised_finetuning/supervised_finetuning.ipynb) |
| [LoRA and PEFT](labs/finetuning/01_lora/lora_peft.ipynb) | Fine-tuning and alignment | Advanced | Hu et al., 2021. Hugging Face PEFT (Apache-2.0) on a GPT-2 built from a config, no download. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/finetuning/01_lora/lora_peft.ipynb) |
| [Reward models and DPO](labs/finetuning/02_dpo_reward_model/dpo_reward_model.ipynb) | Fine-tuning and alignment | Advanced | Christiano et al., 2017. Rafailov et al., 2023. Synthetic preferences. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/finetuning/02_dpo_reward_model/dpo_reward_model.ipynb) |
| [Probing and attention](labs/interpretability/01_probing_and_attention/probing_and_attention.ipynb) | Frontier research | Advanced | Alain and Bengio, 2016. Elhage et al., transformer circuits (Anthropic, 2021). Optional TransformerLens cache (MIT). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/interpretability/01_probing_and_attention/probing_and_attention.ipynb) |
| [RAG fundamentals](labs/rag/01_rag_fundamentals/rag_fundamentals.ipynb) | RAG | Intermediate | Lewis et al., 2020, and the scikit-learn TF-IDF weighting. The introductory version of the portfolio RAG lab. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/rag/01_rag_fundamentals/rag_fundamentals.ipynb) |
| [Advanced RAG](labs/rag/02_advanced_rag/advanced_rag.ipynb) | RAG | Intermediate | Cormack et al., reciprocal rank fusion. Hit rate, MRR, and a lexical faithfulness check. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/rag/02_advanced_rag/advanced_rag.ipynb) |
| [Prompting and tools](labs/genai_apps/01_prompting_and_tools/prompting_and_tools.ipynb) | GenAI applications | Intermediate | Prompt patterns from the Hugging Face LLM course (notebooks not copied). ReAct tool loop (Yao et al., 2022). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/genai_apps/01_prompting_and_tools/prompting_and_tools.ipynb) |

## Layout

```text
labs/portfolio/          production-shaped labs
labs/foundations/        classification, regression, clustering, PCA, UMAP
labs/training/           autograd and a PyTorch training loop
labs/training_llms/      BPE, attention, a tiny GPT, scaling, SFT
labs/finetuning/         LoRA and DPO
labs/interpretability/   probes and attention on a one-layer model
labs/rag/                retrieval, then hybrid search and metrics
labs/genai_apps/         prompts, JSON, tools
labs/_shared/            seed helper and the pluggable model client
docs/labs-guide.md       how to add a lab from a course or a new set of notes
docs/notebook-template.md
ROADMAP.md               tracks that are not built yet, with the papers they should use
```

CI executes every notebook under `labs/` on CPU. A notebook opts out by setting metadata `shai_learns_ai.ci` to `skip`. That is reserved for a future lab that cannot run without a key or a GPU. The inference notebook does not opt out. Its CUDA cell returns immediately on CPU.

## What is still open

- **API keys.** The suite is graded against the local provider. Set `LAB_LLM` and the matching key when you want the same prompts sent to OpenAI or Anthropic. Those calls are not part of CI, and the hosted answers will not match the local assertions.
- **A GPU.** Set `RUN_GPU=1` on a CUDA machine to run the timing cell in the inference lab. Later tracks in `ROADMAP.md` (real fine-tunes, diffusion on images, distributed training) need a GPU before they are worth adding.
- **A public Hugging Face checkpoint.** Set `HF_MODEL` to load one through the Hugging Face provider. The default builds an untrained tiny GPT-2 so CI does not download weights. That sample is not graded.
- **More source material.** `docs/labs-guide.md` is the procedure for turning a course or a later batch of notes into a notebook without copying a restrictive license and without putting private text in a public file.

## License

Code in this repository is [MIT](LICENSE). Third-party libraries, and the courses that were followed but not copied, are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The trading notebook uses a synthetic price path. It is a backtesting exercise. It is not a trading system and not investment advice.
