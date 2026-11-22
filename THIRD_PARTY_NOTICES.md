# Third-party notices

The code in this repository is original and released under the MIT license (see `LICENSE`), copyright Shai Lev, 2026. This file records the libraries the notebooks import and the courses and papers they follow. Where a source's license is restrictive or unclear, the notebook follows the syllabus and does not copy the source.

"Not copied" means the notebook was written for this repository. It is not a cleaned-up export of an upstream notebook, and it does not contain a verbatim implementation file from that project.

## Libraries imported by the labs

| Project | License | Used for |
| --- | --- | --- |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-style | Training loops, the tiny GPT, quantization |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD-3-Clause | Classical models, metrics, splits |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause | Arrays |
| [SciPy](https://github.com/scipy/scipy) | BSD-3-Clause | Dependency of scikit-learn and UMAP |
| [pandas](https://github.com/pandas-dev/pandas) | BSD-3-Clause | Tables |
| [Matplotlib](https://github.com/matplotlib/matplotlib) | PSF-based | Plots |
| [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | A randomly initialized GPT-2 for the PEFT lab, and the optional Hugging Face provider |
| [PEFT](https://github.com/huggingface/peft) | Apache-2.0 | LoRA on that GPT-2 |
| [OpenCV](https://github.com/opencv/opencv) | Apache-2.0 | Contours, moments, drawing |
| [umap-learn](https://github.com/lmcinnes/umap) | BSD-3-Clause | The UMAP embedding in the unsupervised lab |
| [Chroma](https://github.com/chroma-core/chroma) | Apache-2.0 | In-process vector store in the production RAG lab |
| [Qdrant client](https://github.com/qdrant/qdrant-client) | Apache-2.0 | In-memory vector store in the production RAG lab |
| [Jupyter](https://github.com/jupyter/notebook) / nbclient | BSD-3-Clause | Notebooks and CI execution |
| [pytest](https://github.com/pytest-dev/pytest) | MIT | Tests of the shared client |

Install them from `requirements.txt`. Do not vendor their source into this repo.

## Courses and code the labs follow

| Source | License of that source | What this repo did |
| --- | --- | --- |
| Andrej Karpathy, [micrograd](https://github.com/karpathy/micrograd) | MIT | Original `Value` implementation with the same API idea. The upstream file is not copied. |
| Andrej Karpathy, [minbpe](https://github.com/karpathy/minbpe) | MIT | Original byte-level BPE trainer. The upstream files are not copied. |
| Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) and [llm.c](https://github.com/karpathy/llm.c) | MIT | Original small decoder with the same block layout. Neither repository is copied. |
| Andrej Karpathy, Neural Networks: Zero to Hero | MIT for the accompanying code | Syllabus for the from-scratch network lab. Videos and notebooks are not copied. |
| [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html) | BSD-3-Clause | Supervised and unsupervised labs follow the tutorial topics and call the library. |
| [PyTorch tutorials](https://pytorch.org/tutorials/) | BSD-style | Training-loop and quantization APIs. |
| Hugging Face [Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft) course material and docs | Apache-2.0 | API use. The Hugging Face course notebooks are not copied. |
| Hugging Face [LLM course](https://huggingface.co/learn/llm-course) | Course code in the public course repo is Apache-2.0 | Prompting lab follows the topic list. Notebooks are not copied. |
| fast.ai Practical Deep Learning | Course code Apache-2.0 | Cited as the "implement it, then use the framework" order. Lesson notebooks are not copied. |
| [OpenAI Evals](https://github.com/openai/evals) | MIT | The agent-evaluation lab follows the idea of a scored task list. No eval file is copied. |
| [GPTCache](https://github.com/zilliztech/GPTCache) | Apache-2.0 | The semantic-cache lab follows the embed-and-threshold design. Source is not copied. |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | MIT | Named as the cache API for the interpretability lab. An optional cell imports it. Tutorials are not copied. |
| [LangChain](https://github.com/langchain-ai/langchain) | MIT | RAG labs follow the retriever / tool-loop shape. Cookbooks are not copied. |
| [LlamaIndex](https://github.com/run-llama/llama_index) | MIT | Hybrid search is a design reference. Cookbooks are not copied. |
| Stanford CS224N, CS231n, CS336 | Course assignments are not licensed for redistribution | Syllabus only. No assignment code. |
| ARENA mechanistic-interpretability curriculum | Not treated as reusable here | Linked from the interpretability notebook. Not copied. |
| Full Stack LLM bootcamp | Notebook collection not copied; license is not relied on | RAG and prompting labs follow the topic list. |
| UMAP paper and library, McInnes, Healy, Melville | Library BSD-3-Clause | Called as a library. |

## Papers cited as the source of an algorithm

These are citations. The papers are not redistributed, and their figures are not reproduced.

- Vaswani et al., *Attention Is All You Need*, 2017.
- Sennrich, Haddow, and Birch, *Neural Machine Translation of Rare Words with Subword Units*, 2016.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Rafailov et al., *Direct Preference Optimization*, 2023.
- Christiano et al., *Deep Reinforcement Learning from Human Preferences*, 2017.
- Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*, 2022.
- Kaplan et al., *Scaling Laws for Neural Language Models*, 2020.
- Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022.
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020.
- Robertson and Zaragoza, the BM25 framework.
- Cormack, Clarke, and Buettcher, reciprocal rank fusion, 2009.
- Li, Chu, Langford, and Schapire, LinUCB, 2010.
- Dudík, Langford, and Li, doubly robust policy evaluation, 2011.
- Alain and Bengio, linear classifier probes, 2016.
- Elhage et al., *A Mathematical Framework for Transformer Circuits*, 2021.
- Hu, *Visual Pattern Recognition by Moment Invariants*, 1962.
- Moody and Saffell, *Learning to Trade via Direct Reinforcement*, 2001.

The reading lists inside each notebook and in `ROADMAP.md` name further papers. Those lists are citations of the same kind.
