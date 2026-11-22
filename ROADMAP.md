# Roadmap

The notebooks in the repository are the first batch: portfolio labs, then classical learning, a tiny language-model stack, LoRA, a reward-model / DPO lab, one interpretability lab, RAG, and a tool loop. This file is the list of tracks to build next. Each item names the papers the lab should reimplement at small scale or extend. "Extend" means an ablation or a measurement the paper does not report, in the same style as the **Beyond the course** sections that are already in the repo.

Nothing here is a promise about a training run we have not done. A later lab that needs a GPU or a paid API key should say so in the notebook title and set `shai_learns_ai.ci` to `skip` until a CPU path exists.

## LLM evaluation and how to read a paper

Reproduce the metric before reproducing the model.

- Liang et al., *Holistic Evaluation of Language Models* (HELM, Stanford, 2022). A small slice: one task, several metrics, the same prompts.
- Hendrycks et al., *Measuring Massive Multitask Language Understanding* (2020). Format and scoring only, on a handful of public items, not a full dump of the benchmark into the repo.
- Srivastava et al., *Beyond the Imitation Game* (BIG-bench, 2022), as a source of task formats.
- OpenAI Evals (MIT) as the harness shape. Already used as the design reference for the portfolio evaluation lab.
- Anthropic, *Discovering Language Model Behaviors with Model-Written Evaluations* (Perez et al., 2022). The lab should discuss the circularity of model-written tests, not treat them as ground truth.
- Google DeepMind, the evaluation appendix of the Gemini technical report (2023), for what a lab report has to disclose: prompt, temperature, and whether the metric was chosen after seeing the score.

Extension to aim at: two scorers on one set of answers, and the disagreements.

## RLHF

The DPO notebook is the small version of the loss. The next lab is the loop around it: collect preferences, fit a reward model, then optimize a policy with a KL penalty.

- Christiano et al., *Deep Reinforcement Learning from Human Preferences* (2017).
- Ziegler et al., *Fine-Tuning Language Models from Human Preferences* (OpenAI, 2019).
- Stiennon et al., *Learning to Summarize from Human Feedback* (OpenAI, 2020).
- Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT, OpenAI, 2022).
- Bai et al., *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback* (Anthropic, 2022).
- Schulman et al., *Proximal Policy Optimization Algorithms* (OpenAI, 2017), for the policy step.
- Gao, Schulman, and Hilton, *Scaling Laws for Reward Model Overoptimization* (OpenAI, 2022). The extension: a reward model fit on a subset, and a policy that looks better to that model while a held-out reward model disagrees.

## Constitutional AI and reward modeling

Separate written rules from learned preferences. The portfolio guardrail is only the written rule.

- Bai et al., *Constitutional AI: Harmlessness from AI Feedback* (Anthropic, 2022).
- Lee et al., *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback* (Google, 2023).
- Kundu et al. / Anthropic, *Specific versus General Principles for Constitutional AI* (2023), if the lab compares two constitutions on the same prompts.
- The extension: a constitution that changes one clause, and the preference pairs that flip.

## Mechanistic interpretability

The probing notebook is the entry point. Next is a circuit on a model small enough to test, then a sparse autoencoder on its residual stream.

- Elhage et al., *A Mathematical Framework for Transformer Circuits* (Anthropic, 2021).
- Olsson et al., *In-context Learning and Induction Heads* (Anthropic, 2022).
- Elhage et al., *Toy Models of Superposition* (Anthropic, 2022).
- Bricken et al., *Towards Monosemanticity* (Anthropic, 2023).
- Templeton et al., *Scaling Monosemanticity* (Anthropic, 2024).
- Cunningham, Ewart, Riggs, Huben, and Sharkey, *Sparse Autoencoders Find Highly Interpretable Features in Language Models* (2023).
- Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability* (2023).
- Wang et al., *Interpretability in the Wild: A Circuit for Indirect Object Identification* (2022).
- Conmy, Mavor-Parker, Lynch, Heimersheim, and Garriga-Alonso, *Towards Automated Circuit Discovery for Language Models* (DeepMind, 2023).
- TransformerLens (MIT) is the library to use once the task is larger than the one-layer copier in the current lab.

Extension to aim at: an ablation of one head that changes the task accuracy and an ablation of another head that does not.

## Reinforcement learning

Separate from RLHF. Classic control, then a language setting only after the implementation is right.

- Sutton and Barto, *Reinforcement Learning: An Introduction*, for dynamic programming, SARSA, and Q-learning. Reference, not copied.
- Mnih et al., *Human-level Control through Deep Reinforcement Learning* (DeepMind, 2015).
- Schulman et al., PPO (OpenAI, 2017).
- Haarnoja et al., *Soft Actor-Critic* (2018).
- Silver et al., *Mastering the Game of Go without Human Knowledge* (AlphaGo Zero, DeepMind, 2017), as reading on search plus a learned value, not as a lab to reproduce.
- The portfolio trading notebook is a direct policy search on a synthetic path. A later RL lab should not treat that backtest as an environment to beat.

## Diffusion models

- Ho, Jain, and Abbeel, *Denoising Diffusion Probabilistic Models* (2020).
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* (2021).
- Dhariwal and Nichol, *Diffusion Models Beat GANs on Image Synthesis* (OpenAI, 2021).
- Nichol and Dhariwal, *Improved Denoising Diffusion Probabilistic Models* (OpenAI, 2021).
- Ho and Salimans, *Classifier-Free Diffusion Guidance* (2022).
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2022).
- Saharia et al., *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding* (Imagen, Google, 2022).
- Ramesh et al., *Hierarchical Text-Conditional Image Generation with CLIP Latents* (DALL·E 2, OpenAI, 2022).

A first lab can train a small denoiser on a 2D mixture and plot samples across timesteps. That fits on CPU. Latent diffusion on images is a later, GPU, lab.

Hugging Face diffusion course (Apache-2.0 course repo) is the syllabus reference. Do not copy the notebooks. See `docs/labs-guide.md`.

## Multimodal models

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, OpenAI, 2021).
- Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning* (DeepMind, 2022).
- OpenAI, GPT-4 technical report (2023), the vision evaluation section, as a standard for what to report.
- Gemini Team, Gemini technical report (Google DeepMind, 2023).
- The extension: a retrieval setup where the query is text and the items are image embeddings, scored with the same hit rate as the RAG labs, on a tiny synthetic set if the CLIP checkpoint is too heavy for CI.

## Agents and guardrails

The prompting notebook and the evaluation notebook are the start.

- Yao et al., *ReAct* (2022).
- Shinn et al., *Reflexion* (2023).
- Schick et al., *Toolformer* (2023).
- Yao et al., *Tree of Thoughts* (2023).
- Inan et al., *Llama Guard* (Meta, 2023), as a classifier in front of a tool loop. The model weights are a separate license from this repo. Call them. Do not vendor them.
- Bai et al., Constitutional AI (Anthropic, 2022), for a written rule that is critiqued rather than only matched with a regex.
- Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models* (2023), as the problem statement for why a regex guardrail is not a defense. A lab in this public repo should not ship a working attack. Measure the regex against a handful of paraphrases you write yourself, and record the misses.

## Inference optimization and quantization

The portfolio inference lab measures batching and dynamic int8 on CPU.

- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (2018).
- Dettmers et al., *QLoRA* (2023). The LoRA lab is the unquantized half of this.
- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022).
- Lin et al., *AWQ: Activation-aware Weight Quantization* (2023).
- Dao et al., *FlashAttention* (2022), and Dao, *FlashAttention-2* (2023).
- Leviathan, Kalman, and Matias, *Fast Inference from Transformers via Speculative Decoding* (2023).
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023).
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (Google, 2023).
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (2019), on multi-query attention.

GPU timing stays behind `RUN_GPU=1`.

## Distributed training

- Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (NVIDIA, 2019).
- Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (2021).
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020).
- Xu et al., *GSPMD: General and Scalable Parallelization for ML Computation Graphs* (Google, 2021).
- Barham et al., *Pathways* (Google, 2022), as systems reading.
- Hoffmann et al., Chinchilla (DeepMind, 2022), for how much data a given compute budget should see. The current scaling notebook says explicitly that it is not this measurement.

A CPU lab can shard a tiny MLP across two processes with `torch.distributed` on `localhost` and check that the gradient matches the unsharded run. Multi-GPU stays out of CI.

## Semantic caching

The portfolio cache lab is the threshold curve on TF-IDF.

- Bang, GPTCache (Zilliz, Apache-2.0). Design reference. Not copied.
- The extension already started in the lab is invalidation on a corpus version. Next is a shared encoder with the RAG index, so the cache vector and the retrieval vector are the same object, and a measurement of stale hits when they are not.

## Recommendation systems

The next-best-offer lab is a contextual bandit. The rest of the recommender stack is still open.

- Covington, Adams, and Sargin, *Deep Neural Networks for YouTube Recommendations* (Google, 2016).
- Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (Google, 2019).
- He et al., *LightGCN* (2020).
- Li, Chu, Langford, and Schapire, LinUCB (2010), already implemented in the portfolio lab.
- Steck, *Calibrated Recommendations* (2018), as the extension: a policy that matches a user's historical genre mix, and the accuracy it gives up.

Synthetic logs only. No customer data.
