# Notebook template

Every lab notebook uses the same sections, in this order. A section can be short. It should not be missing.

1. **Title block.** Track, level, whether the notebook needs a GPU or an API key.
2. **Goals.** What the notebook will run, in concrete terms.
3. **Sources and license.** The paper or course the lab follows, the license of that source, and a direct statement of whether any of its code was copied.
4. **Concepts.** The definitions a reader needs before the first plot. Write them as claims a later cell will measure.
5. **Setup.** Locate the repository, set the seed, import the shared client if the lab calls a model.
6. **Guided exercise.** Working code. Assertions check the property the section claimed, not a leaderboard number copied from a paper.
7. **Your turn.** A specific change. The following cell is the solution, and it runs.
8. **Beyond the course.** One extension that is not in the source: an ablation, a comparison, or a small negative result. State what was measured.
9. **Research questions.** Three questions the lab does not answer.
10. **Further reading.** Papers and primary docs, with links.

Code cells that need a GPU start with a guard:

```python
import os
import torch
run_gpu = os.environ.get("RUN_GPU", "0") == "1" and torch.cuda.is_available()
if not run_gpu:
    print("SKIP GPU cell: set RUN_GPU=1 on a CUDA machine. CPU CI does not run this cell.")
else:
    ...
```

Model calls go through `labs._shared.llm.get_llm`. Notebooks default to `os.environ.get("LAB_LLM", "local")` so a machine that happens to have an API key does not spend it during a test run.

Notebook metadata:

```json
{"shai_learns_ai": {"ci": "cpu", "needs_gpu": false, "needs_api_key": false}}
```

Set `ci` to `skip` only when the notebook cannot run without a key or a GPU. `scripts/execute_notebooks.py` skips those and runs everything else.
