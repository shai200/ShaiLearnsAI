# Adding a lab

Use this when a new course, paper, or private knowledge base should become a notebook. The knowledge base can be notes, a paper PDF, or a link to a public course. The notebook in this repository is still original code unless the source license allows a copy and the copy is marked.

## 1. Check the license before reading code into the notebook

| License of the source | What to put in the repo |
| --- | --- |
| MIT, Apache-2.0, BSD, ISC, CC-BY, Unlicense | You may adapt code. Keep the original copyright notice in `THIRD_PARTY_NOTICES.md` and name the file you adapted. Prefer a short original implementation when the lesson is the algorithm. |
| CC-BY-SA, GPL, or another share-alike / copyleft license | Do not copy code into this MIT repository. Follow the syllabus and link out. |
| "All rights reserved", a course assignment, or no license | Do not copy. Write a new lab on the same topic and link to the course. Stanford CS224N, CS231n, and CS336 assignments are in this category unless a file says otherwise. ARENA notebooks and the Full Stack LLM bootcamp notebooks were treated this way. |
| A paper | The algorithm and the citation are usable. Figures, tables, and prose are not copied. |

If the license is unclear, treat it as not reusable.

## 2. Pick the track and the level

Tracks that exist today:

- `labs/portfolio/` for a lab that demonstrates a production skill end to end.
- `labs/foundations/`, `labs/training/`, `labs/training_llms/`, `labs/finetuning/`, `labs/interpretability/`, `labs/rag/`, `labs/genai_apps/` for the course path.
- A new track gets a new directory, a row in the root `README.md`, and a paragraph in `ROADMAP.md` if it is not built yet.

Number the directory so the path sorts in teaching order: `labs/<track>/03_short_name/`.

Level is one of Foundations, Intermediate, Advanced, Practitioner.

## 3. Write the notebook from the template

Follow `docs/notebook-template.md`. In particular:

- The **Beyond the course** section is required. It is an ablation, a comparison, or a negative result on this lab's own outputs.
- Assertions check an invariant (a gradient matches a finite difference, a masked loss ignores prompt tokens, a citation check rejects a bad number). They should not require a downloaded model or a lucky seed unless you have rerun the cell.
- Keep the CPU path under a few minutes. A lab that only runs on a GPU sets notebook metadata `shai_learns_ai.ci` to `skip` and says so in the title block.

## 4. Dependencies and CI

Add `requirements.txt` in the lab directory listing the packages that notebook imports. Add anything new to the root `requirements.txt` as well.

Run:

```bash
python scripts/execute_notebooks.py labs/<track>/<lab>/<notebook>.ipynb
pytest -q
```

`scripts/execute_notebooks.py` runs every notebook whose metadata does not say `ci: skip`. A cell that needs a key or a GPU must skip itself, as in the template. Do not weaken an assertion to hide a failure.

## 5. Credit it in three places

1. The notebook section **Sources and license**.
2. `THIRD_PARTY_NOTICES.md`, one row: source, license, URL, and whether code was copied.
3. The root `README.md` table: lab, track, level, source, Colab badge.

Colab badge, with the notebook path under `main`:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shai200/ShaiLearnsAI/blob/main/labs/<track>/<lab>/<notebook>.ipynb)
```

## 6. Knowledge bases that are not public courses

When a later batch of notes arrives:

- Extract the claims and the definitions. Do not paste private text verbatim into a public notebook if the notes were not written for publication. Paraphrase, and keep the public notebook self-contained.
- If the notes point at a paper, cite the paper. If they point at an internal system, describe the mechanism with synthetic data the way the portfolio labs do.
- Do not add personal contact details, account identifiers, or customer data to the notebook, the README, or the commit message.
- Run the new notebook on CPU before opening the pull request, and say in the pull request which source it follows and which license you relied on.
