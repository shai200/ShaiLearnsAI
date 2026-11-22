"""Write notebooks with a shared header and metadata."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def make_notebook(path: str | Path, cells: list[tuple[str, str]], extra: dict) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "pygments_lexer": "ipython3",
    }
    nb.metadata["shai_learns_ai"] = {"ci": "cpu", **extra}
    built = []
    for kind, source in cells:
        text = source.strip("\n") + "\n"
        if kind == "md":
            built.append(nbf.v4.new_markdown_cell(text))
        elif kind == "code":
            built.append(nbf.v4.new_code_cell(text))
        else:
            raise ValueError(kind)
    nb.cells = built
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print("wrote", path)


BOOT = '''
import os
import sys
from pathlib import Path

def find_repo_root() -> Path:
    start = Path.cwd().resolve()
    for cand in [start, *start.parents]:
        if (cand / "labs" / "_shared" / "llm.py").exists():
            return cand
    if "google.colab" in sys.modules:
        root = Path("/content/ShaiLearnsAI")
        marker = root / "labs" / "_shared" / "llm.py"
        if not marker.exists():
            os.system(
                "git clone --depth 1 https://github.com/shai200/ShaiLearnsAI.git /content/ShaiLearnsAI"
            )
        return root
    raise RuntimeError(
        "Clone https://github.com/shai200/ShaiLearnsAI and run this notebook from that directory."
    )

ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
print("repo", ROOT)
'''


def colab(packages: list[str]) -> str:
    pkg_list = ", ".join(repr(p) for p in packages)
    return f'''
import subprocess
import sys
if "google.colab" in sys.modules:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", {pkg_list}])
print("packages ready")
'''
