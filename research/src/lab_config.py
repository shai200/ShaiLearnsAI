"""Shared configuration for the "Training LLMs & Frontier Research" notebooks.

Every notebook does::

    from lab_config import CFG, set_seed, pick
    set_seed()

and then sizes its experiments with ``pick(cpu_value, full_value)`` so that the
default run is tiny (CPU, seconds-to-minutes) and ``LAB_FULL_SIZE=1`` switches
to the larger, closer-to-the-paper setting (usually wants a GPU).

Environment variables
---------------------
LAB_FULL_SIZE=1        use full-size settings (default: 0 -> tiny CPU settings)
LAB_SEED=1337          global random seed
LAB_DEVICE=cpu|cuda    force a device (default: cuda if available, else cpu)
LAB_NUM_THREADS=N      torch CPU threads (default: min(8, os.cpu_count()))
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SRC_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SRC_DIR.parent
DATA_DIR = RESEARCH_ROOT / "data"
OUTPUT_DIR = RESEARCH_ROOT / "outputs"
CACHE_DIR = RESEARCH_ROOT / ".cache"


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _detect_device() -> str:
    forced = os.environ.get("LAB_DEVICE")
    if forced:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover - torch is a hard requirement in practice
        pass
    return "cpu"


@dataclass
class LabConfig:
    full_size: bool = field(default_factory=lambda: _env_flag("LAB_FULL_SIZE"))
    seed: int = field(default_factory=lambda: int(os.environ.get("LAB_SEED", "1337")))
    device: str = field(default_factory=_detect_device)
    num_threads: int = field(
        default_factory=lambda: int(os.environ.get("LAB_NUM_THREADS", min(8, os.cpu_count() or 1)))
    )
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    cache_dir: Path = CACHE_DIR

    @property
    def has_gpu(self) -> bool:
        return self.device.startswith("cuda")

    def summary(self) -> str:
        mode = "FULL_SIZE (closer to paper scale)" if self.full_size else "CPU-friendly tiny defaults"
        return (
            f"mode={mode} | device={self.device} | seed={self.seed} | "
            f"threads={self.num_threads} | research_root={RESEARCH_ROOT}"
        )


CFG = LabConfig()
FULL_SIZE: bool = CFG.full_size
DEVICE: str = CFG.device

# Make sure artifact directories exist (both are git-ignored).
CFG.output_dir.mkdir(parents=True, exist_ok=True)
CFG.cache_dir.mkdir(parents=True, exist_ok=True)


def pick(cpu_value, full_value):
    """Return ``full_value`` when LAB_FULL_SIZE=1, otherwise the tiny ``cpu_value``."""
    return full_value if CFG.full_size else cpu_value


def set_seed(seed: int | None = None) -> int:
    """Seed python, numpy and torch (CPU + CUDA) for reproducible runs."""
    seed = CFG.seed if seed is None else seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(CFG.num_threads)
    except ImportError:
        pass
    return seed


def heavy_cell(reason: str, needs_gpu: bool = False) -> bool:
    """Gate for long-running / GPU-only cells.

    Returns True if the cell should run. Prints a clear message otherwise, so a
    reader of the executed notebook knows *why* the cell was skipped.
    """
    if not CFG.full_size:
        print(f"[skipped] {reason} -- set LAB_FULL_SIZE=1 to run this cell.")
        return False
    if needs_gpu and not CFG.has_gpu:
        print(f"[skipped] {reason} -- needs a CUDA GPU (none detected).")
        return False
    return True


def read_text(name: str) -> str:
    """Read a vendored text file from research/data/."""
    return (CFG.data_dir / name).read_text(encoding="utf-8")


def quiet_hf_logging() -> None:
    """Silence the most common noisy-but-harmless HF / TL warnings in notebooks."""
    import logging
    import warnings

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers|trl|peft|torch")
    for name in ("transformers", "trl", "peft", "datasets", "accelerate", "transformer_lens"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from datasets.utils.logging import disable_progress_bar

        disable_progress_bar()
    except Exception:
        pass


if __name__ == "__main__":
    print(CFG.summary())
