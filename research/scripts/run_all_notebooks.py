#!/usr/bin/env python
"""Execute every research notebook top-to-bottom and report pass/fail.

Usage (from anywhere)::

    python research/scripts/run_all_notebooks.py                 # all notebooks, CPU tiny defaults
    python research/scripts/run_all_notebooks.py 03 07           # only notebooks whose name contains "03" or "07"
    python research/scripts/run_all_notebooks.py --report        # also regenerate the table in outputs/execution_report.md
    python research/scripts/run_all_notebooks.py --full-size     # LAB_FULL_SIZE=1 (slow; GPU recommended)

Executed copies (with outputs) are written to ``research/outputs/executed/``
(git-ignored); the source notebooks are never modified. Exits non-zero if any
notebook fails.

``--report`` regenerates the header and results table of the report and keeps
everything below the ``NOTES_MARKER`` line (the hand-written per-notebook notes)
untouched. It is ignored for subset runs, so the committed report always
describes a full run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

RESEARCH = Path(__file__).resolve().parents[1]
NB_DIR = RESEARCH / "notebooks"
OUT_DIR = RESEARCH / "outputs"
EXEC_DIR = OUT_DIR / "executed"
NOTES_MARKER = "<!-- Hand-written notes below this line are preserved by --report. Update them when results change. -->"
VERSION_PACKAGES = ("torch", "transformers", "trl", "peft", "datasets", "transformer-lens")


def count_skipped(nb) -> list[str]:
    """Collect the '[skipped] ...' messages printed by lab_config.heavy_cell()."""
    msgs = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                msgs += [ln.strip() for ln in out.get("text", "").splitlines() if ln.strip().startswith("[skipped]")]
    return msgs


def run_one(path: Path, timeout: int, kernel: str) -> dict:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel,
        resources={"metadata": {"path": str(path.parent)}},  # cwd = notebooks/
    )
    t0 = time.time()
    status, error = "passed", ""
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError) as exc:
        status = "failed"
        error = str(exc).strip().splitlines()[-1][:500] if str(exc).strip() else repr(exc)
    except Exception as exc:  # kernel died, etc.
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"[:500]
        traceback.print_exc()
    seconds = time.time() - t0
    EXEC_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, EXEC_DIR / path.name)
    return {
        "notebook": path.name,
        "status": status,
        "seconds": round(seconds, 1),
        "error": error,
        "skipped_cells": count_skipped(nb),
    }


def provider_desc() -> str:
    explicit = os.environ.get("LAB_LLM_PROVIDER")
    if explicit:
        return f"`LAB_LLM_PROVIDER={explicit}`"
    if os.environ.get("OPENAI_API_KEY"):
        return "OpenAI (OPENAI_API_KEY set)"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "Anthropic (ANTHROPIC_API_KEY set)"
    return "local Hugging Face fallback (no API keys set)"


def package_versions() -> str:
    from importlib import metadata

    found = []
    for pkg in VERSION_PACKAGES:
        try:
            found.append(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            found.append(f"{pkg} (not installed)")
    return ", ".join(found)


def write_report(results: list[dict], args) -> Path:
    total = sum(r["seconds"] for r in results)
    path = OUT_DIR / "execution_report.md"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    manual_notes = old[old.index(NOTES_MARKER):].rstrip() if NOTES_MARKER in old else NOTES_MARKER
    lines = [
        "# Execution report: Training LLMs & Frontier Research track",
        "",
        f"* Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by `research/scripts/run_all_notebooks.py`",
        f"* Mode: `LAB_FULL_SIZE={'1' if args.full_size else '0'}` "
        f"({'full size' if args.full_size else 'CPU tiny defaults'})",
        f"* Python {platform.python_version()} on {platform.system()} {platform.machine()}, "
        f"{os.cpu_count()} CPU threads, no GPU required",
        f"* Packages: {package_versions()}",
        f"* LLM provider: {provider_desc()}",
        f"* Total wall-clock: {total / 60:.1f} min",
        "",
        "| Notebook | Result | Runtime | Notes |",
        "|---|---|---|---|",
    ]
    for r in results:
        notes = []
        if r["skipped_cells"]:
            notes.append(f"{len(r['skipped_cells'])} heavy/GPU cell(s) skipped by design")
        if r["error"]:
            notes.append(f"error: `{r['error']}`")
        if r.get("note"):
            notes.append(r["note"])
        icon = "✅ ran clean" if r["status"] == "passed" else "❌ failed"
        lines.append(f"| `{r['notebook']}` | {icon} | {r['seconds']:.0f} s | {'; '.join(notes) or '—'} |")
    lines += ["", "Skipped heavy cells (run with `LAB_FULL_SIZE=1`, some also need a GPU):", ""]
    for r in results:
        for msg in r["skipped_cells"]:
            lines.append(f"* `{r['notebook']}`: {msg.removeprefix('[skipped] ')}")
    lines += ["", manual_notes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filters", nargs="*", help="substrings to select notebooks (default: all)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--full-size", action="store_true", help="set LAB_FULL_SIZE=1")
    ap.add_argument("--report", action="store_true", help="write outputs/execution_report.md")
    args = ap.parse_args()

    os.environ["LAB_FULL_SIZE"] = "1" if args.full_size else "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    notebooks = sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))
    if args.filters:
        notebooks = [p for p in notebooks if any(f in p.name for f in args.filters)]
    if not notebooks:
        print("No notebooks matched.", file=sys.stderr)
        return 2

    results = []
    for path in notebooks:
        print(f"▶ {path.name} ...", flush=True)
        res = run_one(path, args.timeout, args.kernel)
        results.append(res)
        mark = "PASS" if res["status"] == "passed" else "FAIL"
        print(f"  {mark} in {res['seconds']:.0f}s" + (f" -- {res['error']}" if res["error"] else ""), flush=True)

    print("\nSummary")
    print("-------")
    for r in results:
        print(f"{r['status'].upper():7} {r['seconds']:7.1f}s  {r['notebook']}")
    (OUT_DIR / "execution_results.json").write_text(json.dumps(results, indent=2))
    if args.report and args.filters:
        print("report: not rewritten (subset run); run without filters to regenerate it.")
    elif args.report:
        print("report:", write_report(results, args))
    failed = [r for r in results if r["status"] != "passed"]
    print(f"\n{len(results) - len(failed)}/{len(results)} notebooks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
