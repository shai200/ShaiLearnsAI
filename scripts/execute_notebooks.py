"""Execute CPU notebooks. GPU-only and key-only cells skip themselves.

A notebook is skipped entirely when its metadata says ``shai_learns_ai.ci``
is ``skip``. Every notebook in the first batch is ``cpu`` and is executed.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def notebooks(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.ipynb") if ".ipynb_checkpoints" not in path.parts)


def execute(path: Path, timeout: int, write: bool) -> None:
    nb = nbformat.read(path, as_version=4)
    meta = nb.metadata.get("shai_learns_ai", {})
    if meta.get("ci") == "skip":
        print(f"SKIP {path}")
        return
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    if write:
        nbformat.write(nb, path)
    print(f"OK   {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="labs")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    if args.paths:
        paths = []
        for raw in args.paths:
            path = Path(raw)
            if path.is_dir():
                paths.extend(notebooks(path))
            else:
                paths.append(path)
    else:
        paths = notebooks(Path(args.root))
    failed = []
    for path in paths:
        try:
            execute(path, args.timeout, args.write)
        except CellExecutionError as exc:
            failed.append(path)
            print(f"FAIL {path}\n{exc}", file=sys.stderr)
        except Exception:
            failed.append(path)
            print(f"FAIL {path}", file=sys.stderr)
            traceback.print_exc()
    if failed:
        print(f"{len(failed)} notebook(s) failed", file=sys.stderr)
        return 1
    print(f"{len(paths)} notebook(s) executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
