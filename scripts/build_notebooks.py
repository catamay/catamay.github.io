from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_GLOB = "assets/demos/**/*.ipynb"


def build_notebook(notebook_path: Path) -> None:
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--output",
            notebook_path.name,
            "--output-dir",
            str(notebook_path.parent),
            str(notebook_path),
        ],
        check=True,
        cwd=ROOT,
    )


def main() -> None:
    notebooks = sorted(ROOT.glob(NOTEBOOK_GLOB))
    if not notebooks:
        print("No notebooks found.")
        return

    for notebook_path in notebooks:
        print(f"Converting {notebook_path.relative_to(ROOT)}")
        build_notebook(notebook_path)


if __name__ == "__main__":
    main()