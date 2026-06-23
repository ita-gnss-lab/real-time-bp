"""Command-line entry point for reproducing the DVC pipeline."""

from __future__ import annotations

import subprocess


def main() -> None:
    """Run `dvc repro` with the repository's active DVC graph."""
    subprocess.run(["dvc", "repro"], check=True)


if __name__ == "__main__":
    main()
