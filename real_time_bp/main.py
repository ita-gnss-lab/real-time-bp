from __future__ import annotations

import subprocess


def main() -> None:
    subprocess.run(["dvc", "repro"], check=True)


if __name__ == "__main__":
    main()
