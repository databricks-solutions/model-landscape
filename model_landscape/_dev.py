"""Developer-experience helpers exposed as console scripts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def init_hooks() -> int:
    """Point `git config core.hooksPath` at the repo's `.githooks/` directory.

    Idempotent. Run once per fresh clone:

        uv run model-landscape-init-hooks

    The hook itself is checked in at `.githooks/pre-commit` — this just
    wires git to look there instead of the default `.git/hooks/`.
    """
    if not shutil.which("git"):
        print("git not found on PATH", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    hooks_dir = repo_root / ".githooks"

    if not hooks_dir.is_dir():
        print(f"missing hooks dir: {hooks_dir}", file=sys.stderr)
        return 1

    result = subprocess.run(
        ["git", "-C", str(repo_root), "config", "core.hooksPath", ".githooks"],
        check=False,
    )
    if result.returncode != 0:
        return result.returncode

    print(f"✓ core.hooksPath set to .githooks for {repo_root}")
    return 0


if __name__ == "__main__":
    sys.exit(init_hooks())
