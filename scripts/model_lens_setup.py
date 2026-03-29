from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_repo_src() -> None:
    src = Path.cwd() / "src"
    if not src.is_dir():
        raise RuntimeError("Run this helper from the Model Lens repo root or set PYTHONPATH=src.")
    src_path = str(src)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> int:
    _bootstrap_repo_src()
    from model_lens.workflows.setup_control_plane import main as setup_main

    return int(setup_main() or 0)

if __name__ == "__main__":
    main()
