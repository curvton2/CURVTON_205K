#!/usr/bin/env python3
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    cur = start
    while cur != cur.parent:
        if (cur / "shared" / "qwen_edit_prompts_core.py").exists():
            return cur
        cur = cur.parent
    return start


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve())
    shared_dir = repo_root / "shared"
    sys.path.insert(0, str(shared_dir))
    from qwen_edit_prompts_core import main as shared_main
    shared_main()


if __name__ == "__main__":
    main()
