#!/usr/bin/env python3
"""Git-index exec-bit audit: every tracked file with a shebang must be 100755.

Filesystem executability probes are environment-dependent (ruff's EXE001
never fires under WSL2), so the gate audits the git index instead — the same
verdict on every machine. Runs in `mise run gate` and the ci.yml lint step.
"""

import subprocess
import sys


def main() -> int:
    tracked = subprocess.run(
        ["git", "ls-files", "-s"], capture_output=True, text=True, check=True
    ).stdout
    problems = []
    for line in tracked.splitlines():
        if not line:
            continue
        meta, path = line.split("\t", 1)
        mode = meta.split()[0]
        if mode != "100644":
            continue  # 100755 is the convention; symlinks/submodules are skip
        with open(path, "rb") as f:
            if f.read(2) == b"#!":
                problems.append(
                    f"{path}: shebang but index mode is {mode} (want 100755)"
                )
    for p in problems:
        print(f"exec-bit audit: {p}", file=sys.stderr)
    if problems:
        return 1
    print("exec-bit audit: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
