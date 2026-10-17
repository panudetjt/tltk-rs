#!/usr/bin/env python3
"""Bundled-license sync audit: per-crate copies must byte-match the canonical
root files.

The published crates bundle LICENSE and LICENSE-THIRD-PARTY copies (cargo
auto-includes LICENSE* at the package root) so license text travels with the
crates.io package. Copies drift silently when a root notice is edited, so the
gate audits content, not memory. Plain file comparison — no git, no cargo —
the same verdict on every machine. Runs in `mise run gate` and the ci.yml
lint step.
"""

import filecmp
import sys
from pathlib import Path

BUNDLED = "LICENSE-THIRD-PARTY"
CANONICAL = {
    "LICENSE": Path("LICENSE"),
    BUNDLED: Path("THIRD_PARTY_LICENSES.md"),
}
CRATES = ("crates/malt-parser", "crates/tltk-core")


def main() -> int:
    problems = []
    for crate in CRATES:
        for bundled, canonical in CANONICAL.items():
            copy = Path(crate) / bundled
            if not canonical.is_file():
                problems.append(f"{canonical}: canonical file missing")
            elif not copy.is_file():
                problems.append(
                    f"{copy}: missing — the published package would ship no "
                    f"{bundled} (fix: cp {canonical} {copy})"
                )
            elif not filecmp.cmp(copy, canonical, shallow=False):
                problems.append(
                    f"{copy}: drifted from canonical {canonical} "
                    f"(fix: cp {canonical} {copy})"
                )
    for p in problems:
        print(f"bundled-license audit: {p}", file=sys.stderr)
    if problems:
        return 1
    print("bundled-license audit: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
