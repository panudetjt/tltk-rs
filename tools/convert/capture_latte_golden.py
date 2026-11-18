#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnxruntime", "numpy"]
# ///
"""Capture LATTE goldens from the iApp Python wheel (NEVER from the Rust binary).

Usage:
  uv run --with /path/to/openthai2_0_latte_wordseg_lite-0.1.2-py3-none-any.whl \
    tools/convert/capture_latte_golden.py \
    /path/to/extracted/wheel/dir \
    corpus/corpus_fn_latte.jsonl \
    > golden/local_python_fn_latte.jsonl

The corpus row's `cutoff` is the unit (`long`/`short`/`minimal`); `expected` is
the wheel's `tokenize()` list serialized as compact JSON (ensure_ascii=False),
which is exactly what the Rust wire returns.
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
from openthai_latte_wordseg_lite import tokenize  # noqa: E402


def main() -> None:
    with open(sys.argv[2], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            unit = row.get("cutoff", "short")
            expected = json.dumps(
                tokenize(row["text"], unit=unit), ensure_ascii=False, separators=(",", ":")
            )
            print(json.dumps(
                {"function": row["function"], "text": row["text"], "cutoff": unit,
                 "expected": expected, "ok": True},
                ensure_ascii=False, separators=(",", ":"),
            ))


if __name__ == "__main__":
    main()
