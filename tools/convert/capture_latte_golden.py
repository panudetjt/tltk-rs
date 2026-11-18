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

Both wire names are captured in one process, in corpus order:

  latte_segment  cutoff is the unit (`long`/`short`/`minimal`); expected is the
                 wheel's `tokenize()` list as compact JSON (ensure_ascii=False),
                 which is exactly what the Rust wire returns.

  latte_nbest    cutoff is the wire grammar `<unit>` or `<unit>:<n>`; absent or
                 `medium` means unit `short` and n=10. Expected is the wheel's
                 `tokenize_nbest()` readings, `[{"tokens":[...],"probability":
                 <number>}, ...]`, and the row carries `"compare":
                 "json-f64-1e-4"` because probabilities cannot be byte-exact
                 (the port's logits differ from ONNX Runtime by f32 rounding).

A row the wheel rejects (bad unit, n out of range, oversized text) becomes a
wire-validated error row: `"expected":""` and `"ok":false`.
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
from openthai_latte_wordseg_lite import tokenize, tokenize_nbest  # noqa: E402

UNITS = ("long", "short", "minimal")
MAX_NBEST = 100
COMPARE = "json-f64-1e-4"


def nbest_args(cutoff: str) -> tuple[str, int]:
    """The wire's n-best cutoff grammar: `<unit>` | `<unit>:<n>`; "" and "medium" mean short, n=10."""
    if cutoff in ("", "medium"):
        return "short", 10
    unit, _, n = cutoff.partition(":")
    if unit not in UNITS:
        raise ValueError(f"invalid unit {unit!r}")
    if not n:
        return unit, 10
    if not n.isdigit() or not 1 <= int(n) <= MAX_NBEST:
        raise ValueError("n must be between 1 and 100")
    return unit, int(n)


def emit_ok(row: dict, cutoff: str, expected: str, compare: str | None) -> None:
    out = {"function": row["function"], "text": row["text"], "cutoff": cutoff}
    if compare is not None:
        out["compare"] = compare
    out["expected"] = expected
    out["ok"] = True
    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))


def emit_err(row: dict, cutoff: str) -> None:
    print(json.dumps(
        {"function": row["function"], "text": row["text"], "cutoff": cutoff,
         "expected": "", "ok": False},
        ensure_ascii=False, separators=(",", ":"),
    ))


def main() -> None:
    with open(sys.argv[2], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            function = row["function"]
            # the verifier keys goldens on the row's cutoff, or "medium" when absent
            cutoff = row.get("cutoff", "medium")
            if function == "latte_segment":
                # the wire's latte_segment maps anything but a known unit to `short`
                unit = cutoff if cutoff in UNITS else "short"
                expected = json.dumps(
                    tokenize(row["text"], unit=unit), ensure_ascii=False, separators=(",", ":")
                )
                emit_ok(row, cutoff, expected, None)
            elif function == "latte_nbest":
                try:
                    unit, n = nbest_args(cutoff)
                    readings = tokenize_nbest(row["text"], n, unit)
                except ValueError:
                    emit_err(row, cutoff)
                    continue
                expected = json.dumps(
                    [{"tokens": toks, "probability": p} for toks, p in readings],
                    ensure_ascii=False, separators=(",", ":"),
                )
                emit_ok(row, cutoff, expected, COMPARE)
            else:
                raise SystemExit(f"unknown function in corpus: {function}")


if __name__ == "__main__":
    main()
