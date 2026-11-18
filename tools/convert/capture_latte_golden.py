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

Three wire names are captured in one process, in corpus order:

  latte_segment       cutoff is the unit (`long`/`short`/`minimal`); expected is
                      the wheel's `tokenize()` list as compact JSON
                      (ensure_ascii=False), which is exactly what the Rust wire
                      returns.

  latte_nbest         cutoff is the wire grammar `<unit>` or `<unit>:<n>`; absent
                      or `medium` means unit `short` and n=10. Expected is the
                      wheel's `tokenize_nbest()` readings, `[{"tokens":[...],
                      "probability":<number>}, ...]`, and the row carries
                      `"compare": "json-f64-1e-4"` because probabilities cannot
                      be byte-exact (the port's logits differ from ONNX Runtime
                      by f32 rounding).

  latte_segment_opts  `text` is the wire's JSON envelope and `cutoff` is ignored
                      (carried only as the golden key). The envelope is validated
                      with the wire's rules — allowlisted keys, no duplicate keys,
                      the 100-texts / 1000-user-words caps, the unit, and the
                      10000-non-whitespace-character cap — before the wheel runs;
                      a rejected envelope is an error row. Success rows capture
                      `Segmenter(user_words=...)`'s `tokenize_batch`. Each text is
                      captured alone and the wheel's batched result must equal the
                      per-text result, because the wire processes texts
                      independently (the accepted batch-dependence departure);
                      a disagreement aborts rather than recording a golden the
                      wire cannot match.

A row the wire rejects (bad unit, n out of range, oversized text, invalid
envelope) becomes a wire-validated error row: `"expected":""` and `"ok":false`.
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
from openthai_latte_wordseg_lite import Segmenter, tokenize, tokenize_nbest  # noqa: E402

UNITS = ("long", "short", "minimal")
MAX_NBEST = 100
MAX_TEXTS = 100
MAX_USER_WORDS = 1000
MAX_TEXT_CHARS = 10_000
OPTION_KEYS = {"text", "texts", "unit", "keep_whitespace", "user_words"}
COMPARE = "json-f64-1e-4"


class BadOptions(Exception):
    """The wire would reject this `latte_segment_opts` envelope."""


def _object_pairs(pairs: list[tuple[str, object]]) -> dict:
    out = {}
    for key, value in pairs:
        if key in out:
            raise BadOptions(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> None:
    raise BadOptions(f"non-finite number {name}")


def opts_args(envelope: str) -> tuple[list[str], str, bool, list[str]]:
    """The wire's validation: (texts, unit, keep_whitespace, user_words)."""
    try:
        obj = json.loads(envelope, object_pairs_hook=_object_pairs, parse_constant=_reject_constant)
    except (json.JSONDecodeError, BadOptions, RecursionError):
        raise BadOptions("malformed JSON") from None
    if not isinstance(obj, dict):
        raise BadOptions("envelope is not an object")
    unknown = set(obj) - OPTION_KEYS
    if unknown:
        raise BadOptions(f"unknown keys {sorted(unknown)}")
    if ("text" in obj) == ("texts" in obj):
        raise BadOptions("exactly one of text/texts is required")
    unit = obj.get("unit", "short")
    if type(unit) is not str or unit not in UNITS:
        raise BadOptions(f"invalid unit {unit!r}")
    keep_whitespace = obj.get("keep_whitespace", True)
    if type(keep_whitespace) is not bool:
        raise BadOptions("keep_whitespace must be a bool")
    user_words = obj.get("user_words", [])
    if type(user_words) is not list or len(user_words) > MAX_USER_WORDS:
        raise BadOptions(f"user_words must be an array of at most {MAX_USER_WORDS} strings")
    if any(type(word) is not str for word in user_words):
        raise BadOptions("user_words must be strings")
    if "text" in obj:
        if type(obj["text"]) is not str:
            raise BadOptions("text must be a string")
        texts = [obj["text"]]
    else:
        if type(obj["texts"]) is not list or len(obj["texts"]) > MAX_TEXTS:
            raise BadOptions(f"texts must be an array of at most {MAX_TEXTS} strings")
        if any(type(text) is not str for text in obj["texts"]):
            raise BadOptions("texts must be strings")
        texts = obj["texts"]
    for text in texts:
        if sum(1 for c in text if not c.isspace()) > MAX_TEXT_CHARS:
            raise BadOptions("text too long")
    return texts, unit, keep_whitespace, user_words


_SEGMENTERS: dict[tuple[str, ...], Segmenter] = {}


def segmenter_with(user_words: list[str]) -> Segmenter:
    """One wheel Segmenter per user-word set (its constructor loads the model)."""
    key = tuple(user_words)
    if key not in _SEGMENTERS:
        _SEGMENTERS[key] = Segmenter(user_words=list(user_words))
    return _SEGMENTERS[key]


def nbest_args(cutoff: str) -> tuple[str, int]:
    """The wire's n-best cutoff grammar: `<unit>` | `<unit>:<n>`; "" and "medium" mean short, n=10."""
    if cutoff in ("", "medium"):
        return "short", 10
    unit, sep, n = cutoff.partition(":")
    if unit not in UNITS:
        raise ValueError(f"invalid unit {unit!r}")
    if not sep:
        return unit, 10
    if not n.isascii() or not n.isdigit() or not 1 <= int(n) <= MAX_NBEST:
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
            elif function == "latte_segment_opts":
                # cutoff is ignored by the wire; it only keys this golden row
                try:
                    texts, unit, keep_whitespace, user_words = opts_args(row["text"])
                except BadOptions:
                    emit_err(row, cutoff)
                    continue
                seg = segmenter_with(user_words)
                alone = [seg.tokenize_batch([text], keep_whitespace, unit)[0] for text in texts]
                batched = seg.tokenize_batch(texts, keep_whitespace, unit)
                if batched != alone:
                    raise SystemExit(
                        "latte_segment_opts: the wheel's batched result disagrees with "
                        f"per-text processing for {row['text']!r}; the wire is per-text "
                        "by design — shorten or split this row"
                    )
                expected = json.dumps(alone, ensure_ascii=False, separators=(",", ":"))
                emit_ok(row, cutoff, expected, None)
            else:
                raise SystemExit(f"unknown function in corpus: {function}")


if __name__ == "__main__":
    main()
