#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx", "numpy"]
# ///
"""Convert the iApp OpenThai 2.0 LATTE-WordSeg-Lite assets into the compact
binary blobs read by `crates/tltk-core/src/latte.rs`.

Usage:
  uv run tools/convert/convert_latte.py \
    crates/tltk-core/data/latte_model.int8.onnx \
    crates/tltk-core/data/latte_lexicon.txt.gz \
    crates/tltk-core/data

Outputs (all little-endian, gzip mtime=0 for reproducibility):
  latte_weights.bin.gz  LT01 - every ONNX initializer: name, dtype, shape, bytes
  latte_lex.bin.gz      LX01 - 64-bit poly hashes of lexicon words, by codepoint length

The charclass table (`latte_charclass.bin.gz`) is already the exact i16 map
over all 0x110000 code points; Rust reads it directly, no conversion needed.
"""
import gzip
import os
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

MUL = 0x9E3779B97F4A7C15
MASK = 0xFFFFFFFFFFFFFFFF


def uvarint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def wstr(s: str) -> bytes:
    b = s.encode("utf-8")
    return uvarint(len(b)) + b


def gz(path: str, payload: bytes) -> None:
    with open(path, "wb") as f:
        f.write(gzip.compress(payload, compresslevel=6, mtime=0))
    print(f"  {os.path.basename(path):24s} raw={len(payload) / 1e6:8.3f}MB gz={os.path.getsize(path) / 1e6:8.3f}MB")


def conv_weights(onnx_path: str, out_dir: str) -> None:
    model = onnx.load(onnx_path)
    graph = model.graph
    out = bytearray(b"LT01")
    out += uvarint(len(graph.initializer))
    print(f"initializers: {len(graph.initializer)}")
    total = 0
    for t in graph.initializer:
        arr = numpy_helper.to_array(t)
        # ONNX raw_data is little-endian; force it explicitly for portability.
        dt = arr.dtype.newbyteorder("<")
        raw = np.ascontiguousarray(arr.astype(dt, copy=False)).tobytes()
        total += len(raw)
        out += wstr(t.name)
        out += uvarint(t.data_type)
        out += uvarint(len(t.dims))
        for d in t.dims:
            out += uvarint(int(d))
        out += uvarint(len(raw))
        out += raw
    print(f"  total initializer bytes: {total / 1e6:.3f}MB")
    gz(os.path.join(out_dir, "latte_weights.bin.gz"), bytes(out))


def conv_lexicon(lex_path: str, out_dir: str) -> None:
    with gzip.open(lex_path, "rt", encoding="utf-8", newline="") as f:
        words = f.read().split("\n")

    by_len: dict[int, list[int]] = {}
    for w in words:
        n = len(w)  # Python: code points (Rust .chars().count()), NOT UTF-16 units
        if n < 2 or n > 16:
            continue
        if any(c.isspace() for c in w):
            continue
        codes = [ord(c) for c in w]
        h = codes[0]
        for c in codes[1:]:
            h = (h * MUL + c) & MASK
        by_len.setdefault(n, []).append(h)

    lens = sorted(by_len)
    out = bytearray(b"LX01")
    out += uvarint(len(lens))
    for n in lens:
        vals = sorted(set(by_len[n]))
        out += uvarint(n)
        out += uvarint(len(vals))
        for v in vals:
            out += struct.pack("<Q", v)
    print(f"lexicon lengths: {lens}  words={sum(len(v) for v in by_len.values())}")
    gz(os.path.join(out_dir, "latte_lex.bin.gz"), bytes(out))


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(2)
    _, onnx_path, lex_path, out_dir = sys.argv
    conv_weights(onnx_path, out_dir)
    conv_lexicon(lex_path, out_dir)


if __name__ == "__main__":
    main()
