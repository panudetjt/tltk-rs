#!/usr/bin/env python3
"""Capture golden responses from the stichula validator (the parity oracle).

Usage: capture_stichula.py <corpus.jsonl> <out.jsonl> [base-url]
Polite: 1s sleep between requests; cached rows are kept if out exists.
"""

import json
import os
import sys
import time
import urllib.request

BASE = "https://app.stichula.org/py/tltk/process"


def post(url, payload, timeout=120):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    corpus_path = sys.argv[1]
    out_path = sys.argv[2]
    base = sys.argv[3] if len(sys.argv) > 3 else BASE
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as done_file:
            for line in done_file:
                if line.strip():
                    d = json.loads(line)
                    done.add((d["function"], d["text"], d.get("cutoff", "medium")))
    n_new = 0
    with (
        open(out_path, "a", encoding="utf-8") as out,
        open(corpus_path, encoding="utf-8") as corpus_file,
    ):
        for line in corpus_file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fn, text = row["function"], row["text"]
            cutoff = row.get("cutoff", "medium")
            if (fn, text, cutoff) in done:
                continue
            try:
                resp = post(base, {"text": text, "function": fn, "cutoff": cutoff})
                expected = resp.get("result", "")
                ok = bool(resp.get("success"))
                if not ok:
                    expected = resp.get("error", "")
            except Exception as e:  # noqa: BLE001 — any capture failure is recorded as a row
                expected = f"!CAPTURE-ERROR! {e}"
                ok = False
            out.write(
                json.dumps(
                    {
                        "function": fn,
                        "text": text,
                        "cutoff": cutoff,
                        "expected": expected,
                        "ok": ok,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            n_new += 1
            print(f"captured {fn} {text[:24]!r} ok={ok}", flush=True)
            time.sleep(1.0)
    print(f"done: {n_new} new rows")


if __name__ == "__main__":
    main()
