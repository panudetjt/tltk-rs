#!/usr/bin/env python3
"""Summarize a bench-report artifacts directory into summary.md.

Reads bench_py_<n>.json / bench_rs_<n>.json / bench_wasm_simd_<n>.json /
bench_wasm_scalar_<n>.json / mem.json produced by scripts/bench_report.sh,
takes the best (lowest) median per function and lane across rounds, and
writes/prints a markdown summary that embeds the recorded environments from
the artifacts themselves — the published numbers and their measurement
conditions always travel together.

Lanes: python oracle (tltk 1.11), native rust (--release), and the shipped
browser wasm (simd128 + scalar flavors) under Node.

Usage: python3 scripts/bench_summary.py <artifacts-dir>
"""

import glob
import json
import os
import sys

DISPLAY_ORDER = [
    "segment",
    "syl_segment",
    "word_segment",
    "ner_tag",
    "g2p",
    "th2roman",
    "th2ipa",
    "pos_tag",
    "th2read",
    "spell_variants",
    "spoonerism",
    "plotwords",
    "similar_words",
    "analogy",
    "similarity",
    "outofgroup",
]

LANES = [
    ("py", "bench_py", "tltk 1.11 (Python) µs/pass"),
    ("rs", "bench_rs", "native Rust µs/pass"),
    ("ws", "bench_wasm_simd", "wasm simd128 µs/pass"),
    ("wc", "bench_wasm_scalar", "wasm scalar µs/pass"),
]


def load_rounds(directory, prefix):
    rounds = []
    for path in sorted(
        glob.glob(os.path.join(directory, f"{prefix}_*.json")),
        key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]),
    ):
        with open(path, encoding="utf-8") as f:
            rounds.append(json.load(f))
    return rounds


def fn_table(report):
    return report.get("functions") or report.get("per_function") or {}


def fmt(n):
    return f"{n:,}" if isinstance(n, int) else str(n)


def verify_table(rows, lane_keys):
    """Cross-foot the rendered table: every speedup/overhead cell must
    recompute from the numbers printed in its own row, and the whole-corpus
    row must equal the per-function sums. A rendered number that disagrees
    with its own inputs means the table lies — refuse to write it."""
    npresent = len(lane_keys)

    def cells(row):
        return [c.strip().strip("*").strip() for c in row.strip().strip("|").split("|")]

    def num(s):
        return int(s.replace(",", ""))

    def ratio_cell(s, want):
        if s == "—" or want in (0, None):
            return
        if abs(float(s.rstrip("x")) - want) > 0.05:
            sys.exit(f"ERROR summary self-check: '{s}' != {want:.1f}x")

    parsed = [cells(row) for row in rows[2:]]
    sums = [0] * npresent
    for cs in parsed[:-1]:
        vals = [num(c) for c in cs[1 : 1 + npresent]]
        for i, v in enumerate(vals):
            sums[i] += v
        ratio_cell(cs[1 + npresent], vals[0] / vals[1] if vals[1] else None)
        if npresent >= 3:
            ratio_cell(cs[2 + npresent], vals[2] / vals[1] if vals[1] else None)
    total = parsed[-1]
    vals = [num(c) for c in total[1 : 1 + npresent]]
    for i, v in enumerate(vals):
        if v != sums[i]:
            sys.exit(
                f"ERROR summary self-check: total {lane_keys[i]} cell "
                f"{v} != per-function sum {sums[i]}"
            )
    ratio_cell(total[1 + npresent], vals[0] / vals[1] if vals[1] else None)
    if npresent >= 3:
        ratio_cell(total[2 + npresent], vals[2] / vals[1] if vals[1] else None)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    directory = sys.argv[1]
    rounds = {key: load_rounds(directory, prefix) for key, prefix, _ in LANES}
    if not rounds["py"] or not rounds["rs"]:
        sys.exit(f"ERROR: no python/rust bench JSONs found in {directory}")
    present = [(key, title) for key, _, title in LANES if rounds[key]]

    fns = list(fn_table(rounds["py"][0]).keys())
    order = [f for f in DISPLAY_ORDER if f in fns] + [
        f for f in fns if f not in DISPLAY_ORDER
    ]

    def best(key, fn):
        return min(fn_table(r)[fn]["median_us_per_pass"] for r in rounds[key])

    header = "| function | " + " | ".join(title for _, title in present) + " |"
    header += " py→native | simd→native |"
    sep = "|---|" + "---:|" * len(present) + "---:|---:|"
    rows = [header, sep]
    for fn in order:
        cells = []
        for key, _ in present:
            cells.append(fmt(best(key, fn)))
        py_best, rs_best = best("py", fn), best("rs", fn)
        speedup = f"{py_best / rs_best:.1f}x" if rs_best else "—"
        if rounds["ws"]:
            ws_best = best("ws", fn)
            overhead = f"{ws_best / rs_best:.1f}x" if rs_best and ws_best else "—"
        else:
            overhead = "—"
        rows.append(f"| {fn} | " + " | ".join(cells) + f" | {speedup} | {overhead} |")
    # whole-corpus totals: sum of per-function bests per lane
    lane_sums = {key: sum(best(key, fn) for fn in fns) for key, _ in present}
    tot_cells = [f"**{fmt(lane_sums[key])}**" for key, _ in present]
    speedup = f"{lane_sums['py'] / lane_sums['rs']:.1f}x" if lane_sums["rs"] else "—"
    if rounds["ws"]:
        overhead = (
            f"{lane_sums['ws'] / lane_sums['rs']:.1f}x"
            if lane_sums["rs"] and lane_sums.get("ws")
            else "—"
        )
    else:
        overhead = "—"
    rows.append(
        "| **whole corpus pass** | "
        + " | ".join(tot_cells)
        + f" | **{speedup}** | **{overhead}** |"
    )

    verify_table(rows, [k for k, _ in present])

    env_py = rounds["py"][0].get("environment", {})
    env_rs = rounds["rs"][0].get("environment", {})

    def env_block(tag, env, extra=()):
        items = [
            ("timestamp", env.get("timestamp_utc")),
            ("host", env.get("host")),
            ("os/kernel", f"{env.get('os')} / {env.get('kernel')}"),
            (
                "cpu",
                f"{env.get('cpu_model')} ({env.get('cpu_cores_available')} cores available)",
            ),
            ("mem_total", f"{fmt(env.get('mem_total_kb') or 0)} KiB"),
            ("load_avg", env.get("load_avg_1m_5m_15m")),
            ("git", env.get("git")),
        ]
        items.extend(extra)
        return "\n".join(f"- {tag} {k}: `{v}`" for k, v in items if v is not None)

    proto = rounds["py"][0].get("protocol", {})
    engine_note = ""
    if rounds["ws"]:
        wproto = rounds["ws"][0].get("protocol", {})
        engine_note = wproto.get("engine", "")

    node_lines = ""
    if rounds["ws"]:
        env_ws = rounds["ws"][0].get("environment", {})
        extra = [
            ("node", env_ws.get("node")),
            ("v8", env_ws.get("v8")),
        ]
        node_lines = env_block("wasm:", env_ws, extra)
        for key, prefix, _ in LANES[2:]:
            if rounds[key]:
                w = rounds[key][0].get("environment", {}).get("wasm", {})
                if w:
                    node_lines += (
                        f"\n- wasm: flavor=`{w.get('flavor')}` bytes=`{fmt(w.get('bytes'))}` "
                        f"sha256=`{w.get('sha256')}`"
                    )

    mem_lines = []
    mem_path = os.path.join(directory, "mem.json")
    if os.path.exists(mem_path):
        with open(mem_path, encoding="utf-8") as f:
            mem = json.load(f)
        mem_lines.append(
            f"rust `verifier mem` (after_load/after_run/vmhwm): "
            f"{mem.get('after_load_kb'):,} / {mem.get('after_run_kb'):,} / "
            f"{mem.get('vmhwm_kb'):,} KiB"
        )
    py_hwm = [
        r["memory"]["vmhwm_kb"]
        for r in rounds["py"]
        if r.get("memory", {}).get("vmhwm_kb")
    ]
    if py_hwm:
        mem_lines.append(
            f"python `bench_python.py` vmhwm: {min(py_hwm):,}–{max(py_hwm):,} KiB "
            f"across {len(py_hwm)} round(s)"
        )
    if rounds["ws"]:
        node_rss = [
            r["memory"]["after_pass_kb"]
            for r in rounds["ws"]
            if r.get("memory", {}).get("after_pass_kb")
        ]
        if node_rss:
            mem_lines.append(
                f"node wasm-simd rss (after_pass): {min(node_rss):,}–{max(node_rss):,} KiB "
                f"across {len(node_rss)} round(s) — node process incl. engine, not browser-isolated"
            )

    lanes_desc = ", ".join(title.split(" µs/pass")[0] for _, title in present)
    summary = f"""# Bench report — {os.path.basename(directory)}

Lanes: {lanes_desc}. Protocol: warmup {proto.get("warmup", "?")} ×
repeat {proto.get("repeat", "?")} per function group, median µs/pass; the wasm
lanes drive the shipped tltk_web glue under Node with a persistent warm engine
(browser semantics: engine built once per page){f" — `{engine_note}`" if engine_note else ""};
the native lane resets its Session per pass; the python lane restores fresh
colloc-counter state per pass. Models warm in all lanes. Best of
{max(len(v) for v in rounds.values() if v)} interleaved rounds (per-round
values below). `mise run bench-report` reproduces this directory.

{chr(10).join(rows)}

"py→native" is the python/native speedup; "simd→native" is what the browser
wasm runtime adds over native (simd128 flavor). wasm scalar is the fallback
flavor for runtimes without SIMD.

## Recorded environment (from the artifacts)

{env_block("py:", env_py)}

{env_block("rs:", env_rs)}

- py tltk: `{env_py.get("tltk")}` on {env_py.get("impl")} `{env_py.get("python")}`
- py deps: {json.dumps(env_py.get("deps", {}), ensure_ascii=False)}
- rs: `{env_rs.get("rustc")}` (`{env_rs.get("profile")}` profile)

{node_lines}

## Memory

{chr(10).join(f"- {line}" for line in mem_lines) or "- (no mem.json)"}
"""
    out_path = os.path.join(directory, "summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)
    print(f"summary written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
