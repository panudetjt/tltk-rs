#!/usr/bin/env python3
# Benchmark the Python oracle (tltk 1.10) over the parity corpus.
#
# Deps (installed by scripts/setup_segment_oracle.sh): numpy<1.24, scipy<1.13,
# scikit-learn==1.1.1, pandas==1.5.3, gensim==4.3.2, nltk, sklearn-crfsuite,
# requests, matplotlib. Run via `mise run bench-python` — the edu segment model
# (sent_segment.pklz) only unpickles under scikit-learn<1.2, so this must run
# in the era-correct py3.9 venv, not an ad-hoc modern interpreter.
#
# The timing counterpart of `verifier bench` (see mise.toml): same corpus, same
# protocol — per function group, `warmup` untimed passes then `repeat` timed
# passes, each pass = run every corpus case of that function once; report the
# median µs per pass. Between passes the mutable colloc counters (BiCount/Count/
# TotalWord/TotalLex/... in nlp and th2ipa) are restored from a snapshot, which
# is what a fresh Python process would see (the Rust bench makes a fresh Session
# per pass for the same reason). Models (pos/ner/useg/w2v) are loaded once
# before timing, matching the Rust load_models() placement.
#
# The per-function dispatch mirrors tools/convert/ref_dump.py (corpus mode) plus
# the three w2v render cases, following the stichula validator's wire semantics.
# Output shapes are not checked here — parity is the verifier's job; this script
# only makes the interpreter do the same work.
#
# Usage: bench_python.py <corpus.jsonl> [--warmup W] [--repeat N] [--json]
"""
Benchmark the Python oracle (tltk 1.10, era-correct venv) over the corpus.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # before anything imports pyplot

import argparse
import base64
import io
import json
import os
import platform
import socket
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime, timezone
import resource  # POSIX: ru_maxrss = peak RSS

TLTK_PKG_DIR = os.environ.get(
    "TLTK_PKG_DIR", os.path.expanduser("~/.cache/tltk-rs/tltk-1.10-pkg"))
W2V_URL = "https://www.arts.chula.ac.th/~ling/contents/Upload/TNCc5model3.bin"
W2V_CACHE = os.path.expanduser("~/.cache/tltk-bench/TNCc5model3.bin")

# deps whose versions belong in every report (the measured stack, not just tltk)
REPORT_DEPS = ("numpy", "pandas", "scikit-learn", "scipy", "gensim",
               "nltk", "sklearn-crfsuite", "matplotlib", "requests")

# the module-global colloc counters a fresh tltk process starts with
COUNTER_NAMES = ("TriCount", "BiCount", "BiType", "Count", "Type",
                 "TotalWord", "TotalLex")


def log(line):
    print(line, file=sys.stderr, flush=True)


def ensure_w2v_bin(pkg_dir):
    """TNCc5model3.bin does not ship in the wheel; corpus.py loads it from its
    own package dir. Seed that dir from $TLTK_W2V_BIN / the local cache,
    downloading from the official URL (corpus.py download_TNCw2v) as last resort."""
    target = os.path.join(pkg_dir, "TNCc5model3.bin")
    if os.path.exists(target):
        return
    src = os.environ.get("TLTK_W2V_BIN") or (
        W2V_CACHE if os.path.exists(W2V_CACHE) else None)
    if src is None:
        os.makedirs(os.path.dirname(W2V_CACHE), exist_ok=True)
        log(f"==> downloading TNCc5model3.bin (official tltk URL) -> {W2V_CACHE}")
        import urllib.request
        urllib.request.urlretrieve(W2V_URL, W2V_CACHE)
        src = W2V_CACHE
    log(f"==> seeding {target}")
    import shutil
    shutil.copy(src, target)


def load_oracle():
    t0 = time.perf_counter()
    if not os.path.isdir(TLTK_PKG_DIR):
        sys.exit(f"ERROR: tltk 1.10 reference tree not at {TLTK_PKG_DIR} "
                 "(run: mise run oracle-setup)")
    sys.path.insert(0, TLTK_PKG_DIR)
    import tltk
    from tltk import corpus, nlp, th2ipa  # noqa: F401  (all import-time work)
    # model loads that production warm processes carry — same placement as
    # the Rust bench's load_models() (outside the timed region)
    nlp.pos_load()
    nlp.ner_load()
    try:
        nlp.useg_model_load()
    except ValueError as e:
        if "incompatible dtype" in str(e):
            sys.exit("ERROR: sent_segment.pklz needs scikit-learn<1.2 — "
                     "run bench in the oracle venv (mise run oracle-setup)")
        raise
    pkg_dir = os.path.dirname(os.path.abspath(corpus.__file__))
    ensure_w2v_bin(pkg_dir)
    corpus.w2v_load()
    log(f"==> oracle ready: tltk {tltk.__version__ if hasattr(tltk, '__version__') else '1.10'} "
        f"on {platform.python_implementation()} {platform.python_version()} "
        f"(load {time.perf_counter() - t0:.1f}s)")
    return nlp, corpus, th2ipa


class Oracle:
    """Validator-semantics dispatch: same calls ref_dump.py makes, plus the
    three functions its corpus mode skips (outofgroup/analogy/plotwords)."""

    def __init__(self, nlp, corpus):
        self.nlp = nlp
        self.corpus = corpus
        self.oov_msg = "ไม่พบคำ '{}' ใน word2vec model"
        self.cutoffs = {"low": 0.3, "medium": 0.5, "high": 0.7}

    def bench_call(self, fn, text, cutoff):
        try:
            return str(self(fn, text, cutoff))
        except Exception as e:  # upstream crash cases are results too
            return str(e)

    def __call__(self, fn, text, cutoff):
        nlp, corpus = self.nlp, self.corpus
        if fn == "word_segment":
            return nlp.word_segment(text)
        if fn == "syl_segment":
            return nlp.syl_segment(text)
        if fn == "g2p":
            return nlp.g2p(text)
        if fn == "th2ipa":
            return nlp.th2ipa(text)
        if fn == "th2roman":
            return nlp.th2roman(text)
        if fn == "th2read":
            return nlp.th2read(text)
        if fn == "spell_variants":
            return str(nlp.spell_variants(text))
        if fn == "spoonerism":
            return str(nlp.spoonerism(text))
        if fn == "pos_tag":
            return "\n".join(
                " ".join(f"{w}/{p}" for (w, p) in lst) for lst in nlp.pos_tag(text))
        if fn == "ner_tag":
            return nlp.ner_tag(text)
        if fn == "segment":
            return nlp.segment(text)
        if fn == "similar_words":
            w = text.split()[0]
            vocabs = list(corpus.w2v_model.wv.index_to_key)
            if w not in vocabs:
                return self.oov_msg.format(w)
            c = self.cutoffs[cutoff]
            return ", ".join(corpus.similar_words(w, cutoff=c))
        if fn == "similarity":
            parts = text.split()[:2]
            vocabs = list(corpus.w2v_model.wv.index_to_key)
            for w in parts:
                if w not in vocabs:
                    return self.oov_msg.format(w)
            return "ความคล้ายระหว่าง '%s' กับ '%s' = %.4f" % (
                parts[0], parts[1], corpus.similarity(parts[0], parts[1]))
        if fn == "outofgroup":
            words = text.split()
            vocabs = list(corpus.w2v_model.wv.index_to_key)
            if sum(w in vocabs for w in words) < 3:
                return "กรุณาใส่คำอย่างน้อย 3 คำ (คำที่มีใน word2vec model)"
            return f"คำที่ไม่เข้าพวก: {corpus.outofgroup(words)}"
        if fn == "analogy":
            parts = text.split()
            vocabs = list(corpus.w2v_model.wv.index_to_key)
            for w in parts:
                if w not in vocabs:
                    return self.oov_msg.format(w)
            ans = corpus.analogy(parts[0], parts[1], parts[2])
            if not ans:
                return "ไม่พบคำตอบ"
            (word, score), = ans
            return f"{parts[0]} : {parts[1]} :: {parts[2]} : {word} (score: {score:.4f})"
        if fn == "plotwords":
            return self.plotwords(text)
        raise ValueError(f"unknown function {fn}")

    def plotwords(self, text):
        corpus = self.corpus
        vocabs = list(corpus.w2v_model.wv.index_to_key)
        words = [w for w in text.split() if w in vocabs]
        if len(words) < 2:
            return "กรุณาใส่คำอย่างน้อย 2 คำ (คำที่มีใน word2vec model)"
        # the validator's w2v_dimplot chart (100-dim raw-vector lines), rendered
        # server-side; pixel parity with the golden PNGs is out of scope
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # missing TH Sarabun New on dev boxes
            import logging
            logging.getLogger("matplotlib").setLevel(logging.ERROR)
            from matplotlib import pyplot
            corpus.matplotlib.rc(
                "font", **{"family": "TH Sarabun New", "weight": "bold", "size": 14})
            xs = list(range(1, 101))
            fig = pyplot.figure(figsize=(12, 2), dpi=80)
            for word in words:
                pyplot.plot(xs, corpus.w2v_model.wv[word], label=word)
            pyplot.legend()
            buf = io.BytesIO()
            pyplot.savefig(buf, format="png", bbox_inches="tight")
            pyplot.close(fig)
        return "IMAGE:" + base64.b64encode(buf.getvalue()).decode()


def make_snapshot(modules):
    snap = {}
    for mod in modules:
        for name in COUNTER_NAMES:
            if hasattr(mod, name):
                snap[(mod.__name__, name)] = deepcopy(getattr(mod, name))
    return snap


def restore_counters(modules, snap):
    for mod in modules:
        for name in COUNTER_NAMES:
            key = (mod.__name__, name)
            if key in snap:
                setattr(mod, name, deepcopy(snap[key]))


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n % 2 == 1:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) // 2


def cpu_model():
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def mem_total_kb():
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def read_proc_status():
    """VmRSS/VmHWM from /proc/self/status — same fields the Rust verifier records."""
    try:
        rss = hwm = None
        for line in open("/proc/self/status", encoding="utf-8"):
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1])
        return {"rss_kb": rss, "vmhwm_kb": hwm}
    except OSError:
        return None


def git_state():
    """Short HEAD sha + dirty-file count, mirroring the Rust env block."""
    import subprocess
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        if not sha:
            return None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10).stdout.count("\n")
        return f"{sha}+{dirty} dirty files" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return None


def environment_block(tltk_version):
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 0
    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        load = None
    import importlib.metadata
    deps = {}
    for name in REPORT_DEPS:
        try:
            deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            deps[name] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
        "os": sys.platform,
        "platform_detail": platform.platform(),
        "kernel": platform.release(),
        "cpu_model": cpu_model(),
        "cpu_cores_available": cores,
        "mem_total_kb": mem_total_kb(),
        "load_avg_1m_5m_15m": load,
        "git": git_state(),
        "python": platform.python_version(),
        "impl": platform.python_implementation(),
        "tltk": tltk_version,
        "deps": deps,
    }


def tltk_version():
    import importlib.metadata
    try:
        return importlib.metadata.version("tltk")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cases = []
    for line in open(args.corpus, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        cases.append((row["function"], row["text"], row.get("cutoff", "medium")))

    env = environment_block(None)
    log(f"# env: {json.dumps(env, ensure_ascii=False)}")
    nlp, corpus, th2ipa = load_oracle()
    env["tltk"] = tltk_version()
    proc_after_load = read_proc_status()
    oracle = Oracle(nlp, corpus)
    modules = [nlp, th2ipa]

    fn_order, groups = [], {}
    for fn, text, cutoff in cases:
        if fn not in groups:
            fn_order.append(fn)
            groups[fn] = []
        groups[fn].append((text, cutoff))

    # smoke each function once; per-case exceptions are data here — tltk 1.10
    # has disclosed crash cases (e.g. spoonerism('สวัสดี') -> KeyError 'dat',
    # golden ok=false) that the validator itself 500s on
    smoke_errors = 0
    for fn in fn_order:
        for text, cutoff in groups[fn]:
            try:
                oracle(fn, text, cutoff)
            except Exception as e:
                smoke_errors += 1
                log(f"!! oracle raises (as upstream does): {fn} {text[:20]!r}: {str(e)[:60]}")
    if smoke_errors:
        log(f"({smoke_errors} case(s) raise; timed passes will count them via the same wrapper)")

    snap = make_snapshot(modules)
    sink = 0  # keep results alive the way black_box does for the Rust bench
    fn_results = {}
    for fn in fn_order:
        group = groups[fn]
        for _ in range(args.warmup):
            restore_counters(modules, snap)
            for text, cutoff in group:
                sink += len(oracle.bench_call(fn, text, cutoff))
        pass_us = []
        for _ in range(args.repeat):
            restore_counters(modules, snap)
            t0 = time.perf_counter_ns()
            for text, cutoff in group:
                sink += len(oracle.bench_call(fn, text, cutoff))
            pass_us.append((time.perf_counter_ns() - t0) // 1000)
        fn_results[fn] = {"cases": len(group), "median_us_per_pass": median(pass_us)}
        log(f"{fn:<19}cases={len(group):<2}  median_us_per_pass={fn_results[fn]['median_us_per_pass']}")

    total = sum(r["median_us_per_pass"] for r in fn_results.values())
    proc_after_pass = read_proc_status()
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KiB
    log(f"{'total':<19}cases={len(cases):<2}  median_us_per_pass={total}")
    log(f"peak_rss_kb={peak_rss_kb} sink={sink}")

    if args.json:
        print(json.dumps({
            "protocol": {"warmup": args.warmup, "repeat": args.repeat,
                         "statistic": "median_us_per_pass"},
            "environment": env,
            "functions": fn_results,
            "total_us_per_pass": total,
            "memory": {
                "after_load_kb": proc_after_load["rss_kb"] if proc_after_load else None,
                "after_pass_kb": proc_after_pass["rss_kb"] if proc_after_pass else None,
                "vmhwm_kb": peak_rss_kb,
            },
        }))
    else:
        print(json.dumps(fn_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
