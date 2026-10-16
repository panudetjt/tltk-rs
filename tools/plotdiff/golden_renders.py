#!/usr/bin/env python3
"""Dump oracle (tltk 1.11) PNG renders for the three plot functions.

Usage: golden_renders.py <tltk-pkg-dir>  (writes JSONL to stdout; redirect)
using the real matplotlib renderers (w2v_diffplot / w2v_dimplot /
w2v_compare_color semantics from corpus.py, rerouted to a PNG buffer).
"""
import sys
import io
import json
import base64
import warnings

sys.path.insert(0, sys.argv[1])
from tltk import corpus  # noqa: E402

corpus.w2v_load()

from matplotlib import pyplot  # noqa: E402


def render(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    pyplot.close(fig)
    return "IMAGE:" + base64.b64encode(buf.getvalue()).decode()


INPUTS = [
    ("w2v_diffplot", "แมว ปลา นก"),
    ("w2v_diffplot", "กิน นอน เดิน"),
    ("w2v_dimplot", "แมว ปลา"),
    ("w2v_dimplot", "กิน นอน เดิน วิ่ง"),
    ("w2v_compare_color", "แมว ปลา นก"),
    ("w2v_compare_color", "กิน นอน เดิน"),
]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import logging
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    corpus.matplotlib.rc(
        "font", **{"family": "TH Sarabun New", "weight": "bold", "size": 14})
    for fn, text in INPUTS:
        parts = text.split()
        if fn == "w2v_diffplot":
            ww, wx, wy = parts
            xs = list(range(1, 101))
            fig = pyplot.figure()
            pyplot.plot(xs, [a - b for a, b in zip(corpus.w2v_model.wv[ww], corpus.w2v_model.wv[wx])], label=ww + "-" + wx)
            pyplot.plot(xs, [a - b for a, b in zip(corpus.w2v_model.wv[ww], corpus.w2v_model.wv[wy])], label=ww + "-" + wy)
            pyplot.legend()
            r = render(fig)
        elif fn == "w2v_dimplot":
            vocabs = list(corpus.w2v_model.wv.index_to_key)
            words = [w for w in parts if w in vocabs]
            xs = list(range(1, 101))
            fig = pyplot.figure()
            for word in words:
                pyplot.plot(xs, corpus.w2v_model.wv[word], label=word)
            pyplot.legend()
            r = render(fig)
        else:
            vocabs = list(corpus.w2v_model.wv.index_to_key)
            words = [w for w in parts if w in vocabs]
            wdsr = words[:]
            wdsr.reverse()
            vs = [corpus.w2v_model.wv[wd] for wd in words]
            dim = len(vs[0])
            fig = pyplot.figure(num=None, figsize=(12, 2), dpi=80, facecolor="w", edgecolor="k")
            ax = fig.add_subplot(111)
            ax.set_facecolor("gray")
            for i, v in enumerate(vs):
                ax.scatter(range(dim), [i] * dim, c=vs[i], cmap="Spectral", s=16)
            pyplot.xlabel("Dimension")
            pyplot.yticks(range(len(words)), words)
            r = render(fig)
        sys.stdout.write(json.dumps(
            {"function": fn, "text": text, "cutoff": "medium",
             "expected": r, "ok": True}, ensure_ascii=False) + "\n")
