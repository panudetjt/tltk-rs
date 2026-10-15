#!/usr/bin/env python3
"""Local oracle: run the real tltk 1.10 (wheel) and dump internals + outputs.

Usage: ref_dump.py <tltk-pkg-dir> <mode>
  modes:
    probe            - print PRON info + sylseg internals for sample inputs
    corpus <jsonl>   - run all functions over corpus inputs, emit golden JSONL
"""
import json
import sys
import os

sys.path.insert(0, sys.argv[1])
import tltk  # noqa: E402
from tltk import nlp  # noqa: E402

mode = sys.argv[2]

if mode == "probe":
    print("PRON patterns:", len(nlp.PRON))
    items = list(nlp.PRON.items())
    for p, codes in items[:5]:
        print("  ", repr(p), "->", codes[:3])
    print("TDICT:", len(nlp.TDICT))
    print("TriCount:", len(nlp.TriCount))
    print("TotalWord/TotalLex:", nlp.TotalWord, nlp.TotalLex)
    inp = "แมวกินปลา"
    # replicate sylseg with internals
    from collections import defaultdict
    from copy import deepcopy
    schart = defaultdict(dict)
    probEnd = defaultdict(float)
    EndOfInput = len(inp)
    for f in nlp.PRON:
        for i in range(EndOfInput):
            Inx = inp[i:]
            m = nlp.re.match(f, Inx)
            if m:
                k = i + len(m.group())
                schart[i][k] = [m.group()]
                probEnd[(i, k)] = nlp.prob_trisyl([m.group()])
    print("edges:")
    for i in sorted(schart):
        for k in sorted(schart[i]):
            print(f"  ({i},{k}) {schart[i][k]} prob={probEnd[(i,k)]:.6f}")
    out = nlp.sylseg(inp)
    print("sylseg:", out)
    # word_segment internals: colloc strengths
    print("word_segment:", nlp.word_segment(inp))
    print("g2p:", nlp.g2p(inp))
elif mode == "corpus":
    out = sys.stdout
    for line in open(sys.argv[3], encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        fn, text = row["function"], row["text"]
        cutoff = row.get("cutoff", "medium")
        try:
            if fn == "word_segment":
                r = nlp.word_segment(text)
            elif fn == "syl_segment":
                r = nlp.syl_segment(text)
            elif fn == "g2p":
                r = nlp.g2p(text)
            elif fn == "th2ipa":
                r = nlp.th2ipa(text)
            elif fn == "th2roman":
                r = nlp.th2roman(text)
            elif fn == "th2read":
                r = nlp.th2read(text)
            elif fn == "spell_variants":
                r = str(nlp.spell_variants(text))
            elif fn == "spoonerism":
                r = str(nlp.spoonerism(text))
            elif fn == "pos_tag":
                r = "\n".join(
                    " ".join(f"{w}/{p}" for (w, p) in lst) for lst in nlp.pos_tag(text)
                )
            elif fn == "ner_tag":
                r = nlp.ner_tag(text)
            elif fn == "segment":
                r = nlp.segment(text)
            elif fn == "similar_words":
                c = {"low": 0.3, "medium": 0.5, "high": 0.7}[cutoff]
                from tltk import corpus
                corpus.w2v_load()
                r = ", ".join(corpus.similar_words(text.split()[0], cutoff=c))
            elif fn == "similarity":
                from tltk import corpus
                corpus.w2v_load()
                w1, w2 = text.split()[:2]
                r = "ความคล้ายระหว่าง '%s' กับ '%s' = %.4f" % (w1, w2, corpus.similarity(w1, w2))
            else:
                continue
            ok = True
        except Exception as e:
            r = str(e)
            ok = False
        out.write(json.dumps({
            "function": fn, "text": text, "cutoff": cutoff,
            "expected": r, "ok": ok,
        }, ensure_ascii=False) + "\n")
else:
    print("unknown mode", mode)
    sys.exit(2)
