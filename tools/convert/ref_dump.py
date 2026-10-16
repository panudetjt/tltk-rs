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
import nltk  # noqa: E402
from nltk.parse.dependencygraph import DependencyGraph  # noqa: E402

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
            elif fn == "chunk":
                r = nlp.chunk(text)
            elif fn == "tnc_tag":
                r = nlp.TNC_tag(text)
            elif fn == "textana":
                # tltk 1.11 TextAna (LdvTTR x100); JSON in the Rust wire format:
                # sorted keys, compact separators — see crates/tltk-core/src/textana.rs
                r = json.dumps(nlp.TextAna(text), ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"))
            elif fn == "wrd_len":
                r = nlp.wrd_len(text)
            elif fn == "match_engtext":
                r = str(nlp.match_Engtext(text))
            elif fn == "known":
                r = str(nlp.known(text.split()))
            elif fn == "edits2":
                words = text.split()
                w = words[0] if words else ""
                r = str(nlp.edits2(w))
            elif fn == "check_thaidict":
                r = str(nlp.check_thaidict(text))
            elif fn == "reset_thaidict":
                r = str(nlp.reset_thaidict())
            elif fn == "read_thaidict":
                import tempfile
                with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tf:
                    tf.write(text)
                    tmp_name = tf.name
                try:
                    r = str(nlp.read_thaidict(tmp_name))
                finally:
                    if os.path.exists(tmp_name):
                        os.remove(tmp_name)
            elif fn == "similar_words":
                c = {"low": 0.3, "medium": 0.5, "high": 0.7}[cutoff]
                from tltk import corpus
                corpus.w2v_load()
                r = ", ".join(corpus.similar_words(text.split()[0], cutoff=c))
            elif fn == "outofgroup":
                from tltk import corpus
                corpus.w2v_load()
                words = text.split()
                vocabs = list(corpus.w2v_model.wv.index_to_key)
                if sum(w in vocabs for w in words) < 3:
                    r = "กรุณาใส่คำอย่างน้อย 3 คำ (คำที่มีใน word2vec model)"
                else:
                    r = f"คำที่ไม่เข้าพวก: {corpus.outofgroup(words)}"
            elif fn == "analogy":
                from tltk import corpus
                corpus.w2v_load()
                parts = text.split()
                vocabs = list(corpus.w2v_model.wv.index_to_key)
                if any(w not in vocabs for w in parts):
                    r = "ไม่พบคำ '{}' ใน word2vec model".format(
                        next(w for w in parts if w not in vocabs))
                else:
                    ans = corpus.analogy(parts[0], parts[1], parts[2])
                    if not ans:
                        r = "ไม่พบคำตอบ"
                    else:
                        (word, score), = ans
                        r = f"{parts[0]} : {parts[1]} :: {parts[2]} : {word} (score: {score:.4f})"
            elif fn == "similarity":
                from tltk import corpus
                corpus.w2v_load()
                parts = text.split()[:2]
                vocabs = list(corpus.w2v_model.wv.index_to_key)
                oov = next((w for w in parts if w not in vocabs), None)
                if oov is not None:
                    r = "ไม่พบคำ '{}' ใน word2vec model".format(oov)
                else:
                    r = "ความคล้ายระหว่าง '%s' กับ '%s' = %.4f" % (
                        parts[0], parts[1], corpus.similarity(parts[0], parts[1]))
            elif fn in ("dg_print", "dg_save", "print_dtree", "scan_dtree",
                        "get_tree_depth", "tree_max_depth"):
                # tree helpers consume the dg_save dict; the corpus row text
                # IS that dict (compact json.dumps of dg_save output from a
                # fixed Malt-Tab DependencyGraph). get_tree_depth carries the
                # node id in cutoff (default "0").
                ud = json.loads(text)
                if fn == "get_tree_depth":
                    r = str(nlp.get_tree_depth(ud, int(cutoff) if cutoff != "medium" else 0))
                elif fn == "tree_max_depth":
                    r = str(nlp.tree_max_depth(ud))
                elif fn == "dg_save":
                    r = json.dumps(nlp.dg_save(
                        DependencyGraph() if not ud["words"] else
                        DependencyGraph("\n".join("%s\t%s\t%s\t%s" % (
                            w["form"], w["pos"], w["head"], w["deprel"])
                            for w in ud["words"] if w["id"] > 0) + "\n"),
                        ud["sentence"]), ensure_ascii=False, separators=(",", ":"))
                elif fn == "dg_print":
                    r = str(nlp.dg_print(nltk.DependencyGraph(
                        "\n".join("%s\t%s\t%s\t%s" % (
                            w["form"], w["pos"], w["head"], w["deprel"])
                            for w in ud["words"] if w["id"] > 0) + "\n")))
                else:
                    # upstream scan_dtree needs the module-global PRN_TREE
                    # pre-initialised (print_dtree sets it); render the same
                    # lines print_dtree prints, trailing print newline dropped
                    nlp.PRN_TREE = []
                    (nlp.print_dtree if fn == "print_dtree" else nlp.scan_dtree)(ud)
                    r = "\n".join(f"{a['id']}:{a['msg']}" for a in sorted(nlp.PRN_TREE, key=lambda k: k['id']))
            elif fn == "th2ipa_all":
                r = str(nlp.th2ipa_all(text))
            elif fn == "sylparse_all":
                r = str(nlp.sylparse_all(text))
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
