#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"  # scipy<1.13 has no cp313 wheels
# dependencies = [
#   "numpy",
#   "scikit-learn",
#   "nltk",
#   "gensim>=4.3.3",
#   "scipy<1.13",        # gensim 4.3.x imports scipy.linalg.triu (removed in 1.13)
#   "python-crfsuite",
# ]
# ///
# Self-contained: `uv run tools/convert/convert.py <tltk-dir> <out-dir>`.
"""Convert tltk (Python) model files into compact binary formats for tltk-core.

Binary format conventions (all little-endian):
  - uvarint: LEB128 unsigned
  - str:     uvarint byte-length + UTF-8 bytes
  - table:   uvarint count, then entries SORTED by key (binary-searchable)

Files produced (into crates/tltk-core/data/):
  thdict.bin.gz   TD01  - dictionary word set
  sylseg.bin.gz   SG01  - syllable trigram counts + derived BiCount/Count/BiType/Type
  sylvar.bin.gz   SV01  - phone -> spelling variants
  phs.bin.gz      PH01  - phone/form n-gram tables derived from PhSTrigram.sts
  pos.bin.gz      PT01  - nltk PerceptronTagger weights/tagdict/classes
  wordtier.bin.gz WT01  - word.pklz lexicon fields read by TextAna (10 fields/word)
  edu.bin.gz      ED01  - DictVectorizer vocab + RandomForest trees (sent_segment.pklz)
  w2v.bin.gz      WV01  - gensim KeyedVectors words + f32 matrix (TNCc5model3.bin)
  textclass.bin.gz TC01 - level and genre RandomForest models (chklevel_rf.pklz, chkgenre_rf.pklz)

Requires: venv with numpy, scikit-learn, nltk, gensim, python-crfsuite.
Usage: convert.py <tltk-package-dir> <out-dir> [--w2v <path>] [--ner-model <crfsuite-file>]
"""
import bz2
import gzip
import io
import os
import pickle
import struct
import sys
import warnings

warnings.filterwarnings("ignore")


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


def table_str(entries) -> bytes:
    """entries: iterable of str -> sorted unique table."""
    uniq = sorted(set(entries))
    out = uvarint(len(uniq))
    for s in uniq:
        out += wstr(s)
    return out


def rows(entries) -> bytes:
    """entries: iterable of (key_tuple, value_bytes-writer-friendly). Caller sorts."""
    out = uvarint(len(entries))
    for e in entries:
        out += e
    return out


def f64(x) -> bytes:
    return struct.pack("<d", float(x))


def f32(x) -> bytes:
    return struct.pack("<f", float(x))


def gz(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(gzip.compress(data, compresslevel=6, mtime=0))
    print(f"  {os.path.basename(path):18s} raw={len(data)/1e6:8.3f}MB  gz={os.path.getsize(path)/1e6:8.3f}MB")


# ---------------------------------------------------------------- thdict
def conv_thdict(tltk_dir: str, out_dir: str) -> None:
    with open(os.path.join(tltk_dir, "thdict"), "rb") as f:
        td = pickle.load(f)
    data = b"TD01" + table_str(td.keys())
    gz(os.path.join(out_dir, "thdict.bin.gz"), data)


# ---------------------------------------------------------------- sylseg
def conv_sylseg(tltk_dir: str, out_dir: str) -> None:
    with open(os.path.join(tltk_dir, "sylseg.3g"), "rb") as f:
        tri = pickle.load(f)
    # read_stat derivations (matches nlp.read_stat exactly)
    bi_count = {}
    count = {}
    bi_type = {}
    type_ = {}
    for (x, y, z), c in tri.items():
        bi_type[(x, y)] = bi_type.get((x, y), 0) + 1  # distinct Z per (X,Y)
        bi_count[(x, y)] = bi_count.get((x, y), 0) + c
        count[y] = count.get(y, 0) + c
    for (x, y) in bi_count:
        type_[x] = type_.get(x, 0) + 1
    total_lex = len(count)
    total_word = sum(count.values())

    syls = []
    for (x, y, z) in tri:
        syls += [x, y, z]
    stbl = sorted(set(syls))
    sidx = {s: i for i, s in enumerate(stbl)}

    out = bytearray(b"SG01")
    out += uvarint(len(stbl))
    for s in stbl:
        out += wstr(s)
    tr = sorted((sidx[x], sidx[y], sidx[z], c) for (x, y, z), c in tri.items())
    out += uvarint(len(tr))
    for x, y, z, c in tr:
        out += uvarint(x) + uvarint(y) + uvarint(z) + uvarint(c)
    # derived maps as sorted (idx-pair, val) tables
    bc = sorted((sidx[x], sidx[y], c) for (x, y), c in bi_count.items())
    out += uvarint(len(bc))
    for x, y, c in bc:
        out += uvarint(x) + uvarint(y) + uvarint(c)
    bt = sorted((sidx[x], sidx[y], c) for (x, y), c in bi_type.items())
    out += uvarint(len(bt))
    for x, y, c in bt:
        out += uvarint(x) + uvarint(y) + uvarint(c)
    cn = sorted((sidx[x], c) for x, c in count.items())
    out += uvarint(len(cn))
    for x, c in cn:
        out += uvarint(x) + uvarint(c)
    ty = sorted((sidx[x], c) for x, c in type_.items())
    out += uvarint(len(ty))
    for x, c in ty:
        out += uvarint(x) + uvarint(c)
    out += uvarint(total_word) + uvarint(total_lex)
    gz(os.path.join(out_dir, "sylseg.bin.gz"), bytes(out))


# ---------------------------------------------------------------- sylvar
def conv_sylvar(tltk_dir: str, out_dir: str) -> None:
    sv = pickle.load(bz2.open(os.path.join(tltk_dir, "sylform_var.pklz"), "rb"))
    out = bytearray(b"SV01")
    items = sorted(sv.items())
    out += uvarint(len(items))
    for phone, forms in items:
        out += wstr(phone)
        out += uvarint(len(forms))
        for w in forms:
            out += wstr(w)
    gz(os.path.join(out_dir, "sylvar.bin.gz"), bytes(out))


# ---------------------------------------------------------------- phs (PhSTrigram.sts derived)
def conv_phs(tltk_dir: str, out_dir: str) -> None:
    import re as _re

    phs_tri = {}
    frm_tri = {}
    phs_bi = {}
    frm_bi = {}
    phs_uni = {}
    frm_uni = {}
    abs_uni = {}
    abs_frm = {}
    with open(os.path.join(tltk_dir, "PhSTrigram.sts"), "r", encoding="cp874") as f:
        for line in f.readlines():
            line = line.rstrip()
            line = line.replace("<w>", "|")
            x, ct = line.split("\t")
            fs, p = x.split("/")
            x1, x2, x3 = fs.split("-")
            ct = float(ct)
            phs_tri[(x1, x2, x3, p)] = phs_tri.get((x1, x2, x3, p), 0.0) + ct
            frm_tri[(x1, x2, x3)] = frm_tri.get((x1, x2, x3), 0.0) + ct
            phs_bi[(x1, x2, p)] = phs_bi.get((x1, x2, p), 0.0) + ct
            frm_bi[(x1, x2)] = frm_bi.get((x1, x2), 0.0) + ct
            phs_bi[(x2, x3, p)] = phs_bi.get((x2, x3, p), 0.0) + ct
            frm_bi[(x2, x3)] = frm_bi.get((x2, x3), 0.0) + ct
            phs_uni[(x2, p)] = phs_uni.get((x2, p), 0.0) + ct
            frm_uni[(x2,)] = frm_uni.get((x2,), 0.0) + ct
            abs_x2 = _re.sub(r"[่้๊๋]", "", x2)
            abs_x2 = _re.sub(r"[ก-ฮ]", "C", abs_x2)
            abs_p = _re.sub(r"[0-9]", "", p)
            abs_p = _re.sub(r"[^aeio@OuxU]", "C", abs_p)
            abs_uni[(abs_x2, abs_p)] = abs_uni.get((abs_x2, abs_p), 0.0) + ct
            abs_frm[(abs_x2,)] = abs_frm.get((abs_x2,), 0.0) + ct

    vocab = set()
    for k in phs_tri:
        vocab.update(k)
    for k in frm_tri:
        vocab.update(k)
    for k in phs_bi:
        vocab.update(k)
    for k in frm_bi:
        vocab.update(k)
    for k in phs_uni:
        vocab.update(k)
    for k in frm_uni:
        vocab.update(k)
    for k in abs_uni:
        vocab.update(k)
    for k in abs_frm:
        vocab.update(k)
    stbl = sorted(vocab)
    sidx = {s: i for i, s in enumerate(stbl)}

    out = bytearray(b"PH01")
    out += uvarint(len(stbl))
    for s in stbl:
        out += wstr(s)

    def emit(tbl, arity):
        rr = sorted((tuple(sidx[x] for x in k), v) for k, v in tbl.items())
        out.extend(uvarint(len(rr)))
        for k, v in rr:
            for i in range(arity):
                out.extend(uvarint(k[i]))
            out.extend(f64(v))

    emit(phs_tri, 4)
    emit(frm_tri, 3)
    emit(phs_bi, 3)
    emit(frm_bi, 2)
    emit(phs_uni, 2)
    emit(frm_uni, 1)
    emit(abs_uni, 2)
    emit(abs_frm, 1)
    gz(os.path.join(out_dir, "phs.bin.gz"), bytes(out))


# ---------------------------------------------------------------- pos (nltk PerceptronTagger)
def conv_pos(tltk_dir: str, out_dir: str) -> None:
    tagger = pickle.load(bz2.open(os.path.join(tltk_dir, "pos_tagger.pklz"), "rb"))
    ap = tagger.model
    wrows = []
    for feat, weights in ap.weights.items():
        for tag, w in weights.items():
            wrows.append((feat, tag, float(w)))
    wrows.sort(key=lambda r: (r[0], r[1]))
    out = bytearray(b"PT01")
    # classes: sorted set
    classes = sorted(tagger.classes)
    out += uvarint(len(classes))
    for c in classes:
        out += wstr(c)
    # tagdict
    td = sorted(tagger.tagdict.items())
    out += uvarint(len(td))
    for w, tv in td:
        tag = tv[0] if isinstance(tv, (tuple, list)) else tv
        out += wstr(w) + wstr(tag)
    # weights (f64 to match Python float64 exactly)
    out += uvarint(len(wrows))
    for feat, tag, w in wrows:
        out += wstr(feat) + wstr(tag) + f64(w)
    gz(os.path.join(out_dir, "pos.bin.gz"), bytes(out))


# ---------------------------------------------------------------- edu (DictVectorizer + RandomForest)
def conv_edu(tltk_dir: str, out_dir: str) -> None:
    class StubTree:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            self.state = state

    class Unp(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "sklearn.tree._tree" and name in ("Tree", "DenseMatrix"):
                return StubTree
            return super().find_class(module, name)

    pipe = Unp(bz2.open(os.path.join(tltk_dir, "sent_segment.pklz"), "rb")).load()
    dv = pipe.steps[0][1]
    rf = pipe.steps[1][1]
    vocab = sorted(dv.vocabulary_.items(), key=lambda kv: kv[1])  # by index
    out = bytearray(b"ED01")
    out += uvarint(len(vocab))
    for name, idx in vocab:
        out += wstr(name) + uvarint(idx)
    out += uvarint(len(rf.classes_))
    for c in rf.classes_:
        out += wstr(str(c))
    trees = rf.estimators_
    out += uvarint(len(trees))
    for est in trees:
        st = est.tree_.state
        nodes = st["nodes"]
        values = st["values"]  # (n_nodes, n_outputs, n_classes)
        n = nodes.shape[0]
        out += uvarint(n)
        for i in range(n):
            nd = nodes[i]
            out += struct.pack("<iii", int(nd["left_child"]), int(nd["right_child"]), int(nd["feature"]))
            out += f64(nd["threshold"])
            v = values[i]
            out += uvarint(v.shape[1] if v.ndim == 2 else 1)
            for j in range(v.shape[1] if v.ndim == 2 else 1):
                out += f64(float(v[0][j]))
    gz(os.path.join(out_dir, "edu.bin.gz"), bytes(out))


# ---------------------------------------------------------------- textclass (level + genre RF)
def conv_textclass(tltk_dir: str, out_dir: str) -> None:
    class StubTree:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            self.state = state

    class Unp(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "sklearn.tree._tree" and name in ("Tree", "DenseMatrix"):
                return StubTree
            return super().find_class(module, name)

    def serialize_rf(rf):
        feats = list(rf.feature_names_in_)
        out = bytearray(uvarint(len(feats)))
        for f in feats:
            out += wstr(f)
        classes = list(rf.classes_)
        out += uvarint(len(classes))
        for c in classes:
            out += wstr(str(c))
        trees = rf.estimators_
        out += uvarint(len(trees))
        for est in trees:
            st = est.tree_.state
            nodes = st["nodes"]
            values = st["values"]  # shape (n_nodes, 1, n_classes)
            n = nodes.shape[0]
            out += uvarint(n)
            for i in range(n):
                nd = nodes[i]
                left = int(nd["left_child"])
                right = int(nd["right_child"])
                feat = int(nd["feature"])
                thresh = float(nd["threshold"])
                out += struct.pack("<iii", left, right, feat)
                out += f64(thresh)
                if left == -1:
                    for v in values[i][0]:
                        out += f64(float(v))
        return bytes(out)

    lvl_p = os.path.join(tltk_dir, "chklevel_rf.pklz")
    gnr_p = os.path.join(tltk_dir, "chkgenre_rf.pklz")
    if not (os.path.exists(lvl_p) and os.path.exists(gnr_p)):
        print("  (chklevel_rf.pklz or chkgenre_rf.pklz missing: skipping textclass)")
        return

    level_rf = Unp(bz2.open(lvl_p, "rb")).load()
    genre_rf = Unp(bz2.open(gnr_p, "rb")).load()

    out = bytearray(b"TC01")
    out += serialize_rf(level_rf)
    out += serialize_rf(genre_rf)
    gz(os.path.join(out_dir, "textclass.bin.gz"), bytes(out))

# ---------------------------------------------------------------- w2v (gensim KeyedVectors)
def conv_w2v(model_path: str, out_dir: str) -> None:
    import numpy as np
    from gensim.models import Word2Vec

    m = Word2Vec.load(model_path)
    wv = m.wv
    words = list(wv.index_to_key)
    vecs = np.asarray(wv.vectors, dtype=np.float32)
    out = bytearray(b"WV01")
    out += uvarint(len(words)) + uvarint(vecs.shape[1])
    for w in words:
        out += wstr(w)
    out += vecs.tobytes()
    gz(os.path.join(out_dir, "w2v.bin.gz"), bytes(out))


# ---------------------------------------------------------------- ner (crfsuite model -> weights)
def conv_ner(model_path: str, out_dir: str) -> None:
    import re as _re
    import pycrfsuite

    t = pycrfsuite.Tagger()
    t.open(model_path)
    import tempfile as _tempfile
    with _tempfile.NamedTemporaryFile("w+", suffix=".dump", delete=False) as tf:
        t.dump(tf.name)
        text = open(tf.name, encoding="utf-8", errors="replace").read()
    # attribute values containing control chars break dump lines: rejoin the
    # " --> label: weight" continuations
    text = text.replace("\n --> ", " --> ")

    labels = []
    transitions = []
    attributes = []
    section = None
    line_re = _re.compile(r"^\s*\(\d+\)\s+(.*?)\s+-->\s+(.*?):\s*(-?[\d.eE+]+)\s*$")
    label_re = _re.compile(r"^\s+\d+:\s*(.+?)\s*$")
    for line in text.splitlines():
        if line.startswith("LABELS"):
            section = "L"
            continue
        if line.startswith("TRANSITIONS"):
            section = "T"
            continue
        if line.startswith("STATE_FEATURES"):
            section = "S"
            continue
        if line.startswith("ATTRIBUTES"):
            section = "A"
            continue
        if line.rstrip().endswith("}"):
            section = None
            continue
        if section == "L":
            m = label_re.match(line)
            if m:
                labels.append(m.group(1).strip())
        elif section == "A":
            pass  # attribute id -> name; names appear again in STATE_FEATURES
        elif section in ("T", "S"):
            m = line_re.match(line)
            if m:
                (transitions if section == "T" else attributes).append(
                    (m.group(1), m.group(2), float(m.group(3)))
                )
            elif line.strip():
                print(f"  ner dump unparsed: {line!r}")
    print(
        f"  ner parsed: labels={len(labels)} transitions={len(transitions)} attributes={len(attributes)}"
    )
    if not labels or not attributes:
        raise SystemExit(
            f"ner dump parse failed: labels={len(labels)} transitions={len(transitions)} attributes={len(attributes)}"
        )
    transitions.sort()
    attributes.sort()
    out = bytearray(b"NR01")
    out += uvarint(len(labels))
    for lb in labels:
        out += wstr(lb)
    out += uvarint(len(transitions))
    for a, b, w in transitions:
        out += wstr(a) + wstr(b) + f64(w)
    out += uvarint(len(attributes))
    for a, b, w in attributes:
        out += wstr(a) + wstr(b) + f64(w)
    gz(os.path.join(out_dir, "ner.bin.gz"), bytes(out))
    print(f"  ner labels={labels} transitions={len(transitions)} attributes={len(attributes)}")


# ---------------------------------------------------------------- wordtier (word.pklz -> TextAna lexicon)
# The 10 WRD fields TextAna reads. Same file in tltk 1.10 and 1.11.
# Unpickling needs numpy: run via `uv run` (reads the PEP 723 block) or the
# era venv — plain python3 dies with ModuleNotFoundError mid-unpickle.
WORDTIER_STR_FIELDS = ("tier", "level", "cont", "seman", "form", "lvocab", "bvocab")
WORDTIER_F64_FIELDS = ("tnc-tfidf", "tnc-disp", "ttc-disp")


def conv_wordtier(tltk_dir: str, out_dir: str) -> None:
    wrd = pickle.load(bz2.open(os.path.join(tltk_dir, "word.pklz"), "rb"))
    out = bytearray(b"WT01")
    words = sorted(wrd)
    out += uvarint(len(words))
    for word in words:
        info = wrd[word]
        vals = [(k, info[k]) for k in WORDTIER_STR_FIELDS if k in info]
        vals += [(k, float(info[k])) for k in WORDTIER_F64_FIELDS if k in info]
        flags = 0
        for k, _ in vals:
            name, _kind = _WORDTIER_INDEX[k]
            flags |= 1 << name
        out += wstr(word) + uvarint(flags)
        for _k, v in vals:
            out += wstr(v) if isinstance(v, str) else f64(v)
    gz(os.path.join(out_dir, "wordtier.bin.gz"), bytes(out))
    print(f"  wordtier.bin.gz    {len(words)} words")


# field name -> (bit index, kind); order matters in the value stream: strings then f64s
_WORDTIER_INDEX = {k: (i, "s") for i, k in enumerate(WORDTIER_STR_FIELDS)}
_WORDTIER_INDEX.update({k: (len(WORDTIER_STR_FIELDS) + i, "f") for i, k in enumerate(WORDTIER_F64_FIELDS)})


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    tltk_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)
    w2v_path = os.path.join(tltk_dir, "TNCc5model3.bin")
    ner_model = None
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--w2v":
            w2v_path = args[i + 1]
            i += 2
        elif args[i] == "--ner-model":
            ner_model = args[i + 1]
            i += 2
        else:
            i += 1

    print(f"converting from {tltk_dir} -> {out_dir}")
    # text pattern files: cp874 -> UTF-8
    for src, dst in (("sylrule.lts", "sylrule.txt"), ("thaisyl.dict", "thaisyl.txt")):
        with open(os.path.join(tltk_dir, src), "r", encoding="cp874") as f:
            txt = f.read()
        with open(os.path.join(out_dir, dst), "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"  {dst:18s} {len(txt)} chars")
    conv_thdict(tltk_dir, out_dir)
    conv_sylseg(tltk_dir, out_dir)
    conv_sylvar(tltk_dir, out_dir)
    conv_phs(tltk_dir, out_dir)
    conv_pos(tltk_dir, out_dir)
    conv_wordtier(tltk_dir, out_dir)
    conv_textclass(tltk_dir, out_dir)
    conv_edu(tltk_dir, out_dir)
    conv_w2v(w2v_path, out_dir)
    if ner_model:
        conv_ner(ner_model, out_dir)
    else:
        print("  (no --ner-model: NER CRF weights not converted; artifact lost upstream)")
    print("done.")


if __name__ == "__main__":
    main()
