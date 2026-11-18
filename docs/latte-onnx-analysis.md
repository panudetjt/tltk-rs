# LATTE-WordSeg-Lite: forward-pass research note

Research for a pure-Rust reimplementation of `iapp/OpenThai2.0-LATTE-WordSeg-Lite`
(no ONNX Runtime). **Read-only research**: no source files were changed. All
artifacts were downloaded with `urllib` (header `User-Agent: Mozilla/5.0`) into
`/tmp/opencode/latte/` and inspected with `onnx` 1.23.2 and `onnxruntime` 1.32.0.

## 0. Provenance and verification method

| Artifact | URL | sha256 |
|---|---|---|
| wheel 0.1.2 | `https://huggingface.co/iapp/OpenThai2.0-LATTE-WordSeg-Lite/resolve/main/openthai2_0_latte_wordseg_lite-0.1.2-py3-none-any.whl` | `39d4162470d2acfb1f375e42847f920d49ee368b656cfe04cef42953731aa926` (3,471,506 B) |
| model | `https://huggingface.co/iapp/OpenThai2.0-LATTE-WordSeg-Lite/resolve/main/model.int8.onnx` | `2f8d349528b3d8342f5cb9150a4e07e774f07e40e892367dbe18bd41ea1d29f3` (2,978,130 B) — **byte-identical** to the wheel copy |
| charclass | wheel `assets/charclass.npz` | `078ea02fd7db27f56c5cf4bb785087785ab4929d4473f1fbd5ddfbc6ec8b41d6` (3,752 B) |
| lexicon | wheel `assets/lexicon.txt.gz` | `dc611971a797e105620ac2791311e8d9c1c5f0c2944527786a712ed5e2571d5a` (844,547 B) |

The manifest's declared model sha256 (`2f8d34…d29f3`) matches. Graph metadata
`config`: `{"emb_dim":64,"use_lexicon":true,"lex_dim":32,"channels":128,
"widths":[3,5,7],"hidden":128,"layers":2,"dropout":0.2,"styles":[],
"style_dim":16,"levels":["large","word","sub"]}`; `vocab` has **944 entries**
(so `emb` has 13+944 = 957 rows). `manifest.json` maps model levels
`["large","word","sub"]` to public units `["long","short","minimal"]`,
`default_unit: "short"`, `unnested_units: ["minimal"]`.

The dynamic-quantization behaviour below (§6) was not merely read from docs: it
was **probed empirically** with synthetic single-node `com.microsoft.DynamicQuantizeLSTM`
and whole-model ORT sessions, inverting controlled gate outputs to read off the
exact scales/zero-points used. Probe scripts are in `/tmp/opencode/latte/probe_lstm{,2,3,4}.py`.

---

## 1. Wheel contents / inference-time assets

| File | Bytes | Needed at inference? |
|---|---|---|
| `openthai_latte_wordseg_lite/__init__.py` | 1,858 | no (API sugar) |
| `openthai_latte_wordseg_lite/__main__.py` | 2,750 | no (CLI only) |
| `openthai_latte_wordseg_lite/_runtime.py` | 23,027 | **yes** — the complete preprocessor/decoder (this is the spec) |
| `assets/charclass.npz` | 3,752 | **yes** — code point → class LUT |
| `assets/lexicon.txt.gz` | 844,547 | **yes** — 177,385 words, 2–16 chars |
| `assets/manifest.json` | 943 | **yes** — asset names, unit mapping, metadata |
| `assets/model.int8.onnx` | 2,978,130 | **yes** — the whole network |
| `*.dist-info/{METADATA,WHEEL,entry_points.txt,LICENSES/*,NOTICE,RECORD}` | ~38 KB | no |

Lexicon word-length histogram: 2:2006, 3:7872, 4:14023, 5:17626, 6:22040,
7:21354, 8:19053, 9:16278, 10:13952, 11:11340, 12:9238, 13:7658, 14:6102,
15:4883, 16:3960. `manifest.json` says `lexicon_words: 177385`,
`lexicon_dropped: 12652` (words outside 2–16 chars or with whitespace).

**Parameter count.** `manifest.json` declares `parameters: 1168771`. The sum of
all ONNX initializer elements is **1,430,493**; the difference is essentially the
8192×32 = 262,144-entry `lex_table` (1,430,493 − 262,144 = 1,168,349, close to
the declared figure), i.e. the manifest appears to exclude the lexicon embedding.
Quote the manifest's 1.17M for the model card; use 1,430,493 when budgeting bytes.

---

## 2. Python runtime — exact algorithm (verbatim from `_runtime.py`)

### 2.1 Constants and code units

```python
N_CLASSES = 13                       # ids 0-12: padding + Unicode-class buckets for unknown characters
N_LEX = 13                           # 6 length buckets x (starts, ends) + inside a word
LEN_BUCKET = np.array([0, 0, 0, 1, 2, 3, 3, 4, 4, 4] + [5] * 200, np.int64)
# Thai spelling: no word starts before a combining mark or a following vowel, none after a leading vowel
NO_START = np.array([0x0E30, 0x0E31, 0x0E32, 0x0E33, *range(0x0E34, 0x0E3B), 0x0E45, *range(0x0E47, 0x0E4F)],
                    np.uint32)
LEADING = np.array(range(0x0E40, 0x0E45), np.uint32)
HASH_MUL = np.uint64(0x9E3779B97F4A7C15)
MAX_ROWS = 512
DEFAULT_BATCH_CHARS = 4096           # 2-4K characters per batch keep the LSTM in cache on one thread
MAX_WINDOW = 10_000
WINDOW_CONTEXT = 500
```

`LEN_BUCKET` indexed by word length `n`: 2,3,4 → 0; 5 → 1; 6 → 2; 7,8 → 3;
9,10,11 → 4; 12…16 → 5 (six buckets).

`NO_START` (word may not start at this code point) = 0E30, 0E31, 0E32, 0E33,
0E34–0E3A, 0E45, 0E47–0E4E. `LEADING` (word may not start at the *next* code
point) = 0E40–0E44.

```python
def _codes(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-32-le", "surrogatepass"), np.uint32)
```

So one input row is the UTF-32 code points of the whitespace-free line, little
endian, one `uint32` per character (surrogate code points preserved).

### 2.2 id table construction (`Segmenter.__init__`, verbatim core)

```python
vocab = json.loads(meta["vocab"])
table = np.load(ASSETS / manifest["charclass"])["classes"].astype(np.int16)
table[np.fromiter(map(ord, vocab), np.int64, len(vocab))] = np.arange(N_CLASSES, N_CLASSES + len(vocab))
self._ids = table
```

`table` starts as the 1,114,112-entry `uint8` LUT from `charclass.npz` (values
1–12; 0 is reserved for padding), is widened to `int16`, and then the **944
vocabulary characters are overwritten with ids 13…956**. Therefore
`ids[c] = 13 + vocab_index(c)` for known chars, `ids[c] = class(c) ∈ 1..12`
for unknown chars, and `ids = 0` for padding. `emb` has exactly 957 rows.

### 2.3 64-bit polynomial lexicon hashing and the 13 lexicon bits

```python
def _hash_rows(codes: np.ndarray) -> np.ndarray:
    """64-bit polynomial hash of each row of a (n, L) code-point matrix."""
    codes = codes.astype(np.uint64)
    h = codes[:, 0].copy()
    for j in range(1, codes.shape[1]):
        h = h * HASH_MUL + codes[:, j]
    return h
```

`HASH_MUL = 0x9E3779B97F4A7C15`; arithmetic is **mod 2^64 with silent wraparound**
(Rust: `wrapping_mul` / `wrapping_add`). At dictionary load:

```python
self.hashes = {n: np.unique(_hash_rows(_codes("".join(ws)).reshape(len(ws), n)))
               for n, ws in groups.items()}
self.lengths = sorted(self.hashes)
```

`groups` keeps words with `2 <= len(w) <= max_len (16)` and no whitespace; each
length has a sorted unique array of hashes.

```python
def bits(self, codes: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """(R, W) code points of a padded batch -> (R, W) int64 dictionary bits."""
    rows, width = codes.shape
    bits = np.zeros((rows, width), np.int64)
    cover = np.zeros((rows, width + 1), np.int64)
    c = codes.astype(np.uint64)
    h, n = c, 1                                   # h[:, i] = hash of the n characters from i
    ends = np.arange(1, width + 1)[None, :]
    for target in self.lengths:
        while n < target and h.shape[1] > 0:
            h = h[:, :-1] * HASH_MUL + c[:, n:]
            n += 1
        if n < target or h.shape[1] == 0:
            break
        table = self.hashes[n]
        hit = table[np.minimum(np.searchsorted(table, h), len(table) - 1)] == h
        hit &= (ends[:, :h.shape[1]] + (n - 1)) <= lengths[:, None]   # window inside the row
        if not hit.any():
            continue
        m, b = hit.shape[1], int(LEN_BUCKET[n])
        hit64 = hit.astype(np.int64)
        bits[:, :m] |= hit64 << b                 # a word of this length starts here
        bits[:, n - 1:] |= hit64 << (6 + b)       # ... ends here
        if n > 2:                                 # ... or continues here
            cover[:, 1:m + 1] += hit64
            cover[:, n - 1:width] -= hit64
    bits |= (np.cumsum(cover[:, :width], axis=1) > 0).astype(np.int64) << 12
    return bits
```

Exact meaning of the 13 bits (indexing `lex_table`'s 8192 rows **as a whole
bitmask**, not a sum of per-bit vectors):

- bit `b` (0…5): some dictionary word of length bucket `b` **starts** here;
- bit `6+b`: some such word **ends** here;
- bit 12: this position is **inside** a word of length ≥ 3 (a word of length 2
  never sets the inside bit, because of `if n > 2`). The cover is a difference
  array: `+1` at `start+1`, `−1` at `start+n−1`, then prefix-sum `> 0`.
- A hit at position `a` is only accepted if `a+n <= lengths[row]` — i.e. the
  whole word fits in the unpadded part of the row.
- `searchsorted` clamped to `len(table)-1` then equality test = membership in a
  sorted hash list.

The n-best lattice uses the same hashes (verbatim, `_lattice`):

```python
def _lattice(self, line: str) -> set[tuple[int, int]]:
    c = _codes(line).astype(np.uint64)
    spans: set[tuple[int, int]] = set()
    for n in self.lexicon.lengths:
        m = len(line) - n + 1
        if m <= 0:
            break
        h = c[:m].copy()
        for j in range(1, n):
            h = h * HASH_MUL + c[j:j + m]
        table = self.lexicon.hashes[n]
        hit = table[np.minimum(np.searchsorted(table, h), len(table) - 1)] == h
        spans.update((int(a), int(a) + n) for a in np.flatnonzero(hit))
    return spans
```

### 2.4 Windowing and batching (verbatim)

```python
def _windows(n: int) -> list[tuple[int, int, int, int]]:
    if n <= MAX_WINDOW:
        return [(0, n, 0, n)]
    out, keep = [], 0
    while keep < n:
        start = max(0, keep - WINDOW_CONTEXT)
        end = min(n, start + MAX_WINDOW)
        keep_to = n if end == n else end - WINDOW_CONTEXT
        out.append((start, end, keep, keep_to))
        keep = keep_to
    return out
```

Lines of up to 10,000 characters are fed whole; longer lines are read in
10,000-character windows overlapping by 2×500 characters, and only the middle
of each window is kept.

```python
def _same_run(a: str, b: str) -> bool:
    """Two adjacent characters of one run of ASCII letters or of one run of digits (never split). Digits are the
    decimal digits of any script: Thai digits (๐-๙) stay together like 0-9."""
    if a.isdecimal() and b.isdecimal():
        return True
    return a.isascii() and b.isascii() and a.isalpha() and b.isalpha()
```

Note `isdecimal()` covers **all Unicode decimal digits**, not just ASCII+Thai;
a Rust port must use the Unicode Nd property (or the exact Python table).

```python
def _batches(lengths: Sequence[int], batch_chars: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    if batch_chars <= 0:
        return [[i] for i in order]
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        new_max = max(cur_max, lengths[i] + 3)
        if cur and (new_max * (len(cur) + 1) > batch_chars or len(cur) >= MAX_ROWS):
            batches.append(cur)
            cur, new_max = [], lengths[i] + 3
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches
```

### 2.5 Feed construction, logits → word starts, unit mapping (verbatim core of `segment_lines`)

```python
k = self._unit_index(unit or str(self.unit)) if self.units else None
rows = [(i, *w) for i, t in enumerate(lines) for w in _windows(len(t))]
lengths = [end - start for _, start, end, _, _ in rows]
word_starts = [np.zeros(len(t), bool) for t in lines]
for group in _batches(lengths, self.batch_chars):
    width = max(lengths[r] for r in group)
    if width == 0:
        continue
    codes = np.zeros((len(group), width), np.uint32)
    for g, r in enumerate(group):
        i, start, end, _, _ = rows[r]
        codes[g, :lengths[r]] = _codes(lines[i][start:end])
    lens = np.array([lengths[r] for r in group], np.int64)
    valid = np.arange(width)[None, :] < lens[:, None]
    ids = np.where(valid, self._ids[codes], 0).astype(np.int64)
    feed = {"ids": ids, "lengths": lens.astype(np.int32)}
    if self._uses_bits:
        feed["bits"] = self.lexicon.bits(codes, lens)
    logits = self.session.run(["logits"], feed)[0]
    if logits.ndim == 3:     # every unit's logits, coarse to fine: a unit's word starts include
        assert k is not None                                               # all coarser units'
        logits = (logits[..., k] if self.units[k] in self.unnested_units   # (nested) unless unnested
                  else logits[..., :k + 1].max(-1))
    allowed = ~np.isin(codes, NO_START)
    allowed[:, 1:] &= ~np.isin(codes[:, :-1], LEADING)
    starts = (logits >= 0.0) & allowed
    starts[:, 0] = True
    for g, r in enumerate(group):
        i, start, _, keep_from, keep_to = rows[r]
        word_starts[i][keep_from:keep_to] = starts[g, keep_from - start:keep_to - start]
out = []
for i, t in enumerate(lines):
    pos = np.flatnonzero(word_starts[i]).tolist() + [len(t)]
    out.append(list(zip(pos[:-1], pos[1:])))
return out
```

Key points:

- Inputs: `ids` int64 `[B,W]`, `bits` int64 `[B,W]`, `lengths` int32 `[B]`.
  **Padding ids are 0, padding bits are 0**, and `W = max(lengths in batch)`.
- Output logits are `[B,W,3]`, one level per unit in coarse→fine order.
- Unit selection: `long` (k=0) → `max over level 0`; `short` (k=1) →
  `max over levels 0..1`; `minimal` (k=2, **unnested**) → **level 2 alone**.
  NumPy `max` on ties returns the *first* (coarsest) maximum — a Rust port
  must use `>` (strict) when folding to preserve ties.
- Decision: `start = (logit >= 0.0) AND code not in NO_START AND previous code not in LEADING`.
  First character of every line/window is always a start.
- Spans are `[start_i, start_{i+1})` with an implicit end at `len(line)`.

### 2.6 Cuts → token strings (verbatim core of `tokenize_batch`)

```python
jobs, maps, owner = [], [], []
for d, text in enumerate(texts):
    start = 0
    for line in text.split("\n"):
        idx = [start + j for j, c in enumerate(line) if not c.isspace()]
        if idx:
            jobs.append("".join(text[j] for j in idx))
            maps.append(idx)
            owner.append(d)
        start += len(line) + 1
cuts: list[set[int]] = [set() for _ in texts]
for d, idx, spans in zip(owner, maps, self.segment_lines(jobs, unit)):
    cuts[d].update(idx[s] for s, _ in spans[1:])
out = []
for d, text in enumerate(texts):
    space = [j for j, c in enumerate(text) if c.isspace()]
    cut = {j for j in cuts[d] if not _same_run(text[j - 1], text[j])} | set(space) | {j + 1 for j in space}
    bounds = [0] + sorted(j for j in cut if 0 < j < len(text)) + [len(text)]
    toks = [text[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    out.append(toks if keep_whitespace else [t for t in toks if not t.isspace()])
return out
```

Rules: split input on `\n`; remove **all** whitespace (`str.isspace()`, which is
Unicode-aware) into a whitespace-free "job" while remembering original indices;
segment each job; map each predicted start (except index 0) back to an original
index; drop a cut if the two characters straddling it are in the same
ASCII-letter or decimal-digit run; add cuts on **both sides** of every whitespace
character; slice between sorted bounds; optionally drop whitespace-only tokens.

### 2.7 The trailing `<s/>`

**The wheel never emits `<s/>`.** Neither `_runtime.py`, `__main__.py`,
`__init__.py`, `manifest.json`, nor the HF model card mentions it. In this
checkout, `web/js/latte.js` (untracked, in-flight work by another session) ends
`tokenize()` with:

```js
return lineResults.join('\n') + ' <s/>';
```

so `<s/>` is a **local wrapper/UI convention**, presumably a sentence boundary
marker for the web app, not part of the model's contract. A Rust wire function
aiming to match that in-flight JS must append `" <s/>"` at the very end; a port
aiming to match the Python wheel must not. (Also note two divergences in that
JS to watch: `isSameRun` uses JS `\d`+Thai digits instead of Python's Unicode
`isdecimal()`, and its nested-unit `max` loop uses `>` in a way that effectively
breaks ties toward the *last* maximum, unlike NumPy's first-maximum.)

### 2.8 Wire surface — `latte_segment`, `latte_nbest`, `latte_segment_opts`

The three LATTE names ride the existing string dispatch
`process(function, text, cutoff)` — no fourth argument and no extra export
(issue #3 chose the `cutoff` slot for scalar parameters). `latte_segment`
mirrors `tokenize(text, keep_whitespace=True, unit)`.

**`latte_nbest(text, cutoff)`.** `text` is plain text. `cutoff` is a strict
grammar, `<unit>` or `<unit>:<n>`:

| field | values | default |
|---|---|---|
| `<unit>` | `long`, `short`, `minimal` | `short` (and `medium`, the verifier's default, also means `short`) |
| `<n>` | base-10 integer in `[1, 100]` | `10` |

Output is compact JSON, one object per reading, best first:

```json
[{"tokens":["ตา","กลม"],"probability":0.4178397973566892},
 {"tokens":["ตากลม"],"probability":0.41417453195918347},
 {"tokens":["ตาก","ลม"],"probability":0.08321445529883417}]
```

`probability` is `exp(score)` as a JSON number and is **informational** — only
the reading order and the plain-first guarantee are contractual. Ranking is by
descending score with the plain `latte_segment` reading forced to index 0 even
on an exact score tie; the remaining ties keep dynamic-program insertion order
(upstream's stable sort). Candidate pieces are the union of the lexicon **and**
the word spans produced by all three units — not only the requested one.
Whitespace is a hard boundary in every reading, and runs of ASCII letters /
decimal digits stay whole. The lattice search reads at most **10,000
non-whitespace characters** (whitespace is free); a longer input errors instead
of allocating. The web UI is stricter than the wire on purpose: it caps n at 10
and the n-best path at 2,000 non-whitespace characters, refusing with a
user-facing Thai error before the wasm call so a browser tab cannot freeze
(plain `latte_segment`, n = 1, keeps the wire's unlimited behaviour).

**`latte_segment_opts(text, cutoff)`.** `text` is a compact JSON envelope and
`cutoff` is **ignored** — the wire name, not a leading `{`, disambiguates JSON
input from plain text. Exactly one of `text` (string) / `texts` (array of
strings) is required.

| key | type | default | notes |
|---|---|---|---|
| `text` | string | — | one text; mutually exclusive with `texts` |
| `texts` | array of strings | — | ≤ 100 entries; processed per text (see below) |
| `unit` | `long` / `short` / `minimal` | `short` | |
| `keep_whitespace` | bool | `true` | `false` drops whitespace-only tokens |
| `user_words` | array of strings | `[]` | ≤ 1000 raw entries; per-call hints |

Unknown keys and wrong types are rejected before any allocation, and every text
is subject to the same 10,000 non-whitespace cap. Output is always a JSON array
of token arrays, one per input text (also for the single-`text` form), mirroring
`tokenize_batch`:

```json
[["ตา","กลม"],["แมว","หมา"]]
```

`{"text":""}` → `[[]]`; `{"texts":[]}` → `[]`. User words are a **per-call
lexicon overlay**: `str.strip()` then silently dropped when shorter than 2
characters, longer than 16, or containing whitespace (there is no warning
channel). They are hints, not forced cuts — the model still decides the final
boundaries — and nothing global or session-scoped is mutated, so concurrent
calls cannot see each other's dictionary.

**Errors** (exact strings, the existing process-error type — never a panic
across the wasm boundary):

| error | when |
|---|---|
| `invalid unit` | unit token is not `long` / `short` / `minimal` |
| `n must be between 1 and 100` | n unparsable or outside `[1, 100]` |
| `text too long (max 10000 non-whitespace characters)` | an n-best text or any `_opts` text is above the cap |
| `malformed JSON` | the `_opts` envelope is not JSON |
| `invalid options` | `_opts` schema/type/cap violation (unknown key, both `text` and `texts`, `texts` > 100, `user_words` > 1000, …) |

**Accepted departure — per-text, not upstream-batched, processing.** The wheel's
`tokenize_batch` packs many lines into one ORT batch, and §6.3 explains why batch
composition moves int8 scales (up to 2.9% of lines). The wire processes each text
on its own (batch size 1), and n-best treats its whole text as one
whitespace-free line; per-text consistency is the chosen semantics. A multi-line
text can therefore differ from the wheel's batched call on the rare line whose
cuts the scale shift flips — disclosed and accepted, mitigated by keeping
multi-line `_opts` golden rows short.

**Parity.** `latte_segment` rows stay byte-exact. N-best rows carry
`"compare":"json-f64-1e-4"` in the golden: strings compare byte-for-byte, numeric
leaves compare with **1e-4 absolute** tolerance (ONNX Runtime's MLAS
accumulation order is not reproducible bit-for-bit). Goldens are captured from
the wheel in one process and corpus order — never from the Rust binary.

---

## 3. `charclass.npz` — exact table

`npz` contains one array `classes`, shape **(1114112,) = 0x110000**, dtype
**uint8**, all entries nonzero, values 1…12; class 0 is reserved for padding.
Class histogram: `{1:46, 2:22, 3:8, 4:10, 5:42, 6:26, 7:26, 8:670, 9:839,
10:7769, 11:135995, 12:968659}`. The table is already `Unicode category` buckets
(ASCII digits = 8, uppercase = 6, lowercase = 7, Thai = 11, "other" = 12, …);
see the sample runs at the end of this section.

Exact, byte-for-byte reconstruction (the gzip+base64 below was produced from
the wheel's `charclass.npz` array `.tobytes()`, so it is authoritative):

```python
import base64, gzip, numpy as np
B64 = "<PASTE FROM APPENDIX A>"
classes = np.frombuffer(gzip.decompress(base64.b64decode(B64)), dtype=np.uint8)
assert len(classes) == 0x110000
# then, in Python's construction:
ids = classes.astype(np.int16)
ids[np.fromiter(map(ord, vocab), np.int64, len(vocab))] = np.arange(13, 13 + len(vocab))
```

For a Rust implementation, embed the 1,114,112 bytes (or an RLE — the table has
only **1,933** runs; `start,end_exclusive,class`). The in-flight
`crates/tltk-core/data/latte_charclass.bin.gz` in this checkout decompresses to
2,228,224 bytes = the same table as **uint16 little-endian** (first two bytes
`0c 00` = class 12), matching the Python `.astype(np.int16)` step.

Sample runs (hex code point intervals, inclusive start/end, class):

```
[0x0000,0x0020) class=12
[0x0021,0x0023) class=9    [0x0024,0x0024) class=10
[0x0030,0x0039) class=8    (ASCII digits)
[0x0041,0x005A) class=6    (A-Z)
[0x0061,0x007A) class=7    (a-z)
[0x0E00,0x0E7F) class=11   (Thai block; 135,995 code points overall)
[0x323B0,0x10FFFF) class=12
```

Full base64 blob in **Appendix A**.

---

## 4. ONNX graph manifest

`ir_version=9`, `producer_name="onnx.quantize"`, `producer_version="0.1.0"`,
`graph.name="latte20_student"`. **Opsets: ai.onnx v17 + `com.microsoft` v1.**
Metadata properties: `onnx.infer="onnxruntime.quant"`, `config`, `vocab`.

Graph I/O (dynamic dims `batch`, `length`):

| direction | name | shape | dtype |
|---|---|---|---|
| in | `ids` | `[batch,length]` | int64 |
| in | `bits` | `[batch,length]` | int64 |
| in | `lengths` | `[batch]` | int32 |
| out | `logits` | `[batch,length,3]` | float32 |

Value info (all float32 unless noted): `x_emb [B,W,64]`, `x_lex [B,W,32]`,
`x_in [B,W,96]`, `pm_shape [2] int64`, `pm_width () int64`, `pm_pos [?] int64`,
`pm_pos_row [1,?] int64`, `pm_len [B] int64`, `pm_len_col [B,1] int64`,
`pm_valid [B,?] bool`, `pm_valid_f [B,?] float`, `pm_mask [B,?,1] float`,
`x_in_masked [B,?,96]`, `x_ncl [B,96,?]`, `conv0/conv0_relu [B,128,?]`,
`conv1/conv1_relu [B,128,?]`, `conv2/conv2_relu [B,128,?]`, `convs [B,384,?]`,
`seq0 [?,B,384]`, `lstm0 [?,2,B,128]`, `lstm0_t [?,B,2,128]`,
`seq1 [?,B,256]`, `lstm1 [?,2,B,128]`, `lstm1_t [?,B,2,128]`,
`seq2 [?,B,256]`, `out_mm [?,B,3]`, `out_add [?,B,3]`.

### 4.1 All 35 nodes, in execution order

Unlisted attributes are absent (defaults). `Cast.to` codes: 1 = FLOAT, 7 = INT64.

| # | op_type | inputs | outputs | attributes |
|---|---|---|---|---|
| 0 | Gather | `emb, ids` | `x_emb` | — |
| 1 | Shape | `ids` | `pm_shape` | — |
| 2 | Cast | `lengths` | `pm_len` | `to=7` (int64) |
| 3 | Gather | `lex_table, bits` | `x_lex` | — |
| 4 | Gather | `pm_shape, pm_one` | `pm_width` | `axis=0` → W |
| 5 | Unsqueeze | `pm_len, pm_ax1` | `pm_len_col` | axes=[1] (from initializer) |
| 6 | Concat | `x_emb, x_lex` | `x_in` | `axis=2` |
| 7 | Range | `pm_zero, pm_width, pm_one` | `pm_pos` | — (0..W−1) |
| 8 | Unsqueeze | `pm_pos, pm_ax0` | `pm_pos_row` | axes=[0] |
| 9 | Less | `pm_pos_row, pm_len_col` | `pm_valid` | — (bool [B,W]) |
| 10 | Cast | `pm_valid` | `pm_valid_f` | `to=1` (float) |
| 11 | Unsqueeze | `pm_valid_f, pm_ax2` | `pm_mask` | axes=[2] → [B,W,1] |
| 12 | Mul | `x_in, pm_mask` | `x_in_masked` | — (zero padding positions) |
| 13 | Transpose | `x_in_masked` | `x_ncl` | `perm=[0,2,1]` → [B,96,W] |
| 14 | Conv | `x_ncl, conv0_w, conv0_b` | `conv0` | `kernel_shape=[3]`, `pads=[1,1]` |
| 15 | Conv | `x_ncl, conv1_w, conv1_b` | `conv1` | `kernel_shape=[5]`, `pads=[2,2]` |
| 16 | Conv | `x_ncl, conv2_w, conv2_b` | `conv2` | `kernel_shape=[7]`, `pads=[3,3]` |
| 17 | Relu | `conv0` | `conv0_relu` | — |
| 18 | Relu | `conv1` | `conv1_relu` | — |
| 19 | Relu | `conv2` | `conv2_relu` | — |
| 20 | Concat | `conv0_relu, conv1_relu, conv2_relu` | `convs` | `axis=1` → [B,384,W] |
| 21 | Transpose | `convs` | `seq0` | `perm=[2,0,1]` → [W,B,384] |
| 22 | **DynamicQuantizeLSTM** (`com.microsoft`) | `seq0, lstm0_W_quantized, lstm0_R_quantized, lstm0_B, lengths, "", "", "", lstm0_W_scale, lstm0_W_zero_point, lstm0_R_scale, lstm0_R_zero_point` | `lstm0` | `direction="bidirectional"`, `hidden_size=128` |
| 23 | Transpose | `lstm0` | `lstm0_t` | `perm=[0,2,1,3]` → [W,B,2,128] |
| 24 | Reshape | `lstm0_t, merge_dirs` | `seq1` | `merge_dirs=[0,0,-1]` → [W,B,256] |
| 25 | **DynamicQuantizeLSTM** (`com.microsoft`) | `seq1, lstm1_W_quantized, lstm1_R_quantized, lstm1_B, lengths, "", "", "", lstm1_W_scale, lstm1_W_zero_point, lstm1_R_scale, lstm1_R_zero_point` | `lstm1` | `direction="bidirectional"`, `hidden_size=128` |
| 26 | Transpose | `lstm1` | `lstm1_t` | `perm=[0,2,1,3]` → [W,B,2,128] |
| 27 | Reshape | `lstm1_t, merge_dirs` | `seq2` | → [W,B,256] |
| 28 | **DynamicQuantizeLinear** | `seq2` | `seq2_quantized, seq2_scale, seq2_zero_point` | — (uint8) |
| 29 | Mul | `seq2_scale, out_w_scale` | `seq2_scale_out_w_scale_mul:0` | — |
| 30 | **MatMulInteger** | `seq2_quantized, out_w_quantized, seq2_zero_point, out_w_zero_point` | `out_mm_output_quantized` | — (int32) |
| 31 | Cast | `out_mm_output_quantized` | `out_mm_output_quantized_cast_output` | `to=1` (float) |
| 32 | Mul | `...cast_output, scale_product` | `out_mm` | — dequantize |
| 33 | Add | `out_mm, out_b` | `out_add` | — bias |
| 34 | Transpose | `out_add` | `logits` | `perm=[1,0,2]` → [B,W,3] |

Conv defaults (not written in the file, but implied by opset 17): `strides=[1,1]`,
`dilations=[1,1]`, `group=1`, `auto_pad="NOTSET"`. With `pad=(k−1)/2` and
stride 1, output width stays `W`.

### 4.2 All 32 initializers

Dtypes: F = float32, I8 = int8, I64 = int64. `numel` is element count.
For large tensors only min/max are shown; the exact bytes are in the wheel.

| name | shape | dtype | numel | values / range |
|---|---|---|---|---|
| `emb` | [957,64] | F | 61,248 | −2.4507365 … 2.7007773 |
| `lex_table` | [8192,32] | F | 262,144 | −1.3771966 … 1.1678770 |
| `conv0_w` | [128,96,3] | F | 36,864 | −1.5956891 … 1.2420664 |
| `conv0_b` | [128] | F | 128 | −0.9101163 … 0.0077202 |
| `conv1_w` | [128,96,5] | F | 61,440 | −2.6002698 … 2.3297067 |
| `conv1_b` | [128] | F | 128 | −0.7360695 … 0.1026723 |
| `conv2_w` | [128,96,7] | F | 86,016 | −2.3231432 … 1.8843909 |
| `conv2_b` | [128] | F | 128 | −0.7643454 … −0.0362809 |
| `merge_dirs` | [3] | I64 | 3 | `[0, 0, -1]` |
| `lstm0_B` | [2,1024] | F | 2,048 | −0.7277831 … 0.6557431 |
| `lstm1_B` | [2,1024] | F | 2,048 | −0.6331467 … 0.4685449 |
| `out_b` | [3] | F | 3 | `[-0.25400403, 0.15514079, -0.02677058]` |
| `pm_zero` | [] | I64 | 1 | `0` |
| `pm_one` | [] | I64 | 1 | `1` |
| `pm_ax0` | [1] | I64 | 1 | `[0]` |
| `pm_ax1` | [1] | I64 | 1 | `[1]` |
| `pm_ax2` | [1] | I64 | 1 | `[2]` |
| `lstm0_W_scale` | [2] | F | 2 | `[0.0131423315, 0.0119661158]` |
| `lstm0_W_zero_point` | [2] | I8 | 2 | `[0, 0]` |
| `lstm0_R_scale` | [2] | F | 2 | `[0.0195380505, 0.0172044467]` |
| `lstm0_R_zero_point` | [2] | I8 | 2 | `[0, 0]` |
| `lstm0_W_quantized` | [2,384,512] | I8 | 393,216 | −119 … 127 |
| `lstm0_R_quantized` | [2,128,512] | I8 | 131,072 | −82 … 127 |
| `lstm1_W_scale` | [2] | F | 2 | `[0.0199967269, 0.0294961613]` |
| `lstm1_W_zero_point` | [2] | I8 | 2 | `[0, 0]` |
| `lstm1_R_scale` | [2] | F | 2 | `[0.0140316077, 0.0158220250]` |
| `lstm1_R_zero_point` | [2] | I8 | 2 | `[0, 0]` |
| `lstm1_W_quantized` | [2,256,512] | I8 | 262,144 | −127 … 127 |
| `lstm1_R_quantized` | [2,128,512] | I8 | 131,072 | −123 … 127 |
| `out_w_scale` | [] | F | 1 | `0.010742840357124805` |
| `out_w_zero_point` | [] | I8 | 1 | `0` |
| `out_w_quantized` | [256,3] | I8 | 768 | −127 … 109 |

Total initializer elements: **1,430,493**.

---

## 5. Exact operator sequence (inputs → logits)

1. `x_emb = emb[ids]` (957×64 table gather) → `[B,W,64]`.
2. `x_lex = lex_table[bits]` (**8192×32 table indexed by the whole 13-bit mask**) → `[B,W,32]`.
3. `x_in = concat(x_emb, x_lex, axis=2)` → `[B,W,96]`.
4. Mask: `valid[b,i] = i < lengths[b]`; `x_in_masked = x_in * valid.float()[...,None]`
   (padding rows become exactly 0; note conv **biases** still act there).
5. `x_ncl = transpose(x_in_masked, [0,2,1])` → `[B,96,W]`.
6. Three parallel 1-D convs (float32): widths 3/5/7, pads 1/2/3, 128 output
   channels each → ReLU → concat along channel → `convs [B,384,W]`.
7. `seq0 = transpose(convs,[2,0,1])` → `[W,B,384]` (time-major, as LSTM expects).
8. **BiLSTM #0** (`DynamicQuantizeLSTM`, 384→128 per direction) with
   `lengths` → `lstm0 [W,2,B,128]`; transpose `[0,2,1,3]` and
   reshape `[0,0,-1]` → `seq1 [W,B,256]` (forward‖backward concatenation).
9. **BiLSTM #1** (256→128 per direction) → `lstm1 [W,2,B,128]`; same
   transpose+reshape → `seq2 [W,B,256]`.
10. `DynamicQuantizeLinear(seq2)` → `uint8 q [W,B,256]`, scalar `s` (float),
    scalar `zp` (uint8).
11. `MatMulInteger(q [W,B,256], out_w_quantized [256,3] int8, zp, 0)` → int32
    `[W,B,3]`; cast to float; multiply by `s * out_w_scale`; add `out_b`.
12. `transpose(...,[1,0,2])` → `logits [B,W,3]`.

### Custom Rust implementations required

Trivial: Gather, Shape, Cast, Concat, Range, Unsqueeze, Less, Mul, Transpose,
Reshape, Relu, Add. Non-trivial:

1. **1-D float convolution** (3 kernels; can be fused into one im2col/GEMM or
   direct loop). Straightforward.
2. **`DynamicQuantizeLSTM` ×2** — the hard one: dynamic uint8 quantization of
   the input matrix, per-direction int8 weight dequantization, i/o/f/c gates,
   forward+backward passes, `sequence_lens` masking. See §6.2.
3. **`DynamicQuantizeLinear`** — scale/zero-point computation with round-to-even.
4. **`MatMulInteger`** with mixed uint8×int8 and zero points, int32 accumulate,
   then float dequantization.

No quantize/dequantize scale nodes, no `Clip`, no `Gemm`/`MatMul` (float), no
subgraphs, no control flow.

---

## 6. Quantization scheme — exact details

The model is a **hybrid**: embeddings, lexicon table, conv weights/biases, LSTM
biases and output bias stay **float32**; only the two LSTM weight/recurrence
matrices and the final output weight are int8.

### 6.1 `DynamicQuantizeLinear` on `seq2` (final head)

Empirically verified against ORT on the real model (probe input `W=6`):
`seq2 min=-0.8998271, max=0.9730081` gave `scale=0.0073444517`,
`zero_point=123` (uint8), and my recomputed `q` matched ORT's exactly.

- Output `y_quantized` is **uint8 in [0,255]** (confirmed by runtime tensor type),
  `y_scale` and `y_zero_point` are scalars.
- `scale = (max(X) − min(X)) / 255.0` (float32 arithmetic, over **all elements**
  of `seq2` — i.e. over the *whole batch*, all timesteps and all 256 channels).
- `zp = saturate(round_to_nearest_even(−min(X)/scale), 0, 255)`.
- `q = saturate(round_to_nearest_even(X/scale) + zp, 0, 255)`.
  ORT's MLAS uses `nearbyintf` / SSE `_mm_cvtps_epi32` → **banker's rounding**,
  not `round-half-away-from-zero`.

Final head: `out_w_quantized` is **int8** `[256,3]`, **per-tensor** scale
`0.010742840357124805`, zero point `0` (scalar). `MatMulInteger` accumulates
`int32 += (uint8 a − zp_a) * (int8 b − 0)` (uint8×int8 mixed inputs are allowed),
then `out = int32_as_float * (scale_seq2 * scale_out_w) + out_b`.

### 6.2 `DynamicQuantizeLSTM` internals (empirically pinned down)

Schema (ORT contrib v1): X `[T,B,I]` float; W `[dir,I,4H]` int8; R `[dir,H,4H]`
int8; B `[dir,8H]` float; `sequence_lens` int32 `[B]`; optional `initial_h`,
`initial_c`, P (all empty here); `W_scale/W_zero_point`, `R_scale/R_zero_point`.
Here all zero points are 0 and scales have shape `[2]` — i.e. **per-tensor per
direction**, *not* per-channel. Output Y `[T,2,B,128]` (only Y is consumed; no
`Y_h`/`Y_c`).

Exact behaviour, each direction independently:

- Gate order in W/R/B is **i, o, f, c** (input, output, forget, cell), each a
  contiguous block of `H=128` columns; B's first `4H` = `Wb[iofc]`, last `4H` =
  `Rb[iofc]`. Activations: `f = σ`, `c-gate = tanh`, `h = tanh`; no `clip`, no
  peepholes, `input_forget=0`.
- **The input X is dynamically quantized once for the entire 3-D X tensor**
  (all `T×B×I` elements share one uint8 scale and zero point, computed with
  the same formula as §6.1). This was verified by feeding two timesteps with
  different ranges: both timesteps used the *global* scale, not a per-step one.
- Gates: `gi = float(int32[(Xq − zp_x) @ Wq[dir]]) * (scale_x * W_scale[dir]) + Wb[dir]`
  for the input term, and the **recurrence term is dynamically quantized at
  every timestep**: `H_{t−1}`'s 128 values get their own uint8 scale and zero
  point each step (min/max over those 128 elements, ties-to-even — the same
  formula as §6.1), and the step computes
  `acc = int32[Σ_i (Hq[i] − zp_h) · Rq[dir][i][g]]`, scaled by
  `(scale_h * R_scale[dir])` before `+ Rb[dir]`.
  **Correction (LATTE port, 2026-11-18):** earlier revisions of this note
  claimed the recurrence was computed in float and that the hidden state was
  *not* dynamically quantized, citing a two-step probe that read
  `h0 ∈ [−0.76, 0.76]` back through the R matmul to ≤1.8e−7. That conclusion was
  wrong. The float-recurrence port drifted from ORT by ~8.6e−3 in the recurrence
  and up to ~5.7e−2 in the logits — enough to flip an n-best reading order. With
  the per-step hidden quantization the logits agree with ORT to ≤2.2e−6 and the
  n-best probabilities to ~1e−6. `crates/tltk-core/src/latte.rs`'s `lstm()` is
  the verified reference behaviour; a port that reproduces it is what the parity
  goldens measure.
- Update: `c_t = σ(f) ⊙ c_{t−1} + σ(i) ⊙ tanh(g)`; `h_t = σ(o) ⊙ tanh(c_t)`;
  initial `h_0 = c_0 = 0`.
- Direction 0 processes `t = 0 … T−1`, direction 1 processes `t = T−1 … 0`;
  outputs are concatenated as `[forward, backward]` along the last axis,
  then reshaped into 256 channels (node 24/27: dims `[W,B,2,128]` → `[W,B,256]`,
  so the layout is `f0…f127, b0…b127`).
- `sequence_lens`: output Y is **zeroed for timesteps `t >= length`** for that
  batch row. I confirmed empirically (also that ORT applies this same
  `t < length` mask to the *backward* direction, rather than masking the first
  `T−length` positions — an implementation quirk to replicate). `Y_h` returns
  `h` at the last *valid* timestep.

### 6.3 Why batching can change results

Because `DynamicQuantizeLinear` computes `seq2`'s scale from the **whole
batch** (and the LSTM's X scale from the whole `[T,B,I]` tensor), the logits for
one line change with the other lines in its batch. This is exactly the effect
acknowledged in the model card: "because of int8 quantisation, a line can be
segmented slightly differently alone than in a batch: up to 2.9% of the lines
of a test set, with no score moving by more than 0.05." A Rust port must
reproduce the *same batch grouping* (`_batches`, `batch_chars=4096`, `MAX_ROWS=512`,
sorted by length) to match a given Python call — and for a single-line wire
function, use batch size 1. That 0.05 line is a bound on the **batch effect** of
int8 scales, not a parity tolerance for this port: after the per-step
hidden-state quantization of §6.2 the logits agree with ORT to ≤2.2e−6 (see §7
item 3 for the asserted tolerance on probability wires).

---

## 7. Parity risks (ranked)

1. **Batch-dependent scales.** Both dynamic quantizers use tensor-wide
   min/max. Different batch composition ⇒ different logits ⇒ possibly
   different cuts. Match `_batches()` exactly or always batch size 1.
2. **Round-to-nearest-even** in quantization (`nearbyintf`/SSE) and **exact
   float32 scale arithmetic** `(max−min)/255`. Using `round()` (half away)
   or f64 intermediates will shift integer codes at ties and flip logits near 0.
3. **Gate threshold `logit >= 0.0`.** Tiny float differences in conv/LSTM/MatMul
   accumulation order (MLAS vs naive loops) can flip a start, so parity is
   compared at the logit level before token strings. Settled rule for the wire:
   `latte_segment` output stays **byte-exact**; probability-bearing wires
   (`latte_nbest`, `latte_segment_opts`) are asserted at **1e−4 absolute on
   numeric leaves, strings byte-exact** (`json-f64-1e-4` in the goldens). With
   the per-step hidden-state quantization of §6.2 the port's logits agree with
   ORT to ≤2.2e−6 and the n-best probabilities to ~1e−6, so 1e−4 is a guard with
   margin, not the observed drift. The model card's "no score moving by more
   than 0.05" is a statement about batch effects on int8 scales (§6.3) and is
   **obsolete as a parity bound** — never use it as one.
4. **Nested-unit `max` tie behaviour**: NumPy's `max` keeps the first maximum
   (coarser level). Use strict `>` when folding levels 0..k.
5. **64-bit hash wraparound** (`wrapping_mul`/`wrapping_add`) and the exact
   `searchsorted`-clamped membership test; a different hash formulation breaks
   all lexicon bits.
6. **Lexicon bits**: `lex_table[bitmask]` is a single gather of a 13-bit mask,
   not a sum of per-bit embeddings; bit 12 only from words of length ≥3;
   hit acceptance requires `start+n <= row_length`; padded positions keep bits 0.
7. **Thai orthography and runs**: exact `NO_START`/`LEADING` code point sets;
   Unicode `isdecimal()` (all Nd scripts) and `isspace()`; `isascii() && isalpha()`
   for the letter run. The in-flight JS differs here.
8. **`sequence_lens` semantics** in both LSTMs, including the backward-direction
   quirk and zeroing beyond the length; `h,c` initialised to zero.
9. **Whitespace/windowing**: `split("\n")` (not universal newlines), cuts on both
   sides of whitespace, first character always a start, 10,000-char windows with
   500-char context, and lines with zero non-whitespace characters skipped.
10. **`<s/>`** is a repo-local wrapper convention; decide explicitly which
    surface the Rust wire should match.

Recommended verification: capture logits (`[B,W,3]`) from the Python wheel
(ORT 1.32, CPU, 1 thread) for a fixed corpus with `batch_chars=0` (one line per
batch), then compare Rust logits element-wise before comparing token strings;
then repeat with the default batched path to validate §6.3.

---

## Appendix A — exact charclass table (base64 gzip of the uint8 LUT)

```text
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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD7Nv99jiyMAABEA
```

(4,456 base64 chars → 3,342 gzip bytes → 1,114,112 uint8 bytes. Verify with
`len(np.frombuffer(gzip.decompress(base64.b64decode(B64)), np.uint8)) == 0x110000`.)

## Appendix B — reproduction commands

```bash
mkdir -p /tmp/opencode/latte && cd /tmp/opencode/latte
python3 - <<'PY'
import urllib.request
for url, out in [
 ("https://huggingface.co/iapp/OpenThai2.0-LATTE-WordSeg-Lite/resolve/main/openthai2_0_latte_wordseg_lite-0.1.2-py3-none-any.whl","latte.whl"),
 ("https://huggingface.co/iapp/OpenThai2.0-LATTE-WordSeg-Lite/resolve/main/model.int8.onnx","model.int8.onnx.hf")]:
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    open(out,"wb").write(urllib.request.urlopen(req, timeout=180).read())
PY
unzip -o latte.whl -d wheel
uv run --with onnx --with onnxruntime --with numpy python - <<'PY'
import onnx
m = onnx.load("model.int8.onnx.hf")
print(m.ir_version, [(o.domain,o.version) for o in m.opset_import], len(m.graph.node), len(m.graph.initializer))
PY
```
