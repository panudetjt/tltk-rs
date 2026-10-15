# tltk-rs

Rust rewrite of [TLTK](https://github.com/panudetjt/tltk) (Thai Language
Toolkit, Chulalongkorn University) — the Python package behind
[app.stichula.org/py/tltk](https://app.stichula.org/py/tltk/) — served as a
JSON API on a **Cloudflare Worker**.

Parity is **defined by the stichula validator** and enforced by the
`tltk-valifier` harness: every corpus case is sent to the live validator, the
response is stored as a golden row, and the Rust implementation must produce
byte-identical output.

```
corpus.jsonl ──▶ scripts/capture_stichula.py ──▶ golden/stichula.jsonl
      │                                                 │
      └──▶ tltk-valifier check ◀─────────────────────────┘
```

## Layout

| path | what |
|---|---|
| `crates/tltk-core/` | the library: all 16 functions, no network, no async |
| `crates/tltk-worker/` | Cloudflare Worker (workers-rs) — thin HTTP shell over the core |
| `crates/tltk-valifier/` | parity harness (`run` / `check` / `selftest`) |
| `tools/convert/` | Python scripts: pickle → portable binary models, local oracle |
| `corpus/` | test inputs (`{function, text, cutoff}` JSONL) |
| `golden/` | captured responses (stichula + local Python oracle) |
| `scripts/` | corpus gen, stichula capture, R2 upload |

Core and server are **separate crates**: `tltk-worker` depends on `tltk-core`,
which never touches HTTP so it can also be used as a plain library.

## API (mirrors the validator)

`POST /process` with `{"text": "...", "function": "...", "cutoff": "medium"}` →
`{"success": true, "function": ..., "input": ..., "result": "..."}` (or
`success: false` + `error`). Functions: `th2ipa`, `th2roman`, `word_segment`,
`syl_segment`, `segment`, `g2p`, `th2read`, `spell_variants`, `spoonerism`,
`pos_tag`, `ner_tag`, `similar_words`, `similarity`, `outofgroup`, `analogy`,
`plotwords`.

## Model pipeline

The Python package ships pickled models (nltk PerceptronTagger, sklearn-crfsuite
CRF, sklearn RandomForest, gensim Word2Vec, trigram dicts). `tools/convert/convert.py`
converts each into a compact sorted-entry binary format (`*.bin.gz`), decoded
with binary search at load:

| blob | size (gz) | source | served from |
|---|---|---|---|
| thdict | 0.4 MB | thdict pickle | embedded in WASM |
| sylseg | 2.5 MB | sylseg.3g + read_stat derivations | embedded |
| sylvar | 0.06 MB | sylform_var.pklz | embedded |
| phs | 1.2 MB | PhSTrigram.sts (8 derived tables) | embedded |
| pos | 1.1 MB | pos_tagger.pklz (PerceptronTagger) | embedded |
| edu | 6.2 MB | sent_segment.pklz (DictVectorizer+RF) | **R2** (`MODELS` binding) |
| w2v | 11.6 MB | TNCc5model3.bin (gensim KeyedVectors) | **R2** |
| ner | 0.09 MB | crfsuite model (see NER note) | R2 (optional) |

Workers cap uploads at 10 MB gz, which is why edu + w2v ride in R2 and are
fetched once per isolate at cold start.

## Parity status (valifier, corpus = corpus/corpus.jsonl, 244 cases)

- **vs the live stichula validator: 241/244 (98.8%)**
- vs the local Python oracle (tltk 1.10 wheel in a venv): 227/229
  (one case is a ref_dump formatting artifact — the local wrapper lacks the
  server's OOV check; `selftest` covers the anchor case 9/9)

Known, documented divergences:

1. **NER weight precision** — the crfsuite model artifact is recoverable only
   through `Tagger.dump()` (6-decimal weights; the original
   `ner_tagger.pklz` carries the model in a FileResource temp file that only
   materialises at unpickle time). One tag-boundary case in the corpus flips
   on a razor-thin margin. TODO: parse the crfsuite binary for full f32
   weights.
2. **plotwords** — same `IMAGE:<base64 png>` wire shape (a Rust-rendered PCA
   scatter); the pixels differ from the server's matplotlib rendering, which
   is out of scope.
3. **Stateful colloc counters** — Python mutates module-global BiCount/Count /
   TotalWord once per new syllable pair per PROCESS; we replicate per request
   (= a fresh Python process). A long-lived server process drifts slightly
   from this; the golden capture tolerates it.

## Commands

```bash
# build the valifier
cargo build -p tltk-valifier

# sanity checks on the anchor case
./target/debug/valifier selftest crates/tltk-core/data

# run the core over the corpus
./target/debug/valifier run corpus/corpus.jsonl crates/tltk-core/data

# parity check vs golden (pass/fail per case; exit 1 on any mismatch)
./target/debug/valifier check corpus/corpus.jsonl golden/stichula.jsonl crates/tltk-core/data

# regenerate models from the Python sources (needs the venv in tools/convert)
python3 tools/convert/convert.py <tltk-package-dir> crates/tltk-core/data \
    --ner-model tools/convert/ner.crfsuite
```

## Deploy the worker

```bash
npx wrangler login
npx wrangler r2 bucket create tltk-models          # once
./scripts/upload_r2.sh tltk-models                 # edu + w2v (+ ner)
cd crates/tltk-worker && npx wrangler deploy       # builds wasm + uploads
```

Local dev (models fetched from R2 live): `npx wrangler dev` inside
`crates/tltk-worker`. Without the R2 binding, the five embedded models still
serve 11 of 16 functions; edu/w2v functions return an explicit error.

## Credits

TLTK by Wirote Aroonmanakun, Chulalongkorn University (PyPI `tltk` 1.10).
This project reimplements the inference pipelines in Rust for the
[stichula.org](https://app.stichula.org/py/tltk/) API surface; model artifacts
remain © their original authors.
