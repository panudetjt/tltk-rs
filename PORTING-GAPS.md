# PORTING-GAPS.md — upstream tltk 1.10 surface not yet ported

Scope note: the project is named after tltk, but the parity contract
(AGENTS.md) covers the 16 validator functions + `textana` (1.11 semantics).
This document inventories everything else the upstream package exposes, so
expanding the scope is a checklist, not archaeology.

- **Source of truth**: the Python oracle copy at
  `~/.cache/tltk-rs/tltk-1.10-pkg/` (`tltk/nlp.py` + `tltk/corpus.py`) — the
  same copy `tools/convert/convert.py` reads. `mise run upstream-check` keeps
  it honest.
- **1.11 relevance**: upstream 1.11's only functional change is inside
  TextAna (LdvTTR ×1000 → ×100) — **ported 2026-10-16**: `textana` carries
  1.11 semantics while the other 16 functions stay 1.10. Future upstream
  versions: re-run `mise run upstream-check` and re-open this inventory.

Status legend:

- **wire** — callable via `Session::process` / worker `POST /process`
- **internal** — ported into `tltk-core`, not exposed on the wire
- **MISSING** — no Rust counterpart yet
- **N/A** — replaced by a different mechanism (lazy loaders → `Models::load`)
- **OUT OF SCOPE** — training / corpus-construction / download tooling; not
  inference, deliberately not planned

---

## 1. nlp.py — what `from tltk.nlp import *` gives you

31 names in the `__init__` export line. Currently:

| status | names |
|---|---|
| wire (14) | `word_segment`, `syl_segment`, `spell_variants`, `pos_tag`, `segment`, `ner_tag`, `g2p`, `th2ipa`, `th2roman`, `th2read`, `spoonerism`, `textana`, `th2ipa_all`, `sylparse_all` |
| internal (6) | `initial` (→ `Models::load`), `spell_candidates` (`spell.rs:53`), `pos_tag_wordlist` (folded into `pos.rs::pos_tag`), `ner` (`ner.rs:133`), `SylAna` + `WordAna` (`textana.rs`, with non-export helpers `ARF`/`Duplicate_word`/`pack_ner`) |
| MISSING (11) | see below |

### 1a. Segmentation variants — MISSING

| upstream | notes |
|---|---|
| `word_segment_mm` / `wordseg_mm` | the `method="mm"` maximal-matching segmenter. Rust `word_segment` is colloc-only (`wordseg.rs:12`). Needs the mm chart (`chartparse_mm`) + `BigramProb`. |
| `word_segment_nbest` | n-best segmentation; needs `wordsegmm_bn` + `chartparse_mm_bn` + `chartparse_ngram`. |
| `chartparse_mm`, `chartparse_ngram`, `chartparse_mm_bn`, `BigramProb` | the charts backing the two above. (`chart_parse` in `g2p.rs:727` is the g2p-side chart — a different animal, already ported.) |

### 1b. Analysis layer — TextAna family ported, the rest MISSING

| upstream | notes |
|---|---|
| ~~`SylAna(s, p)`~~ | ✅ `textana.rs::syl_ana` (only the cluster/leading outputs WordAna counts). |
| ~~`WordAna(w)`~~ | ✅ `textana.rs::wordana` — g2p + per-syllable SylAna + tone/final-correction counters. |
| ~~`TextAna(alltext, ...)`~~ | ✅ `textana.rs::textana` — default options only (`TextOption="par"`, `WordOption="colloc"`, `UDParse="none"`); **tltk 1.11 semantics** (LdvTTR ×100). Goldens: `local_python_extra.jsonl` (1.11 oracle). |
| `TextAna2json` | TextAna → file/JSON wrapper — trivially derivable from the wire JSON; not exposed. |
| `TextClass`, `txt2feat` | level/genre classification; needs `chklevel_rf.pklz` + `chkgenre_rf.pklz` (sklearn random forests — **not converted yet**, would need an RF decoder in convert.py/model.rs). |
| `TranformData*` | TextClass helper — MISSING. |
| `SylProp`, `WordProp`, `TextProp` | folded into `textana.rs` counters / JSON emit. |

Data status: the word-tier lexicon TextAna reads (`word.pklz` → WT01) is
**converted, decoded and in use** (`Models::wordtier`, consumed by
`textana.rs`). The two RF classifiers and `thamalt.mco` are not converted.

### 1c. Pipelines & parsing — tree helpers ported, MaltParser MISSING

| upstream | notes |
|---|---|
| ~~`chunk(txt)`~~ | ✅ wire (one-shot sylseg → EDU → wordseg → POS → NER). |
| ~~`TNC_tag`~~ | ✅ wire (XML tagging transcription + POS for corpus prep). |
| `MaltParser`, `MaltParser_wordlist` | UD dependency parsing, model `thamalt.mco`. Hardest single item: transition-based parser + model conversion. `TextAna(UDParse="Malt")` depends on it. |
| ~~`print_dtree`, `dg_print`, `dg_save`, `scan_dtree`, `get_tree_depth`, `tree_max_depth`~~ | ✅ wire (`dtree.rs`) — consume the parse-RESULT dict (`dg_save` shape), NOT MaltParser; oracle goldens from nltk `DependencyGraph` via `ref_dump`. Cycle edge cases: `RecursionError` mirrored; pure-cycle `print_dtree` prints nothing upstream (nothing reachable from node 0). |

### 1d. Transliteration variants — wire

| upstream | notes |
|---|---|
| `th2ipa_all(inp)` | ✅ wire — all-candidate IPA (vs the single best path `th2ipa` returns). |
| `sylparse_all` | ✅ wire — all-candidate variant of `sylparse` (`g2p.rs`). |

### 1e. thaidict management + utils — ✅ wire (7) / N/A (1)

| upstream | status | notes |
|---|---|---|
| `read_thaidict` | ✅ wire | reads newline-separated words into TDICT custom set (`spell.rs:323`). |
| `reset_thaidict` | ✅ wire | clears TDICT custom and stock entries (`spell.rs:318`). |
| `check_thaidict` | ✅ wire | probes presence in TDICT, returns 1 or 0 (`spell.rs:328`). |
| `known` | ✅ wire | filters words present in TDICT (`spell.rs:336`). |
| `edits2` | ✅ wire | Norvig edits2 spelling candidates (edits1∘edits1, `spell.rs:260`). |
| `match_Engtext` | ✅ wire | matches English/symbols ASCII 32..=126 except '!' (`spell.rs:247`). |
| `wrd_len` | ✅ wire | classifies word length: >20 -> 'l', >10 -> 'm', else 's' (`spell.rs:235`). |
| `read_thdict` | N/A | model-loader path for stock pickled dict; replaced by `thdict.bin.gz` in `Models::load`. |
### 1f. Model loaders — N/A

`pos_load`, `ner_load`, `useg_model_load`, `useg_model_compress`,
`text_class_model_load`, `read_sylpattern`, `read_syldict`, `read_PhSTrigram`,
`read_sylvar`, `read_wordtier`, `read_stat` — all replaced by
`Models::load`/`embedded` + `tools/convert/convert.py` (one-shot converted
artifacts instead of lazy pickle loads).

---

## 2. corpus.py — what `from tltk.corpus import *` gives you

25 names. Currently:

| status | names |
|---|---|
| wire (5) | `similarity`, `similar_words`, `outofgroup`, `analogy`, `w2v_plot` (as `plotwords`, PCA PNG — `w2v.rs:523`) |
| internal (3) | `w2v_load` (→ `Models::w2v`), `cosine_similarity` (`w2v_similarity`, `w2v.rs:293`), `w2v_exist` (implicit via `most_similar`) |
| MISSING (inference) | see below |
| OUT OF SCOPE | see below |

### 2a. Word-level n-gram statistics — MISSING

`unigram`, `bigram`, `trigram`, `collocates`, `compute_colloc2`,
`TNC_load`/`trigram_load`/`TNC3g_load`.

These query **TNC3g** (word-level n-grams from the Thai National Corpus),
which is a separate upstream download (`download_TNC3g`), not shipped inside
the tltk package — convert.py doesn't touch it today. Note `sylseg.rs` has
syllable-level stats (needed by the segmenter); this is a different dataset.

### 2b. w2v extras — PARTIAL (`w2v` + `compound` + 3 plot renderers ✅ wire 2026-10-16)

| upstream | notes |
|---|---|
| `w2v(w)` | ✅ wire 2026-10-16. Raw fetch: the WV01 blob now carries the RAW f32 matrix (regenerated from TNCc5model3.bin via convert.py's gensim path); `W2v` keeps `vectors_raw` alongside the unit vectors. Wire emits `str(list(v))` (numpy float32 shortest-repr); OOV → `()`. |
| `compound(w1, w2)` | ✅ wire 2026-10-16. Oracle pipeline: dnrm2 (exact-f64 sqrt of f64 Σx²) + single-rounded `f32(f64(x)·(1/norm))` units + f32 dot (`w2v_oracle_similarity`). fn_w2v golden pairs are byte-stable under this dot layout (277/4000 vocab pairs match the oracle exactly; those are the corpus rows). |
| `similar_edu(seg, n)` | ❌ BLOCKED-DATA: needs TNCd2v.bin — upstream URL `https://www.arts.chula.ac.th/ling/wp-content/uploads/TNCd2v.bin` returns HTTP 403 (checked 2026-10-16); no alternative source. |
| `w2v_diffplot`, `w2v_dimplot`, `w2v_compare_color` | ✅ wire 2026-10-16. Hand-rolled Rust PNG rendering (plotwords precedent; no image crate). Structural parity: same axes (measured off the oracle renders: line plots 555x418, spines x=50..546/y=10..379; compare_color gray plot area 45..787 × 9..130, dot rows y≈12.5/68.5/124.5, dots x 60..780), tab10/Spectral colors, legends, y-tick labels. **Disclosed diff**: PNG bytes differ from matplotlib (same class as the 2 plotwords rows) — `corpus_fn_plots.jsonl` × `golden/local_python_fn_plots.jsonl` is fail=N by design (every row a PNG-byte diff, missing_golden=0); `mise run parity` special-cases it. OOV: diffplot/compare_color raise (upstream KeyError), dimplot filters. |

### 2c. Training / building / downloading — OUT OF SCOPE

`Corpus_build`, `W2V_train`, `D2V_train`, `W2V_load`, `D2V_load`, `Corpus`,
`Xwordlist`, `download_TNCw2v`, `download_TNC3g`, `download_TNCd2v`.
Corpus construction and gensim training are authoring-side tooling, not
runtime inference; they stay in Python even under an expanded scope.

---

## 3. Model artifacts not yet converted (convert.py checklist)

| artifact | needed by | status |
|---|---|---|
| `thdict`, `sylseg.3g`, `sylform_var.pklz`, `sylrule.lts`+`thaisyl.txt`, `PhSTrigram.sts`, `pos_tagger.pklz`, `ner_tagger.pklz`, `sent_segment.pklz`, `word.pklz`→WT01, `TNCc5model3.bin` | current 17 | ✅ converted |
| `wordtier.bin.gz` (WT01) | TextAna/WordAna tier features | ✅ in use (`textana`) |
| `chklevel_rf.pklz`, `chkgenre_rf.pklz` | `TextClass` | ❌ needs sklearn-RF converter |
| `thamalt.mco` | `MaltParser` | ❌ needs parser port + converter |
| TNC3g n-grams | corpus.py word n-grams | ❌ separate upstream download, no converter |

---

## 4. If/when scope expands — suggested order

Cheapest → hardest, keeping the parity discipline (new goldens captured from
the **Python oracle**, never from the Rust binary; byte-identical contract per
AGENTS.md):

**Wire-emit traps the oracle will spring on you** — each of these cost a probe
round-trip while porting `textana`, so probe the raw Python output with a
throwaway script before writing the Rust emit:

- Python dicts expose **insertion-ordered** keys, and a field born from a
  conditional `+=` exists in `vars(obj)` only when non-zero (`CrfCVL` appears
  as a raw int only when > 0; `DesNegC` vanishes at 0) — emit `sort_keys`-style
  and replicate presence rules field by field.
- `statistics.mean` over ints returns **int on exact division**, float
  otherwise; `statistics.stdev` is exact-fraction arithmetic then one
  correctly-rounded sqrt (for int inputs the closed form
  `(n·S2−S1²)/(n·(n−1))` in i128 is bit-identical).
- Floats serialize as **CPython repr** (shortest round-trip digits; fixed
  notation only while −4 < decpt ≤ 16), and float-sum order follows
  first-appearance order — preserve the accumulator sequence, not just the
  value.

Capture goldens with `tools/convert/ref_dump.py` corpus mode — one process,
corpus order (colloc counter state accumulates exactly like the verifier's
single session) — and extend its dispatch when the corpus gains a function.
`textana.rs`'s module doc is the worked example of the whole dance.

1. `th2ipa_all`, `sylparse_all` — variants of already-ported logic.
2. ~~`chunk`, `TNC_tag`~~ — ✅ wire 2026-10-16 (pure composition of existing primitives).
1. ~~`th2ipa_all`, `sylparse_all`~~ — ✅ wire (2026-10-16, g2p.rs).
2. `chunk`, `TNC_tag` — pure composition of existing primitives.
3. ~~`SylAna`, `WordAna`~~ — ✅ done 2026-10-16 (inside `textana.rs`).
4. corpus.py word n-grams (`unigram`/`bigram`/`trigram`/`collocates`) — needs TNC3g download + converter + new goldens.
5. ~~`TextAna`~~ — ✅ done 2026-10-16 (defaults surface, 1.11 semantics); `TextAna2json` remains trivial.
6. `word_segment_mm` / `word_segment_nbest` — new charts; note the wire currently has no way to select the method.
7. `TextClass` — RF model conversion is a new converter capability.
8. `MaltParser` — biggest lift; candidate to keep out of scope.

Expanding the wire surface touches, per function:
`convert.py` (if new model) → `model.rs` loader → core module →
`Session::process` arm (`lib.rs:83`) → worker function list
(`tltk-worker/src/lib.rs:33`) → web UI (`web/js/app.js`) → verifier corpus +
goldens from the Python oracle.
