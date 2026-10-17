# tltk-core

Rust port of [TLTK](https://pypi.org/project/tltk/) (Thai Language Toolkit,
Chulalongkorn University) — the inference core behind
[tltk-rs](https://github.com/panudetjt/tltk-rs). No network, no async, no
global mutable state beyond session counters: a `Tltk` carries converted
model blobs, a `Session` runs the 43 wire functions over plain strings and
returns JSON-rendered results, byte-identical to the source of truth
([port surface: tltk 1.11](https://github.com/panudetjt/tltk-rs/blob/main/PORTING-GAPS.md)).

## API

```rust
// Everything baked in (default feature "embedded-models"): the small blobs
// (thdict, sylseg, sylvar, phs, pos, ner, wordtier, textclass + the two text
// tables) are include_bytes!'d — no data dir needed.
let mut tltk = tltk_core::Tltk::embedded();
let seg = tltk.process("word_segment", "ฉันกินข้าว", "medium")?;

// Full 43-function surface: edu-segmentation, w2v similarity and dependency
// parsing need the large external blobs (edu/w2v/malt .bin.gz), loaded from a
// directory of converted models (see "Models" below).
let models = tltk_core::Models::load_dir(std::path::Path::new("data"))?;
let mut tltk = tltk_core::Tltk::new(models);
let dtree = tltk.process("malt_parser", "ฉัน กิน ข้าว", "medium")?;
```

`process(function, text, cutoff) -> Result<String, ProcessError>` mirrors the
validator/tltk API one-to-one: the 16 stichula validator names (`th2ipa`,
`th2roman`, `word_segment`, `syl_segment`, `segment`, `g2p`, `th2read`,
`spell_variants`, `spoonerism`, `pos_tag`, `ner_tag`, `similar_words`,
`similarity`, `outofgroup`, `analogy`, `plotwords`) plus text analysis,
transliteration variants, dictionary/spell utilities, tree helpers, w2v
extras and `malt_parser` / `malt_parser_wordlist`. The full wire inventory
lives in the [repo README](https://github.com/panudetjt/tltk-rs#readme).

For the MaltParser engine itself (inputs already tokenized + tagged), see the
[`malt-parser`](https://crates.io/crates/malt-parser) crate — `tltk-core`
wraps it with the segmentation/tagging front half.

## Parity

Parity is defined by goldens, not opinions: the 16 validator functions are
byte-identical to the live stichula validator; the rest are byte-identical to
the tltk 1.11 Python oracle over the checked-in corpora (the `w2v_*plot` /
`plotwords` renderers are disclosed-diff PNG). The verifier harness and the
golden files live in the repo — every algorithm change is held to
`byte-identical or it doesn't land`.

## Models

Model artifacts are deterministic binary conversions of the Python pickles
(`*.bin.gz`), produced by `tools/convert/convert.py` in the repo. They are not
distributed on crates.io (size + upstream licensing questions — see
[THIRD_PARTY_LICENSES](https://github.com/panudetjt/tltk-rs/blob/main/THIRD_PARTY_LICENSES.md));
build them from a tltk 1.11 install or take them from a tltk-rs checkout.

## License

MIT — see [LICENSE](LICENSE). The port and the converted model blobs derive
from TLTK (BSD-3-Clause, © 2018 Dept of Linguistics, Chulalongkorn
University); the bundled [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY) carries
the required notices, which must travel with redistributions.
