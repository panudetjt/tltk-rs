# malt-parser

JVM-free Rust port of [MaltParser](https://www.maltparser.org/) 1.9.2's
parsing engine — the arc-eager (Nivre) transition system plus the MCSVM
decision model — bit-exact against the Java engine from a precompiled model
blob. This is the engine behind [tltk](https://pypi.org/project/tltk/)'s
`MaltParser()` and its `thamalt.mco` Thai dependency model; the port exists so
[tltk-rs](https://github.com/panudetjt/tltk-rs) needs no JVM at runtime.

Part of the tltk-rs workspace, but usable standalone.

## Scope

- **Inference only.** No training, no learning mode — the model comes
  precompiled. Decisions are deterministic; the same blob always produces the
  same parse (Java parity depends on it).
- **Input is already tokenized and POS-tagged**: one `&[(form, postag)]` pair
  list per sentence. Tagging upstream of this crate is the caller's job
  (tltk-core's `malt_parser` wire runs segmentation + POS tagging first).
- **Model blob**: `malt.bin.gz` — a deterministic gzip(JSON) conversion of
  upstream's `thamalt.mco`, produced by `tools/convert/convert.py malt` in the
  tltk-rs repo. Decode is strict on the schema contract
  (`format` / `format_version`) and tolerant of unknown extra keys.
- The module docs (docs.rs) map each module to the Java class it ports
  (`model.rs` ↔ the blob contract, `transition.rs` ↔ `ArcEager.java`,
  `scoring.rs` ↔ `MaltLiblinearModel.predict_one`, …).

## Usage

```rust
use std::io::Read;

let mut blob = Vec::new();
flate2::read::GzDecoder::new(std::fs::File::open("malt.bin.gz")?)
    .read_to_end(&mut blob)?;
let model = malt_parser::MaltModel::decode(&blob)?;

let tokens = vec![
    ("ฉัน".to_string(), "PRON".to_string()),
    ("กิน".to_string(), "VERB".to_string()),
    ("ข้าว".to_string(), "NOUN".to_string()),
];
let tree = model.parse_sentence(&tokens)?;
// Vec<MaltToken> — one { head, deprel } per input token:
// ฉัน -> head 2 / nsubj, กิน -> head 0 / ROOT, ข้าว -> head 2 / obj
```

`parse_sentence` is `&self` — load the model once and parse concurrently or
in a loop; nothing mutates.

## Measured against the original JVM engine

Same box, same model, same tagged inputs; 7 interleaved rounds, median of
round medians. Corpus: tltk's malt corpus (45 Thai sentences, 261 tokens),
tagged by the tltk 1.11 oracle's own POS tagger. Revision `11f3723` (clean);
WSL2, Intel i5-14400, rustc 1.99.1, OpenJDK 21.0.2, MaltParser 1.9.2.

| engine, model loaded once | per sentence (median) | peak RSS |
|---|---|---|
| **malt-parser (this crate)** | **0.097 ms** | **13 MiB** |
| MaltParser 1.9.2 JVM (batch, one process) | 7.33 ms | 92 MiB |

- Output cross-checked token-for-token (heads + deprels, all 45 sentences /
  261 tokens) against the JVM on the same inputs.
- The JVM number still amortizes process startup across 45 sentences; a bare
  JVM process on this box is 42.6 MiB before any model, so most of the JVM
  column is runtime, not data. The like-for-like engine gap is therefore
  larger than the table implies.
- Upstream tltk's actual user path is worse than either engine figure: NLTK
  spawns a fresh JVM and reloads the model **per sentence** (~0.31 s and
  ~86 MiB of fixed cost per call). End-to-end on the same box — segmentation
  + tagging + parsing + rendering — the tltk-rs wire is ~2.4 ms/sentence vs
  ~387 ms/sentence for the oracle, about **163×**.
- Cold model load: ~32 ms (gunzip + decode).

Single-box numbers — treat the ratios, not the absolute milliseconds, as the
finding. Raw per-round JSON and the environment block live in
`target/bench/malt-20261017T091741Z/` in the tltk-rs repo.

## License

MIT — see [LICENSE](LICENSE). This crate derives from MaltParser 1.9.2
(BSD-3-Clause, © 2007–2018 Johan Hall, Jens Nilsson and Joakim Nivre); the
bundled [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY) carries the required
notices, which must travel with redistributions.
