# MaltParser Model Blob Format (Schema v1)

Specification of the deserialized format produced by `tools/convert/malt_dumper/DumpMco.java`
and saved as `crates/tltk-core/data/malt.bin.gz` (gzip with no mtime, compresslevel=6).

This document serves as the implementation contract for the Rust MaltParser engine in `tltk-core`.

---

## 1. Overview & Top-Level Schema

The binary payload is a gzipped UTF-8 JSON document (`schema v1`).

```json
{
  "format": "tltk-malt",
  "format_version": 1,
  "model_name": "thamalt",
  "maltparser_version": "1.9.2",
  "source_mco_sha256": "f15ec776f1614e2428c386db4404ba94f9b7e573960cb29aa7ce76fc423d7e02",
  "options": {
    "2planar_reduceonswitch": "false",
    "covington_allow_root": "true",
    "covington_allow_shift": "false",
    "graph_head_rules": "",
    "graph_root_label": "ROOT",
    "guide_classitem_separator": "~",
    "guide_data_split_column": "",
    "guide_data_split_structure": "",
    "guide_data_split_threshold": "50",
    "guide_decision_settings": "T.TRANS+A.DEPREL",
    "guide_features": "NivreEager.xml",
    "guide_kbest": "-1",
    "guide_learner": "class org.maltparser.ml.lib.LibLinear",
    "input_charset": "UTF-8",
    "input_format": "conllx.xml",
    "lib_options": "",
    "lib_save_instance_files": "false",
    "multiplanar_planar_root_handling": "normal",
    "nivre_allow_reduce": "false",
    "nivre_allow_root": "true",
    "output_charset": "UTF-8",
    "planar_acyclicity": "true",
    "planar_connectedness": "none",
    "planar_no_covered_roots": "false",
    "pproj_covered_root": "none",
    "pproj_lifting_order": "shortest",
    "pproj_marking_strategy": "none",
    "singlemalt_guide_model": "class org.maltparser.parser.guide.SingleGuide",
    "singlemalt_null_value": "one",
    "singlemalt_parsing_algorithm": "class org.maltparser.parser.algorithm.nivre.NivreArcEagerFactory",
    "singlemalt_propagation": "",
    "singlemalt_use_partial_tree": "false"
  },
  "data_format_xml": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<dataformat name=\"conllx\">\n\t<column name=\"ID\" category=\"INPUT\" type=\"INTEGER\"/>\n\t<column name=\"FORM\" category=\"INPUT\" type=\"STRING\"/>\n\t<column name=\"LEMMA\" category=\"INPUT\" type=\"STRING\"/>\n\t<column name=\"CPOSTAG\" category=\"INPUT\" type=\"STRING\"/>\n\t<column name=\"POSTAG\" category=\"INPUT\" type=\"STRING\"/>\n\t<column name=\"FEATS\" category=\"INPUT\" type=\"STRING\"/>\n\t<column name=\"HEAD\" category=\"HEAD\" type=\"INTEGER\"/>\n\t<column name=\"DEPREL\" category=\"DEPENDENCY_EDGE_LABEL\" type=\"STRING\"/>\n\t<column name=\"PHEAD\" category=\"IGNORE\" type=\"INTEGER\" default=\"_\"/>\n\t<column name=\"PDEPREL\" category=\"IGNORE\" type=\"STRING\" default=\"_\"/>\n</dataformat>",
  "feature_spec_xml": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<featuremodels>\n\t<featuremodel name=\"nivreeager\">\n\t\t<feature>InputColumn(POSTAG, Stack[0])</feature>\n...",
  "feature_strings": [
    "InputColumn(POSTAG, Stack[0])",
    "InputColumn(POSTAG, Input[0])",
    "..."
  ],
  "symbol_tables": [
    {
      "name": "FORM",
      "header": ["1", "1", "one"],
      "symbols": [
        {"code": 1, "symbol": "นาย"},
        "..."
      ]
    },
    "..."
  ],
  "decision_models": [
    {
      "name": "odm0",
      "feature_map": {
        "next_index": 13282,
        "entries": [
          ["281474976710657", 1],
          "..."
        ]
      },
      "liblinear": {
        "solver": "MCSVM_CS",
        "nr_class": 62,
        "labels": [1, 2, 3, 4, 5, 6, "...", 62],
        "nr_feature": 13282,
        "bias": 0.0,
        "w": [
          [
            [0, 0.32379464005260744],
            "..."
          ],
          "..."
        ]
      }
    }
  ]
}
```

---

## 2. Field Specifications and Java Origins

| JSON Field | Type | Java Origin in `.mco` / Class | Description |
|---|---|---|---|
| `format` | string | Hardcoded | Always `"tltk-malt"` |
| `format_version` | integer | Hardcoded | `1` |
| `model_name` | string | Filename prefix of `.mco` | `"thamalt"` |
| `maltparser_version` | string | Hardcoded / info file | `"1.9.2"` |
| `source_mco_sha256` | string | SHA-256 of input `.mco` | 64-char hex string |
| `options` | object | `savedoptions.sop` | Map of `task_param` -> verbatim string value |
| `data_format_xml` | string | `conllx.xml` | Raw XML string of CoNLL-X data format |
| `feature_spec_xml` | string | `NivreEager.xml` | Raw XML string of feature model specification |
| `feature_strings` | array of string | `<feature>` tags in `NivreEager.xml` | 22 feature expressions in spec order |
| `symbol_tables` | array of object | `symboltables.sym` via `HashSymbolTableHandler` | 15 symbol tables sorted alphabetically by name |
| `symbol_tables[i].name` | string | `HashSymbolTable.getName()` | Table name (e.g. `FORM`, `POSTAG`, `T.TRANS+A.DEPREL`) |
| `symbol_tables[i].header` | array of string | `category`, `type`, `nullValueStrategy` | Informational header fields: `[category, type, nullValueStrategy]` |
| `symbol_tables[i].symbols` | array of object | `codeSymbolMap` in `HashSymbolTable` | Entries `{"code": int, "symbol": str}`, sorted by `code` ascending |
| `decision_models` | array of object | `odm0.liblinear.{map,moo}` | Decision models (single model `odm0` in thamalt) |
| `decision_models[i].name` | string | Filename prefix | `"odm0"` |
| `feature_map.next_index` | integer | `FeatureMap.featureCounter` | Next available feature index (`13282`) |
| `feature_map.entries` | array of `[str, int]` | `FeatureMap.map` | Packed `[key_decimal_string, index]`, sorted by `index` ascending |
| `liblinear.solver` | string | `MaltLiblinearModel.solverType` | Solver identifier (`"MCSVM_CS"`) |
| `liblinear.nr_class` | integer | `MaltLiblinearModel.nr_class` | Number of decision classes (`62`) |
| `liblinear.labels` | array of int | `MaltLiblinearModel.labels` | Class label identifiers |
| `liblinear.nr_feature` | integer | `MaltLiblinearModel.nr_feature` | Number of input features (`13282`) |
| `liblinear.bias` | float | `MaltLiblinearModel.bias` | Bias parameter (`0.0`) |
| `liblinear.w` | array of array of `[int, float]` | `MaltLiblinearModel.w` (transposed) | Sparse rows per class: `[col, value]` sorted by `col` ascending |

---

## 3. Critical Conventions & Invariants

### 3.1 Feature Map Key Packing (`position << 48 | code`)
- Keys in `feature_map.entries` represent `(position << 48) | code`, where:
  - `position` (high 16 bits): 1-based index (1 to 22) of the feature in `feature_strings`.
  - `code` (low 48 bits): integer symbol code returned by the corresponding symbol table.
- Keys are serialized as **decimal strings** (e.g. `"281474976710657"`), not integers, to avoid precision loss across languages/tooling without 64-bit integer JSON support.
- In Rust, parse the decimal string to `u64`. When looking up features during parsing, compute `((pos as u64) << 48) | (code as u64)`.
- `entries` are strictly sorted by `index` ascending (1 .. 13281).

### 3.2 Feature Specification Order
- `feature_strings` has exactly 22 features, matching the 22 `<feature>` tags in `NivreEager.xml`.
- Feature positions in `feature_map` correspond 1-to-1 with `feature_strings` using 1-based indexing:
  - Position 1 = `InputColumn(POSTAG, Stack[0])`
  - Position 2 = `InputColumn(POSTAG, Input[0])`
  - ...
  - Position 22 = `InputColumn(FORM, head(Stack[0]))`

### 3.3 Sparse Weight Matrix `w`
- Java's `MaltLiblinearModel` internally holds `double[][] w` indexed as `[feature_index][class_index]`.
- Schema v1 transposes this to **per-class sparse rows**:
  - `w.len() == nr_class` (62 classes).
  - Each class `c` has a list of `[col, value]` pairs where `col` is the 0-based feature column (`0 <= col < nr_feature`), and `value` is non-zero.
  - Columns within each class row are **strictly ascending**.
  - All exact `0.0` entries are dropped.
- Floating point values are printed with Java's `Double.toString()` (shortest round-trip representation), ensuring exact round-trip reconstruction to `f64` in Rust.

### 3.4 Symbol Tables
- Loaded using MaltParser's `HashSymbolTableHandler.load(InputStreamReader)` directly from `symboltables.sym`.
- Tables are sorted alphabetically by `name` for deterministic ordering across conversions.
- Within each table, `symbols` are sorted strictly by `code` ascending.
- Tables include all input columns (`FORM`, `POSTAG`, etc.) as well as the transition/action table `T.TRANS+A.DEPREL`.
- String symbols preserve UTF-8 Thai text and escaped special characters (`\n`, `\t`, `\r`, `\"`, `\\`).

---

## 4. Truncated Schema Sample

```json
{
  "format": "tltk-malt",
  "format_version": 1,
  "model_name": "thamalt",
  "maltparser_version": "1.9.2",
  "source_mco_sha256": "f15ec776f1614e2428c386db4404ba94f9b7e573960cb29aa7ce76fc423d7e02",
  "options": {
    "2planar_reduceonswitch": "false",
    "covington_allow_root": "true",
    "..." : "..."
  },
  "data_format_xml": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<dataformat name=\"conllx\">\n\t<column name=\"ID\" category=\"INPUT\" type=\"INTEGER\"/>\n...",
  "feature_spec_xml": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<featuremodels>\n\t<featuremodel name=\"nivreeager\">\n\t\t<feature>InputColumn(POSTAG, Stack[0])</feature>\n...",
  "feature_strings": [
    "InputColumn(POSTAG, Stack[0])",
    "..."
  ],
  "symbol_tables": [
    {
      "name": "CPOSTAG",
      "header": ["1", "1", "one"],
      "symbols": [
        {"code": 1, "symbol": "NOUN"},
        "..."
      ]
    },
    {
      "name": "FORM",
      "header": ["1", "1", "one"],
      "symbols": [
        {"code": 1, "symbol": "นาย"},
        "..."
      ]
    },
    {
      "name": "T.TRANS+A.DEPREL",
      "header": ["-1", "1", "one"],
      "symbols": [
        {"code": 1, "symbol": "RE"},
        {"code": 2, "symbol": "SH"},
        {"code": 3, "symbol": "LA~flat"},
        "..."
      ]
    },
    "..."
  ],
  "decision_models": [
    {
      "name": "odm0",
      "feature_map": {
        "next_index": 13282,
        "entries": [
          ["281474976710657", 1],
          ["281474976710658", 2],
          "..."
        ]
      },
      "liblinear": {
        "solver": "MCSVM_CS",
        "nr_class": 62,
        "labels": [1, 2, 3, 4, 5, "...", 62],
        "nr_feature": 13282,
        "bias": 0.0,
        "w": [
          [
            [0, 0.32379464005260744],
            [1, -0.06381387660293308],
            "..."
          ],
          "..."
        ]
      }
    }
  ]
}
```

## Note on `bias` rows

The dumper enumerates `w` rows bounded by `nr_feature`. For a model trained with `bias >= 1`, liblinear appends one extra weight column (the bias feature); such a row would be silently dropped here. `thamalt` has `bias = 0.0` so the shipped blob is unaffected — revisit the dumper before reusing it on a bias-trained model.
