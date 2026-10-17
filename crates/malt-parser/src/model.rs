//! Blob schema v1 (`malt.bin.gz` = gzip(JSON)) — decode contract owned by
//! the converter task; this module is strictly the reader side.

use serde::Deserialize;

use crate::symbol::SymbolTables;
use crate::{ParseError, transition::Transition};

/// A liblinear feature node as consumed by `predict_one`.
#[derive(Debug, Clone, Copy)]
pub struct FeatureNode {
    /// 1-based liblinear index.
    pub index: i32,
    pub value: f64,
}

#[derive(Debug, Deserialize)]
pub struct MaltModel {
    format: String,
    format_version: u32,
    #[allow(dead_code)]
    model_name: String,
    #[allow(dead_code)]
    maltparser_version: String,
    #[allow(dead_code)]
    source_mco_sha256: String,
    options: std::collections::HashMap<String, String>,
    #[serde(default)]
    #[allow(dead_code)]
    data_format_xml: String,
    #[serde(default)]
    #[allow(dead_code)]
    feature_spec_xml: String,
    feature_strings: Vec<String>,
    symbol_tables: Vec<SymbolTableJson>,
    decision_models: Vec<DecisionModelJson>,

    // Derived state (not deserialized).
    #[serde(skip)]
    tables: SymbolTables,
    #[serde(skip)]
    pub allow_root: bool,
    #[serde(skip)]
    pub allow_reduce: bool,
    #[serde(skip)]
    pub root_label_code: i32,
}

#[derive(Debug, Deserialize)]
pub struct SymbolTableJson {
    pub name: String,
    pub header: Vec<String>,
    pub symbols: Vec<SymbolJson>,
}

#[derive(Debug, Deserialize)]
pub struct SymbolJson {
    pub code: i32,
    pub symbol: String,
}

#[derive(Debug, Deserialize)]
pub struct DecisionModelJson {
    #[allow(dead_code)]
    pub name: String,
    pub feature_map: FeatureMapJson,
    pub liblinear: LiblinearJson,
}

#[derive(Debug, Deserialize)]
pub struct FeatureMapJson {
    #[allow(dead_code)]
    pub next_index: u32,
    /// `(<packed key as decimal string>, <1-based index>)`, sorted by index.
    pub entries: Vec<(String, i32)>,
}

#[derive(Debug, Deserialize)]
pub struct LiblinearJson {
    #[allow(dead_code)]
    pub solver: String,
    pub nr_class: usize,
    pub labels: Vec<i32>,
    pub nr_feature: u32,
    pub bias: f64,
    /// Sparse rows: `[<0-based col>, <value>]`, col ascending, exact zeros dropped.
    pub w: Vec<Vec<(u32, f64)>>,
}

/// A compiled decision model: packed-key lookup + scoring data.
pub struct DecisionModel {
    /// packed key -> 1-based liblinear index
    pub feature_map: std::collections::HashMap<u64, i32>,
    pub labels: Vec<i32>,
    pub bias: f64,
    pub nr_feature: u32,
    /// Per 1-based index (row 0 unused): sparse (0-based col, value) ascending.
    pub w: Vec<Vec<(u32, f64)>>,
}

impl MaltModel {
    /// Decode a gzip(JSON) blob. Strict on `format`/`format_version`;
    /// unknown extra keys are ignored (serde default).
    pub fn decode(bytes: &[u8]) -> Result<MaltModel, ParseError> {
        let json: MaltModel = serde_json::from_reader(
            flate2::read::GzDecoder::new(std::io::Cursor::new(bytes)),
        )
        .map_err(|e| ParseError::Decode(format!("blob decode failed: {e}")))?;

        if json.format != "tltk-malt" {
            return Err(ParseError::Decode(format!(
                "unknown blob format {:?} (expected \"tltk-malt\")",
                json.format
            )));
        }
        if json.format_version != 1 {
            return Err(ParseError::Decode(format!(
                "unsupported blob format_version {} (expected 1)",
                json.format_version
            )));
        }
        if json.decision_models.len() != 1 {
            return Err(ParseError::Decode(format!(
                "expected exactly 1 decision model, found {}",
                json.decision_models.len()
            )));
        }
        let mut model = json;
        model.tables = SymbolTables::new(&model.symbol_tables)
            .map_err(|e| ParseError::Decode(format!("symbol tables: {e}")))?;
        model.allow_root = model.options.get("nivre_allow_root").map(|v| v == "true").unwrap_or(true);
        model.allow_reduce = model.options.get("nivre_allow_reduce").map(|v| v == "true").unwrap_or(false);
        // `graph.root_label` = ROOT; its DEPREL-table code is needed when
        // linking leftover trees to the root at the end of parsing.
        let root_symbol = model.options.get("graph_root_label").map(|s| s.as_str()).unwrap_or("ROOT");
        model.root_label_code = model
            .tables
            .code_of("DEPREL", root_symbol)
            .map_err(|e| ParseError::Decode(format!("root label: {e}")))?;
        Ok(model)
    }

    pub fn feature_strings(&self) -> &[String] {
        &self.feature_strings
    }

    pub fn decision_model(&self) -> DecisionModel {
        let dm = &self.decision_models[0];
        let mut feature_map = std::collections::HashMap::with_capacity(dm.feature_map.entries.len());
        for (key, index) in &dm.feature_map.entries {
            feature_map.insert(key.parse::<u64>().expect("valid decimal key"), *index);
        }
        // liblinear's `Model.w` is class-major `double[nr_class][n]` and the
        // schema-v1 contract carries it that way; scoring iterates it
        // feature-major (`w[index-1]`, cols = class). Canonicalize both
        // shapes here — mathematically identical (per-class accumulations
        // keep their feature order either way), and older dev blobs were
        // dumped transposed.
        let nr_class = dm.liblinear.labels.len();
        let w = if dm.liblinear.w.len() == nr_class {
            let mut wt: Vec<Vec<(u32, f64)>> = vec![Vec::new(); dm.liblinear.nr_feature as usize];
            for (c, row) in dm.liblinear.w.iter().enumerate() {
                for &(col, v) in row {
                    wt[col as usize].push((c as u32, v));
                }
            }
            wt
        } else {
            dm.liblinear.w.clone()
        };
        DecisionModel {
            feature_map,
            labels: dm.liblinear.labels.clone(),
            bias: dm.liblinear.bias,
            nr_feature: dm.liblinear.nr_feature,
            w,
        }
    }

    /// `LWClassifier.predict`: for feature position i (1-based), look up
    /// `(i, code)`; misses are omitted. Values are 1 (all our features are
    /// single STRING features with value 1). Built in position order — the
    /// f64 sum order of scoring depends on this.
    pub fn build_feature_nodes(&self, codes: &[Option<(i32, String)>]) -> Vec<FeatureNode> {
        let dm = self.decision_model();
        let mut nodes = Vec::new();
        for (pos, value) in codes.iter().enumerate() {
            let Some((code, _symbol)) = value else { continue };
            // Feature positions are 1-based on the wire (Lib.predict iterates
            // i = 1..size; FeatureMap.addIndex packs that i).
            let key = (((pos + 1) as u64) << 48) | (*code as u64);
            if let Some(&index) = dm.feature_map.get(&key) {
                nodes.push(FeatureNode { index, value: 1.0 });
            }
        }
        nodes
    }

    /// Split the predicted combined decision symbol on `~` into transition
    /// + deprel and look up their codes (`CombinedTableContainer.split`).
    pub fn split_decision(&self, symbol: &str) -> (Transition, i32) {
        let (trans_sym, deprel_sym) = match symbol.split_once('~') {
            Some((t, d)) => (t, d),
            None => (symbol, ""),
        };
        let trans = Transition::from_symbol(trans_sym);
        let deprel = self
            .tables
            .code_of("DEPREL", deprel_sym)
            .unwrap_or(-1);
        (trans, deprel)
    }

    pub fn tables(&self) -> &SymbolTables {
        &self.tables
    }
}
