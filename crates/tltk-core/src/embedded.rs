//! Embedded small-model bundle (feature "embedded-models").

use crate::model::Models;
use std::collections::HashMap;

macro_rules! blob {
    ($f:literal) => {
        (include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/", $f)).as_slice())
    };
}

/// The five small models ship in the binary; edu + w2v come from a loader.
pub fn models() -> Models {
    let mut blobs: HashMap<String, Vec<u8>> = HashMap::new();
    blobs.insert("thdict.bin".into(), blob!("thdict.bin.gz").to_vec());
    blobs.insert("sylseg.bin".into(), blob!("sylseg.bin.gz").to_vec());
    blobs.insert("sylvar.bin".into(), blob!("sylvar.bin.gz").to_vec());
    blobs.insert("phs.bin".into(), blob!("phs.bin.gz").to_vec());
    blobs.insert("pos.bin".into(), blob!("pos.bin.gz").to_vec());
    blobs.insert("ner.bin".into(), blob!("ner.bin.gz").to_vec());
    blobs.insert("sylrule.txt".into(), blob!("sylrule.txt").to_vec());
    blobs.insert("thaisyl.txt".into(), blob!("thaisyl.txt").to_vec());
    Models::from_blobs(&blobs)
}

/// Extend an embedded base with optional big models (edu/w2v/ner), e.g. from R2.
pub fn with_models(_base: Models, extra: &HashMap<String, Vec<u8>>) -> Models {
    let mut blobs: HashMap<String, Vec<u8>> = HashMap::new();
    blobs.insert("thdict.bin".into(), base_raw("thdict.bin.gz").to_vec());
    blobs.insert("sylseg.bin".into(), base_raw("sylseg.bin.gz").to_vec());
    blobs.insert("sylvar.bin".into(), base_raw("sylvar.bin.gz").to_vec());
    blobs.insert("phs.bin".into(), base_raw("phs.bin.gz").to_vec());
    blobs.insert("pos.bin".into(), base_raw("pos.bin.gz").to_vec());
    blobs.insert("ner.bin".into(), base_raw("ner.bin.gz").to_vec());
    blobs.insert("sylrule.txt".into(), base_raw("sylrule.txt").to_vec());
    blobs.insert("thaisyl.txt".into(), base_raw("thaisyl.txt").to_vec());
    for (k, v) in extra {
        // accept both "edu.bin" and "edu.bin.gz" spellings
        blobs.insert(k.trim_end_matches(".gz").to_string(), v.clone());
    }
    Models::from_blobs(&blobs)
}

fn base_raw(name: &str) -> &'static [u8] {
    match name {
        "thdict.bin.gz" => blob!("thdict.bin.gz"),
        "sylseg.bin.gz" => blob!("sylseg.bin.gz"),
        "sylvar.bin.gz" => blob!("sylvar.bin.gz"),
        "phs.bin.gz" => blob!("phs.bin.gz"),
        "pos.bin.gz" => blob!("pos.bin.gz"),
        "ner.bin.gz" => blob!("ner.bin.gz"),
        "sylrule.txt" => blob!("sylrule.txt"),
        "thaisyl.txt" => blob!("thaisyl.txt"),
        _ => panic!("unknown embedded blob {}", name),
    }
}
