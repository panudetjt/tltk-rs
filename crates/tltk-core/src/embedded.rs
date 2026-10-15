//! Embedded small-model bundle (feature "embedded-models").

use crate::model::Models;
use std::collections::HashMap;

macro_rules! blob {
    ($f:literal) => {
        (include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/", $f)).as_slice())
    };
}

/// The raw embedded blob map (six .bin.gz + sylrule.txt + thaisyl.txt), no decoding.
pub fn blobs() -> HashMap<String, Vec<u8>> {
    let mut blobs: HashMap<String, Vec<u8>> = HashMap::new();
    blobs.insert("thdict.bin".into(), blob!("thdict.bin.gz").to_vec());
    blobs.insert("sylseg.bin".into(), blob!("sylseg.bin.gz").to_vec());
    blobs.insert("sylvar.bin".into(), blob!("sylvar.bin.gz").to_vec());
    blobs.insert("phs.bin".into(), blob!("phs.bin.gz").to_vec());
    blobs.insert("pos.bin".into(), blob!("pos.bin.gz").to_vec());
    blobs.insert("ner.bin".into(), blob!("ner.bin.gz").to_vec());
    blobs.insert("sylrule.txt".into(), blob!("sylrule.txt").to_vec());
    blobs.insert("thaisyl.txt".into(), blob!("thaisyl.txt").to_vec());
    blobs
}

/// The six small models ship in the binary; edu + w2v come from a loader.
pub fn models() -> Models {
    Models::from_blobs(&blobs())
}

/// Extend an embedded base with optional big models (edu/w2v/ner), e.g. from R2.
pub fn with_models(extra: &HashMap<String, Vec<u8>>) -> Models {
    let mut b = blobs();
    for (k, v) in extra {
        // accept both "edu.bin" and "edu.bin.gz" spellings
        b.insert(k.trim_end_matches(".gz").to_string(), v.clone());
    }
    Models::from_blobs(&b)
}

/// Same as `with_models` but returns Result on decode failure.
pub fn try_with_models(extra: &HashMap<String, Vec<u8>>) -> Result<Models, String> {
    let mut b = blobs();
    for (k, v) in extra {
        b.insert(k.trim_end_matches(".gz").to_string(), v.clone());
    }
    Models::try_from_blobs(&b)
}

pub fn base_raw(name: &str) -> Result<&'static [u8], String> {
    match name {
        "thdict.bin.gz" => Ok(blob!("thdict.bin.gz")),
        "sylseg.bin.gz" => Ok(blob!("sylseg.bin.gz")),
        "sylvar.bin.gz" => Ok(blob!("sylvar.bin.gz")),
        "phs.bin.gz" => Ok(blob!("phs.bin.gz")),
        "pos.bin.gz" => Ok(blob!("pos.bin.gz")),
        "ner.bin.gz" => Ok(blob!("ner.bin.gz")),
        "sylrule.txt" => Ok(blob!("sylrule.txt")),
        "thaisyl.txt" => Ok(blob!("thaisyl.txt")),
        _ => Err(format!("unknown embedded blob {}", name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_models_extra_edu() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = std::path::Path::new(manifest_dir).join("data/edu.bin.gz");
        let data = std::fs::read(&path).expect("edu.bin.gz should exist");
        let mut extra = HashMap::new();
        extra.insert("edu.bin.gz".to_string(), data);
        let m = with_models(&extra);
        assert!(m.edu.is_some(), "m.edu should be Some");
        assert!(m.thdict.contains("แมว"), "embedded thdict should contain 'แมว'");
    }
}
