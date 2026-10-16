//! TextClass — text classification into level (L1..L4) or genre (10 genres),
//! and txt2feat feature vector extraction.
//!
//! Mirrors `nlp.TextClass` and `nlp.txt2feat` from tltk 1.10.
//! UDParse is pinned to "none" (same deviation as textana).

use crate::textana::{self, py_float_repr};
use crate::ProcessError;
use crate::Session;

/// The 129 sorted feature names returned by `txt2feat(Option="name")`.
pub const TXT2FEAT_NAMES: &[&str] = &[
    "CrfCNL",
    "CrfCTL",
    "CrfCVL",
    "CrfCWL",
    "DesChC",
    "DesClusterC",
    "DesCorrFinalC",
    "DesCorrToneC",
    "DesDoubVowelC",
    "DesEduC",
    "DesEduL",
    "DesEduLd",
    "DesInCorrFinalC",
    "DesInCorrToneC",
    "DesKaranC",
    "DesLeadC",
    "DesNElC",
    "DesNEoC",
    "DesNEpC",
    "DesNegC",
    "DesPC",
    "DesPL",
    "DesSpC",
    "DesSymbC",
    "DesTNCt1C",
    "DesTNCt2C",
    "DesTNCt3C",
    "DesTNCt4C",
    "DesTTC1",
    "DesTTC2",
    "DesTTC3",
    "DesTTC4",
    "DesTotT",
    "DesTotW",
    "DesWrdL",
    "DesWrdLd",
    "LdvTTR",
    "SynDepth",
    "UDROOT",
    "UDX",
    "UDacl",
    "UDadvcl",
    "UDadvmod",
    "UDamod",
    "UDappos",
    "UDaux",
    "UDcase",
    "UDcc",
    "UDccomp",
    "UDclf",
    "UDcompound",
    "UDconj",
    "UDcop",
    "UDcsubj",
    "UDdep",
    "UDdet",
    "UDdiscourse",
    "UDdislocated",
    "UDexpl",
    "UDfixed",
    "UDflat",
    "UDgoeswith",
    "UDiobj",
    "UDlist",
    "UDmark",
    "UDmwe",
    "UDnmod",
    "UDnsubj",
    "UDnummod",
    "UDobj",
    "UDobl",
    "UDorphan",
    "UDparataxis",
    "UDpunct",
    "UDreparandum",
    "UDvocative",
    "UDxcomp",
    "WrdADJ",
    "WrdADP",
    "WrdADV",
    "WrdAUX",
    "WrdArf",
    "WrdBasicM",
    "WrdBasicT",
    "WrdC",
    "WrdCCONJ",
    "WrdCF",
    "WrdCluster",
    "WrdCorrFinal",
    "WrdCorrTone",
    "WrdDET",
    "WrdDoubVowel",
    "WrdF",
    "WrdFrmComp",
    "WrdFrmSing",
    "WrdFrmTran",
    "WrdINTJ",
    "WrdInCorrFinal",
    "WrdInCorrTone",
    "WrdKaran",
    "WrdLead",
    "WrdNEl",
    "WrdNEo",
    "WrdNEp",
    "WrdNOUN",
    "WrdNUM",
    "WrdNeg",
    "WrdPART",
    "WrdPRON",
    "WrdPROPN",
    "WrdPUNCT",
    "WrdSCONJ",
    "WrdSYM",
    "WrdSemOpaq",
    "WrdSemSemi",
    "WrdSemSimp",
    "WrdSemTran",
    "WrdTNCt1",
    "WrdTNCt2",
    "WrdTNCt3",
    "WrdTNCt4",
    "WrdTTC1",
    "WrdTTC2",
    "WrdTTC3",
    "WrdTTC4",
    "WrdTfidf",
    "WrdTncDisp",
    "WrdTtcDisp",
    "WrdVERB",
];

/// Classify text into level (L1..L4) or genre (10 genres) or both.
/// Cutoff carries Classifier: "level" (default) | "genre" | "all".
pub fn text_class(sess: &mut Session, text: &str, cutoff: &str) -> Result<String, ProcessError> {
    let tc_model = sess.m.textclass.as_ref().ok_or_else(|| {
        ProcessError("textclass model missing (text_class requires textclass.bin)".into())
    })?;

    let ana = textana::compute_textana(sess, text)?;

    let classifier = match cutoff {
        "genre" => "genre",
        "all" => "all",
        _ => "level",
    };

    let level_pred = if classifier == "level" || classifier == "all" {
        let mut x = [0.0f64; 64];
        let n = tc_model.level.feature_names.len();
        for (i, name) in tc_model.level.feature_names.iter().enumerate() {
            x[i] = ana.features.get(name.as_str()).copied().unwrap_or(0.0);
        }
        Some(tc_model.level.predict(&x[..n]))
    } else {
        None
    };

    let genre_pred = if classifier == "genre" || classifier == "all" {
        let mut x = [0.0f64; 128];
        let n = tc_model.genre.feature_names.len();
        for (i, name) in tc_model.genre.feature_names.iter().enumerate() {
            x[i] = ana.features.get(name.as_str()).copied().unwrap_or(0.0);
        }
        Some(tc_model.genre.predict(&x[..n]))
    } else {
        None
    };

    match (level_pred, genre_pred) {
        (Some(l), None) => Ok(format!("{{\"level\":\"{}\"}}", l)),
        (None, Some(g)) => Ok(format!("{{\"genre\":\"{}\"}}", g)),
        (Some(l), Some(g)) => {
            // sorted keys: "genre" before "level"
            Ok(format!("{{\"genre\":\"{}\",\"level\":\"{}\"}}", g, l))
        }
        _ => unreachable!(),
    }
}

/// Create feature vector from text input.
/// If cutoff == "name": return list of 129 feature names.
/// If cutoff == "value" (default): return list of 129 feature values under UDParse="none".
pub fn txt2feat(sess: &mut Session, text: &str, cutoff: &str) -> Result<String, ProcessError> {
    if cutoff == "name" {
        let mut out = String::from("[");
        for (i, name) in TXT2FEAT_NAMES.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('"');
            out.push_str(name);
            out.push('"');
        }
        out.push(']');
        return Ok(out);
    }

    let ana = textana::compute_textana(sess, text)?;
    let mut out = String::with_capacity(2048);
    out.push('[');
    for (i, name) in TXT2FEAT_NAMES.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let val = ana.features.get(*name).copied().unwrap_or(0.0);
        out.push_str(&py_float_repr(val));
    }
    out.push(']');
    Ok(out)
}
