//! NER — port of nlp.ner_tag / ner / pack_ner / word2features, with a
//! linear-chain CRF viterbi decoder over converted crfsuite weights.
//!
//! ⚠ The upstream ner_tagger.pklz ships a FileResource without the trained
//! model bytes (the crfsuite temp file was deleted before pickling), so the
//! exact model artifact is unrecoverable from the package. With `ner.bin`
//! absent, tags default to 'O' — byte-exact parity for entity-free inputs.
//! Recover the model from the server box and run
//! `convert.py --ner-model <file>` to enable full tagging.

use crate::pos::pos_tag;
use crate::Session;
use std::collections::HashMap;

/// nlp.wrd_len
fn wrd_len(word: &str) -> &'static str {
    let n = word.chars().count();
    if n > 20 {
        "l"
    } else if n > 10 {
        "m"
    } else {
        "s"
    }
}

/// nlp.word2features — insertion-ordered dict; values: str → "k=v" column,
/// float/bool → "k" column (sklearn-crfsuite ItemSequence semantics).
pub fn word2features(sent: &[(String, String)], i: usize) -> Vec<String> {
    let word = &sent[i].0;
    let postag = &sent[i].1;
    let mut f: Vec<String> = Vec::new();
    // sklearn-crfsuite attribute naming: str values -> "key:value",
    // bool True / numeric -> bare "key"; float 'bias' 1.0 -> "bias" (no suffix)
    let push_str = |f: &mut Vec<String>, k: &str, v: &str| {
        if v.is_empty() {
            f.push(k.to_string());
        } else {
            f.push(format!("{}:{}", k, v));
        }
    };
    let push_bool = |f: &mut Vec<String>, k: &str, v: bool| {
        if v {
            f.push(k.to_string());
        }
    };

    push_str(&mut f, "bias", ""); // {'bias': 1.0} -> bare "bias"
    push_str(&mut f, "word", word);
    push_str(&mut f, "postag", postag);
    push_str(&mut f, "len", wrd_len(word));
    if i > 0 {
        push_str(&mut f, "-1:word", &sent[i - 1].0);
        push_str(&mut f, "-1:postag", &sent[i - 1].1);
    } else {
        push_bool(&mut f, "BOS", true);
    }
    if i > 2 {
        push_str(&mut f, "-2:word", &sent[i - 2].0);
        push_str(&mut f, "-2:postag", &sent[i - 2].1);
    }
    if i < sent.len() - 1 {
        push_str(&mut f, "+1:word", &sent[i + 1].0);
        push_str(&mut f, "+1:postag", &sent[i + 1].1);
    } else {
        push_bool(&mut f, "EOS", true);
    }
    f
}

/// CRF viterbi over state (attribute,label) and transition weights.
/// crfsuite scores in log space: sum of attribute weights + transition weights.
fn crf_viterbi(
    labels: &[String],
    transitions: &HashMap<(String, String), f64>,
    attributes: &HashMap<(String, String), f64>,
    seq: &[Vec<String>],
) -> Vec<String> {
    let n = seq.len();
    if n == 0 {
        return Vec::new();
    }
    let _li: HashMap<&str, usize> =
        labels.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();
    let k = labels.len();
    let neg_inf = f64::NEG_INFINITY;

    let mut score = vec![vec![neg_inf; k]; n];
    let mut bp = vec![vec![0usize; k]; n];
    for (t, feats) in seq.iter().enumerate() {
        for (yi, y) in labels.iter().enumerate() {
            let mut best = neg_inf;
            let mut bestp = 0usize;
            for (yp, _) in labels.iter().enumerate() {
                let prev = if t == 0 {
                    0.0
                } else {
                    transitions
                        .get(&(labels[yp].clone(), y.clone()))
                        .copied()
                        .unwrap_or(0.0)
                };
                let s = if t == 0 { prev } else { score[t - 1][yp] + prev };
                if s > best {
                    best = s;
                    bestp = yp;
                }
            }
            let mut st = best;
            for feat in feats {
                st += attributes.get(&(feat.clone(), y.clone())).copied().unwrap_or(0.0);
            }
            score[t][yi] = st;
            bp[t][yi] = bestp;
        }
    }
    // best final state
    let mut yi = 0usize;
    for y in 1..k {
        if score[n - 1][y] > score[n - 1][yi] {
            yi = y;
        }
    }
    let mut out = vec![String::new(); n];
    for t in (0..n).rev() {
        out[t] = labels[yi].clone();
        if t > 0 {
            yi = bp[t][yi];
        }
    }
    out
}

/// nlp.ner — returns (word, pos, tag) triples.
pub fn ner(sess: &mut Session, sent: &[(String, String)]) -> Vec<(String, String, String)> {
    let mut r = Vec::with_capacity(sent.len());
    match &sess.m.ner {
        Some(model) => {
            let feats: Vec<Vec<String>> = (0..sent.len()).map(|i| word2features(sent, i)).collect();
            let tags = crf_viterbi(
                &model.labels,
                &model
                    .transitions
                    .iter()
                    .map(|(a, b, w)| ((a.clone(), b.clone()), *w))
                    .collect(),
                &model
                    .attributes
                    .iter()
                    .map(|(a, b, w)| ((a.clone(), b.clone()), *w))
                    .collect(),
                &feats,
            );
            for (i, (w, p)) in sent.iter().enumerate() {
                r.push((w.clone(), p.clone(), tags[i].clone()));
            }
        }
        None => {
            for (w, p) in sent {
                r.push((w.clone(), p.clone(), "O".to_string()));
            }
        }
    }
    r
}

/// nlp.pack_ner — word/pos stream with <NEp>/<NEo>/<NEl> wrappers.
pub fn pack_ner(sent: &[(String, String, String)]) -> String {
    let mut out = String::new();
    let mut flag = "O";
    for (w, p, t) in sent {
        let is_p = t.ends_with("-P") && (t.starts_with('B') || t.starts_with('I'));
        let is_o = t.ends_with("-O") && (t.starts_with('B') || t.starts_with('I'));
        let is_l = t.ends_with("-L") && (t.starts_with('B') || t.starts_with('I'));
        if is_p && flag == "O" {
            out.push_str(&format!("<NEp>{}/{}|", w, p));
            flag = "NEp";
            continue;
        } else if t == "O" && flag == "NEp" {
            out.push_str(&format!("</NEp>{}/{}|", w, p));
            flag = "O";
            continue;
        } else if is_o && flag == "O" {
            out.push_str(&format!("<NEo>{}/{}|", w, p));
            flag = "NEo";
            continue;
        } else if t == "O" && flag == "NEo" {
            out.push_str(&format!("</NEo>{}/{}|", w, p));
            flag = "O";
            continue;
        } else if is_l && flag == "O" {
            out.push_str(&format!("<NEl>{}/{}|", w, p));
            flag = "NEl";
            continue;
        } else if t == "O" && flag == "NEl" {
            out.push_str(&format!("</NEl>{}/{}|", w, p));
            flag = "O";
            continue;
        } else {
            out.push_str(&format!("{}/{}|", w, p));
        }
    }
    if flag == "NEp" {
        out.push_str("</NEp>");
    } else if flag == "NEo" {
        out.push_str("</NEo>");
    } else if flag == "NEl" {
        out.push_str("</NEl>");
    }
    out
}

impl<'a> Session<'a> {
    /// nlp.ner_tag — pos tags + CRF + pack + post substitutions.
    pub fn ner_tag(&mut self, input: &str) -> String {
        let mut tag_result: Vec<(String, String)> = Vec::new();
        for lst in pos_tag(self, input) {
            tag_result.extend(lst);
        }
        // nlp.ner drops a trailing ' ' word
        if tag_result.last().map(|(w, _)| w.as_str()) == Some(" ") {
            tag_result.pop();
        }
        let packed = pack_ner(&ner(self, &tag_result));
        let out = packed.replace("<s/>", " ");
        static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re = RE.get_or_init(|| regex::Regex::new(r"/[A-Z]+\|").unwrap());
        re.replace_all(&out, "").into_owned()
    }
}
