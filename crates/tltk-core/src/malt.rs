//! `malt_parser` / `malt_parser_wordlist` — port of tltk 1.10 `nlp.py`
//! `MaltParser()` (L141) / `MaltParser_wordlist()` (L160) with the JVM
//! engine replaced by the `malt-parser` crate port (bit-exact, verify-gated
//! 8/8 sentences against the real Java engine).
//!
//! # Pipeline (upstream shape, engine swapped)
//!
//! 1. `match_Engtext(txt)` → `''` on English-only input.
//! 2. `pos_tag(txt)` / `pos_tag_wordlist(wlst)` → (word, POS) pairs.
//! 3. `remove_sp_join` — flattens ALL sentence groups into one list, dropping
//!    the `("<s/>", "PUNCT")` sentinel and empty / space / tab-containing
//!    forms (tuple-equality on the sentinel only; a token whose form was
//!    mapped to `"<s/>"` by pos_tag survives with its own tag).
//! 4. parse (arc-eager + MCSVM_CS, the `malt-parser` crate).
//! 5. `dg_save(dg, sentence)` — `{"sentence", "words":[{id,form,pos,head,
//!    deprel}...]}` with node 0 = artificial root.
//!
//! # words order (the probe-record trap)
//!
//! Upstream iterates `dg.nodes` — an nltk `DependencyGraph` `defaultdict`.
//! nltk's parser touches `nodes[index]` then `nodes[head]` per CoNLL row,
//! so first-mention order is: node 0, then for each token i (id order): i,
//! then head(i) if unseen. This reproduces the oracle's scrambled arrays
//! (e.g. ids `1,3,2,4,5,7,6,8`) byte-for-byte — verified against
//! `malt_output.conll` rows and the nltk defaultdict reproduction.

use crate::pos;
use crate::spell;
use crate::{ProcessError, Session};

/// `remove_sp_join` (nlp.py L182): flatten sentence groups, drop sentinel +
/// empty / space / tab forms.
fn remove_sp_join(sents: &[Vec<(String, String)>]) -> Vec<(String, String)> {
    sents.iter()
        .flatten()
        .filter(|(w, p)| !(w == "<s/>" && p == "PUNCT") && !w.is_empty() && w != " " && !w.contains('\t'))
        .cloned()
        .collect()
}

/// `MaltParser(txt)` (nlp.py L141).
pub fn malt_parser(sess: &mut Session, text: &str) -> Result<String, ProcessError> {
    if spell::match_engtext(text) == "True" {
        // upstream returns Python `''`; ref_dump json.dumps makes the wire
        // answer the two-character `""`.
        return Ok("\"\"".to_string());
    }
    let lst = pos::pos_tag(sess, text);
    let pairs = remove_sp_join(&lst);
    parse_and_save(sess, &pairs, text)
}

/// `MaltParser_wordlist(wlst)` (nlp.py L160): `text` carries the word list
/// newline-joined; spaces are stripped per item; the sentence field is the
/// space-removed items concatenated.
pub fn malt_parser_wordlist(sess: &mut Session, text: &str) -> Result<String, ProcessError> {
    let wlst: Vec<String> = text.split('\n').map(|s| s.replace(' ', "")).collect();
    if spell::match_engtext(&wlst.join(" ")) == "True" {
        return Ok("\"\"".to_string());
    }
    // pos_tag_wordlist (nlp.py L?): tag the given words directly — no
    // re-segmentation, no sentinel, change_tag applied.
    let tagged: Vec<(String, String)> = pos::perceptron_tag(sess, &wlst)
        .into_iter()
        .map(|(w, p)| {
            let p = pos::change_tag(&w, &p);
            (w, p)
        })
        .collect();
    let pairs = remove_sp_join(&[tagged]);
    let sentence = wlst.join("");
    parse_and_save(sess, &pairs, &sentence)
}

fn parse_and_save(sess: &mut Session, pairs: &[(String, String)], sentence: &str) -> Result<String, ProcessError> {
    let model = sess
        .m
        .malt
        .as_ref()
        .ok_or_else(|| ProcessError("malt model missing (malt.bin.gz not loaded)".into()))?;
    let parsed = model.parse_sentence(pairs).map_err(|e| ProcessError(e.to_string()))?;

    // nltk DependencyGraph first-mention order (see module doc).
    let mut order: Vec<usize> = vec![0];
    for (i, t) in parsed.iter().enumerate() {
        let id = i + 1;
        if !order.contains(&id) {
            order.push(id);
        }
        let h = t.head as usize;
        if !order.contains(&h) {
            order.push(h);
        }
    }

    let mut out = String::from("{\"sentence\":");
    out.push_str(&serde_json::to_string(sentence).unwrap());
    out.push_str(",\"words\":[");
    for (k, &node) in order.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        if node == 0 {
            out.push_str("{\"id\":0,\"form\":\"root\",\"pos\":\"TOP\",\"head\":null,\"deprel\":null}");
        } else {
            let t = &parsed[node - 1];
            let (form, pos_tag) = &pairs[node - 1];
            out.push_str(&format!(
                "{{\"id\":{},\"form\":{},\"pos\":{},\"head\":{},\"deprel\":{}}}",
                node,
                serde_json::to_string(form).unwrap(),
                serde_json::to_string(pos_tag).unwrap(),
                t.head,
                serde_json::to_string(&t.deprel).unwrap(),
            ));
        }
    }
    out.push_str("]}");
    Ok(out)
}
