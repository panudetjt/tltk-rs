//! Chunk segmentation and TNC corpus-tagging pipelines (port of tltk 1.10
//! `nlp.chunk` and `nlp.TNC_tag`).
//!
//! # AC1 Oracle Probes (verified against tltk 1.10 python oracle)
//!
//! Probed using `$HOME/.local/share/tltk-rs/venv-segment-oracle/bin/python`:
//!
//! 1. `chunk('แมวกินปลาทะเล')`
//!    Output: `""`
//!    Quirk: Upstream `chunk()` only flushes accumulated text `syl_seq` into
//!    the output when the EDU model predicts `<u/>`. Unlike `nlp.segment()`,
//!    `nlp.chunk()` does NOT append a trailing space (`txt = txt + ' '`) prior
//!    to syllable segmentation. If no `<u/>` boundary is predicted, the loop
//!    terminates without flushing and returns the empty string `""`.
//!
//! 2. `chunk('นายสมชาย ใจดี เดินทางไปทำงานที่กรุงเทพฯ')`
//!    Output: `"<NEp>นาย/NOUN|สม/PROPN|ชาย/NOUN|</NEp><s/>/PUNCT|ใจดี/ADV|<u/>"`
//!    Quirk: Shows `<NEp>` entity grouping and flushing on the first `<u/>`
//!    boundary after "ใจดี". The remainder ("เดินทางไปทำงานที่กรุงเทพฯ") has no
//!    subsequent `<u/>` boundary and is left unflushed in `syl_seq` when the loop
//!    finishes.
//!
//! 3. `chunk('เด็กๆ กำลังวิ่งเล่นในสนามหญ้าอย่างสนุกสนาน ขณะที่คุณพ่อคุณแม่นั่งดูอยู่ใต้ต้นไม้ใหญ่')`
//!    Output: `"เด็กๆ/NOUN|<s/><s/>/SYM|กำลัง/AUX|วิ่ง/VERB|เล่น/VERB|ใน/ADP|สนามหญ้า/NOUN|อย่าง/PART|สนุกสนาน/ADV|<u/>ขณะที่/SCONJ|คุณ/NOUN|พ่อคุณ/NOUN|แม่/NOUN|นั่ง/VERB|ดู/VERB|อยู่/ADV|ใต้/ADP|ต้นไม้/NOUN|ใหญ่/ADJ|<u/>"`
//!    Non-empty output with multiple `<u/>` segments demonstrating multi-sentence
//!    EDU chunking, word segmentation, POS tagging, and NER packing.
//!
//! 4. `chunk('123 456')`
//!    Output: `"123<s/>456/NUM|<u/>"`
//!
//! 5. `chunk('ทดสอบระบบ 123 test\nบรรทัดที่สอง 456 end')`
//!    Output: `"ทดสอบ/NOUN|ระบบ/NOUN|<u/>123<s/>test/NUM|<u/>บรรทัด/NOUN|ที่/ADP|สอง/NUM|<u/>456<s/>end/NUM|<u/>"`
//!
//! 6. `TNC_tag('แมวกินปลา')`
//!    Output: `"<w tran=\"mxxw0kin0plaa0\" POS=\"NOUN\">แมวกินปลา</w><s/>\n"`
//!    Quirk: Notice the trailing `<s/>\n`.
//!
//! 7. `TNC_tag('นายสมชาย ใจดี')`
//!    Output: `"<w tran=\"naaj0som4chaaj0\" POS=\"NOUN\">นายสมชาย</w><s/><w tran=\"caj0dii0\" POS=\"CCONJ\">ใจดี</w><s/>\n"`
//!
//! 8. `TNC_tag('')`
//!    Output: `"\n"`
//!    Quirk: Empty input produces a single newline `"\n"`.

use crate::edu::{edu_features, rf_predict, vectorize};
use crate::ner::{ner, pack_ner};
use crate::pos::{change_tag, perceptron_tag};
use crate::{ProcessError, Session};
use std::sync::LazyLock;

static RE1: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"([^~])<s/>").unwrap());
static RE2: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"<s/>([^~])").unwrap());
static RE_DIG: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"~[0-9\.\,]+~").unwrap());
static RE_FOR: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r#"~[a-zA-Z0-9\\\/\?'"\(\)\.]+~"#).unwrap());
static RE_NUM: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"([0-9])<s/>/NUM").unwrap());

/// Port of nlp.pos_tag_wordlist — tags a pre-split word list.
pub fn pos_tag_wordlist(sess: &Session, sent: &[String]) -> Vec<(String, String)> {
    let mut results = Vec::with_capacity(sent.len());
    for (w, pos) in perceptron_tag(sess, sent) {
        let pos = change_tag(&w, &pos);
        results.push((w, pos));
    }
    results
}

impl<'a> Session<'a> {
    /// Port of nlp.chunk (tltk 1.10).
    ///
    /// Pipeline:
    /// syl_segment -> <s/> insertion -> digit/foreign substitution ->
    /// EDU segmentation -> per-EDU wordseg_colloc + pos_tag + change_tag + ner + pack_ner
    /// -> join with <u/> -> re.sub NUM/PUNCT fixup.
    pub fn chunk(&mut self, txt: &str) -> String {
        if self.m.edu.is_none() {
            return String::new();
        }
        let sylseg_out = self.syl_segment(txt);
        let sylseg_out = sylseg_out.replace(' ', "<s/>");
        let sylseg_out = RE1.replace_all(&sylseg_out, "${1}~<s/>").into_owned();
        let sylseg_out = RE2.replace_all(&sylseg_out, "<s/>~${1}").into_owned();

        let sylcopy = RE_DIG.replace_all(&sylseg_out, "~DIGIT~").into_owned();
        let sylcopy = RE_FOR.replace_all(&sylcopy, "~FOREIGN~").into_owned();

        let parcopy: Vec<&str> = sylcopy.split('~').collect();
        let par: Vec<&str> = sylseg_out.split('~').collect();

        let mut tags: Vec<&str> = Vec::with_capacity(par.len());
        for index in 0..par.len() {
            let feats = edu_features(&parcopy, index);
            let row = vectorize(self, &feats);
            let (cls, _) = rf_predict(self, &row);
            tags.push(if cls == 0 { "" } else { "<u/>" });
        }

        let mut outs = String::new();
        let mut syl_seq = String::new();
        for (w, t) in par.iter().zip(tags.iter()) {
            if *t == "<u/>" {
                let out = self.wordseg_colloc(&syl_seq);
                if !out.is_empty() {
                    let mut sent: Vec<String> = out.split('|').map(|s| s.to_string()).collect();
                    sent.retain(|a| !a.is_empty());
                    let mut tag_result = Vec::with_capacity(sent.len());
                    for (w, pos) in perceptron_tag(self, &sent) {
                        let pos = change_tag(&w, &pos);
                        tag_result.push((w, pos));
                    }
                    let out = pack_ner(&ner(self, &tag_result));
                    outs.push_str(&out);
                }
                syl_seq.clear();
                outs.push_str("<u/>");
            } else {
                syl_seq.push_str(w);
                syl_seq.push('~');
            }
        }

        RE_NUM.replace_all(&outs, "${1}/NUM|<s/>/PUNCT").into_owned()
    }

    /// Port of nlp.TNC_tag (tltk 1.10).
    ///
    /// Pipeline:
    /// For each line, for each whitespace-separated token:
    /// g2p -> split('<tr/>') -> pos_tag_wordlist -> wrap `<w tran=... POS=...>` -> `<s/>`
    /// Trailing `\n` at the end of the text.
    pub fn tnc_tag(&mut self, txt: &str) -> Result<String, ProcessError> {
        let parx: Vec<&str> = txt.split('\n').collect();
        let mut tag = String::new();
        for par in parx {
            for sent in par.split_whitespace() {
                let tx = self.g2p(sent);
                let parts: Vec<&str> = tx.split("<tr/>").collect();
                if parts.len() != 2 {
                    return Err(ProcessError(if parts.len() > 2 {
                        "too many values to unpack (expected 2)".to_string()
                    } else {
                        "not enough values to unpack (expected 2, got 1)".to_string()
                    }));
                }
                let wx = parts[0];
                let trx = parts[1];
                let w_lst: Vec<String> = wx.split('|').map(|s| s.to_string()).collect();
                let tr_lst: Vec<&str> = trx.split('|').collect();
                let wpos_lst = pos_tag_wordlist(self, &w_lst);
                for i in 0..w_lst.len() {
                    let tr = tr_lst
                        .get(i)
                        .copied()
                        .unwrap_or("")
                        .replace(['~', '\'', '^'], "");
                    let wrd = w_lst[i].replace(['~', '\'', '^'], "");
                    tag.push_str("<w tran=\"");
                    tag.push_str(&tr);
                    tag.push_str("\" POS=\"");
                    tag.push_str(&wpos_lst[i].1);
                    tag.push_str("\">");
                    tag.push_str(&wrd);
                    tag.push_str("</w>");
                }
                tag.push_str("<s/>");
            }
        }
        tag.push('\n');
        Ok(tag)
    }
}

#[cfg(test)]
mod tests {
    use crate::embedded;
    use crate::Tltk;
    use std::collections::HashMap;

    fn test_session() -> Tltk {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = std::path::Path::new(manifest_dir).join("data/edu.bin.gz");
        let data = std::fs::read(&path).expect("edu.bin.gz should exist");
        let mut extra = HashMap::new();
        extra.insert("edu.bin.gz".to_string(), data);
        Tltk::new(embedded::with_models(&extra))
    }

    #[test]
    fn test_chunk_quirk_empty() {
        let tltk = test_session();
        let mut sess = tltk.session();
        assert_eq!(sess.chunk("แมวกินปลาทะเล"), "");
    }

    #[test]
    fn test_chunk_probed_cases() {
        let tltk = test_session();
        let mut sess = tltk.session();
        assert_eq!(
            sess.chunk("นายสมชาย ใจดี เดินทางไปทำงานที่กรุงเทพฯ"),
            "<NEp>นาย/NOUN|สม/PROPN|ชาย/NOUN|</NEp><s/>/PUNCT|ใจดี/ADV|<u/>"
        );
        assert_eq!(
            sess.chunk("เด็กๆ กำลังวิ่งเล่นในสนามหญ้าอย่างสนุกสนาน ขณะที่คุณพ่อคุณแม่นั่งดูอยู่ใต้ต้นไม้ใหญ่"),
            "เด็กๆ/NOUN|<s/><s/>/SYM|กำลัง/AUX|วิ่ง/VERB|เล่น/VERB|ใน/ADP|สนามหญ้า/NOUN|อย่าง/PART|สนุกสนาน/ADV|<u/>ขณะที่/SCONJ|คุณ/NOUN|พ่อคุณ/NOUN|แม่/NOUN|นั่ง/VERB|ดู/VERB|อยู่/ADV|ใต้/ADP|ต้นไม้/NOUN|ใหญ่/ADJ|<u/>"
        );
        assert_eq!(sess.chunk("123 456"), "123<s/>456/NUM|<u/>");
    }

    #[test]
    fn test_tnc_tag_probed_cases() {
        let tltk = test_session();
        let mut sess = tltk.session();
        assert_eq!(
            sess.tnc_tag("แมวกินปลา").unwrap(),
            "<w tran=\"mxxw0kin0plaa0\" POS=\"NOUN\">แมวกินปลา</w><s/>\n"
        );
        assert_eq!(
            sess.tnc_tag("นายสมชาย ใจดี").unwrap(),
            "<w tran=\"naaj0som4chaaj0\" POS=\"NOUN\">นายสมชาย</w><s/><w tran=\"caj0dii0\" POS=\"CCONJ\">ใจดี</w><s/>\n"
        );
        assert_eq!(sess.tnc_tag("").unwrap(), "\n");
    }
}
