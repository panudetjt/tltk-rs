//! tltk-core — Rust rewrite of TLTK (Thai Language Toolkit) inference primitives.
//!
//! Faithful port of tltk 1.10 (PyPI), the version the stichula.org validator
//! runs — with one deliberate exception: `textana` carries tltk 1.11
//! semantics (1.11's only functional change is inside TextAna: LdvTTR ×100).
//! Model artifacts are converted from the Python pickles by
//! `tools/convert/convert.py`. Behaviour parity is defined by the validator API
//! and checked by `tltk-verifier`.

pub mod dtree;
pub mod chunk;
pub mod edu;
pub mod g2p;
pub mod model;
pub mod ner;
pub mod pos;
pub mod prelude;
pub mod spell;
pub mod sylseg;
pub mod textana;
pub mod textclass;
pub mod translit;
pub mod w2v;
pub mod wordseg;

use std::collections::HashMap;
use std::sync::Arc;

pub use model::Models;
pub use prelude::{SEG_SEP, SSEG_SEP, SYL_SEP, WORD_SEP};

/// The loaded model set (immutable, shareable).
pub struct Tltk {
    pub models: Arc<Models>,
}

/// Per-request mutable workspace. Python tltk keeps these as module globals,
/// mutated across calls in one process; the worker mirrors that with one
/// Session per isolate, the verifier uses one Session per request (== a fresh
/// Python process).
pub struct Session<'a> {
    pub m: &'a Models,

    // ---- compute_colloc / read_stat state (mutated once-ever per pair) ----
    pub total_word: i64,
    pub total_lex: i64,
    /// overlay for BiCount mutations (syllable string pair -> value delta vs base)
    pub bicount_ov: HashMap<(String, String), i64>,
    /// overlay for Count mutations (syllable -> delta)
    pub count_ov: HashMap<String, i64>,
    /// pairs already bumped by compute_colloc's `if < 1` arm
    pub colloc_bumped: std::collections::HashSet<(String, String)>,
    /// syllable form -> candidate pronunciations (per sylparse call; Python global PRONUN)
    pub pronun: HashMap<String, Vec<String>>,
}

#[derive(Debug)]
pub struct ProcessError(pub String);

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ProcessError {}

impl Tltk {
    pub fn new(models: Models) -> Tltk {
        Tltk { models: Arc::new(models) }
    }

    pub fn session(&self) -> Session<'_> {
        Session {
            m: &self.models,
            total_word: self.models.sylseg.total_word as i64,
            total_lex: self.models.sylseg.total_lex as i64,
            bicount_ov: HashMap::new(),
            count_ov: HashMap::new(),
            colloc_bumped: std::collections::HashSet::new(),
            pronun: HashMap::new(),
        }
    }
}

impl<'a> Session<'a> {
    /// Mirror of the stichula validator's process endpoint.
    /// Returns the `result` string (or the error message with success=false).
    pub fn process(&mut self, function: &str, text: &str, cutoff: &str) -> Result<String, ProcessError> {
        match function {
            "th2ipa" => Ok(self.th2ipa(text)),
            "th2roman" => Ok(self.th2roman(text)),
            "word_segment" => Ok(self.word_segment(text)),
            "syl_segment" => Ok(self.syl_segment(text)),
            "segment" => Ok(self.segment(text)),
            "g2p" => Ok(self.g2p(text)),
            "th2read" => self.th2read(text),
            "spell_variants" => Ok(spell::spell_variants(self, text)),
            "spoonerism" => spell::spoonerism(self, text),
            "pos_tag" => Ok(self.pos_tag_wire(text)),
            "ner_tag" => Ok(self.ner_tag(text)),
            "similar_words" => self.wire_similar_words(text, cutoff),
            "similarity" => self.wire_similarity(text),
            "outofgroup" => self.wire_outofgroup(text),
            "analogy" => self.wire_analogy(text),
            "plotwords" => self.wire_plotwords(text),
            "textana" => textana::textana(self, text),
            "dg_print" => dtree::dg_print(self, text),
            "dg_save" => dtree::dg_save(self, text),
            "print_dtree" => dtree::print_dtree(self, text),
            "scan_dtree" => dtree::scan_dtree(self, text),
            "get_tree_depth" => dtree::get_tree_depth(self, text, cutoff),
            "tree_max_depth" => dtree::tree_max_depth(self, text),
            "text_class" => textclass::text_class(self, text, cutoff),
            "txt2feat" => textclass::txt2feat(self, text, cutoff),
            "chunk" => Ok(self.chunk(text)),
            "tnc_tag" => self.tnc_tag(text),
            "th2ipa_all" => Ok(self.th2ipa_all(text)),
            "sylparse_all" => Ok(self.sylparse_all(text)),
            "wrd_len" => Ok(spell::wrd_len(text).to_string()),
            "match_engtext" => Ok(spell::match_engtext(text).to_string()),
            "known" => Ok(spell::wire_known(self, text)),
            "edits2" => Ok(spell::wire_edits2(text)),
            "check_thaidict" => Ok(spell::check_thaidict(self, text).to_string()),
            "reset_thaidict" => Ok(spell::reset_thaidict().to_string()),
            "read_thaidict" => Ok(spell::read_thaidict(text).to_string()),
            "w2v" => self.wire_w2v(text),
            "w2v_diffplot" => self.wire_w2v_diffplot(text),
            "w2v_dimplot" => self.wire_w2v_dimplot(text),
            "w2v_compare_color" => self.wire_w2v_compare_color(text),
            "compound" => self.wire_compound(text),
            other => Err(ProcessError(format!("Unknown function: {}", other))),
        }
    }
}

#[cfg(feature = "embedded-models")]
pub mod embedded;

#[cfg(feature = "embedded-models")]
impl Tltk {
    /// Small embedded model set (thdict/sylseg/sylvar/phs/pos/ner). edu + w2v are
    /// optional (worker loads them from R2; verifier from disk).
    pub fn embedded() -> Tltk {
        Tltk::new(embedded::models())
    }
}
