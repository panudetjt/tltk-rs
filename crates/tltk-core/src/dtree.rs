//! UD dependency-tree helpers — port of tltk 1.10 `nlp.py`
//! (`dg_print` L197, `dg_save` L212, `print_dtree` L228, `scan_dtree` L235,
//! `get_tree_depth` L264, `tree_max_depth` L286), byte-identical on the wire.
//!
//! # Wire adaptation
//!
//! Upstream `dg_save(dg, txt)` consumes an nltk `DependencyGraph` and emits
//! `{"sentence": ..., "words": [{"id","form","pos","head","deprel"}, ...]}`
//! (nltk always inserts the artificial root node `{"id":0,"form":"root",
//! "pos":"TOP","head":null,"deprel":null}` first). The parser itself stays
//! upstream-only (JVM MaltParser); on our wire the **text field carries the
//! parse-result JSON dict** (compact `json.dumps` of the `dg_save` output)
//! and every function operates on that dict. `dg_save` canonicalises: it
//! inserts the root node if absent and emits words in id order.
//! `get_tree_depth` takes the node id in the cutoff field (numeric string,
//! default `"0"` = the artificial root, whose depth is the tree height).
//!
//! Wire result shapes (JSON with `ensure_ascii=False`):
//! - `dg_print` → Python `repr` of the list of `(rel, "word/tag-i",
//!   "head/tag-j")` tuples, in stored word order, root word mutated to
//!   `"root"`; e.g. `[('nsubj', 'ฉัน/PRON-1', 'กิน/VERB-2'), ...]`.
//! - `print_dtree`/`scan_dtree` → the printed lines joined with `"\n"`,
//!   NO trailing newline (upstream `print()` adds one per line; probed and
//!   dropped on the wire). Line format:
//!   `"{id}:" + "--"*level + "{form} ({pos}, {deprel} - {head})"`, sorted by
//!   id, root node (id 0) prints nothing. `head`/`deprel` null render as
//!   `None` (Python `str(None)`).
//! - `get_tree_depth(ud, word_id)` → int; missing id → 0; no dependents →
//!   1; else max(child depths) + 1. Upstream quirk: this is depth counted
//!   from the node down, NOT the printed level — for a chain w1→…→w12,
//!   get_tree_depth(1) = 12 while get_tree_depth(12) = 1 (leaf), and
//!   get_tree_depth(0) = 13 (the root has all dependents).
//! - `tree_max_depth` → max over nodes with id > 0.
//!
//! # Probe record (era venv, tltk 1.10 oracle; graphs built with
//! `nltk.parse.dependencygraph.DependencyGraph` from Malt-Tab input)
//!
//! Probe 1 — `ฉัน/PRON → กิน/VERB (nsubj)` + `ข้าว/NOUN → กิน (obj)`,
//! `dg_save(dg, "ฉัน กิน ข้าว")`:
//! ```json
//! {"sentence":"ฉัน กิน ข้าว","words":[{"id":0,"form":"root","pos":"TOP","head":null,"deprel":null},{"id":1,"form":"ฉัน","pos":"PRON","head":2,"deprel":"nsubj"},{"id":2,"form":"กิน","pos":"VERB","head":0,"deprel":"root"},{"id":3,"form":"ข้าว","pos":"NOUN","head":2,"deprel":"obj"}]}
//! ```
//! `dg_print` → `[('nsubj', 'ฉัน/PRON-1', 'กิน/VERB-2'), ('root', 'กิน/VERB-2', 'root/TOP-0'), ('obj', 'ข้าว/NOUN-3', 'กิน/VERB-2')]`;
//! `print_dtree` printed text (trailing newline from print dropped on wire):
//! ```text
//! 1:----ฉัน (PRON, nsubj - 2)
//! 2:--กิน (VERB, root - 0)
//! 3:----ข้าว (NOUN, obj - 2)
//! ```
//! `get_tree_depth(ud,1)=1`, `get_tree_depth(ud,0)=3`, missing id → 0,
//! `tree_max_depth=2`.
//!
//! Probe 2 — multi-branch `ก/V(root) ← ข/N, ค/N(obj)`, `ง/N → ค (nc)`:
//! ```text
//! 1:--ก (V, root - 0)
//! 2:----ข (N, obj - 1)
//! 3:----ค (N, obj - 1)
//! 4:------ง (N, nc - 3)
//! ```
//! `get_tree_depth(ud,3)=2`, `get_tree_depth(ud,0)=4`,
//! `tree_max_depth=3` (root counts as a node).
//!
//! Probe 3 — single node `หนึ่ง/N(root)`:
//! `print_dtree` → `1:--หนึ่ง (N, root - 0)`;
//! `tree_max_depth=1`, `get_tree_depth(ud,1)=1`.
//!
//! Probe 4 — edge cases. Empty words list: `print_dtree` prints nothing
//! (wire: empty string), `tree_max_depth=0`, `get_tree_depth(ud,0)=0`.
//! Deep chain w1→…→w12 (each wi head = wi−1): `tree_max_depth=12`,
//! `get_tree_depth(ud,1)=12`, `get_tree_depth(ud,12)=1`.
//! Head cycle (1→2→1) and self-loop (1→1): `print_dtree`/`scan_dtree`
//! print only what is reachable from node 0 — in a pure cycle nothing is
//! reachable, so the oracle prints NOTHING and does NOT recurse (probed).
//! `get_tree_depth`/`tree_max_depth` raise
//! `RecursionError('maximum recursion depth exceeded in comparison')` —
//! mirrored on the wire as ok=false with exactly that error message
//! (budget-emulated recursion limit).

use serde::Deserialize;
use serde_json::Value;

use crate::{ProcessError, Session};

const RECURSION_MSG: &str = "maximum recursion depth exceeded in comparison";

#[derive(Deserialize, Clone)]
struct Word {
    id: i64,
    form: String,
    pos: String,
    head: Value,
    deprel: Value,
}

fn parse_ud(text: &str) -> Result<Vec<Word>, ProcessError> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| ProcessError(format!("invalid ud json: {}", e)))?;
    let words = v
        .get("words")
        .and_then(|w| w.as_array())
        .ok_or_else(|| ProcessError("invalid ud json: missing words".into()))?;
    serde_json::from_value(Value::Array(words.clone()))
        .map_err(|e| ProcessError(format!("invalid ud json: {}", e)))
}

impl Word {
    fn head_id(&self) -> Option<i64> {
        self.head.as_i64()
    }
    /// Python `str(v)` for a JSON scalar — strings unquoted, null → "None".
    fn py_str(v: &Value) -> String {
        match v {
            Value::Null => "None".into(),
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }
    /// JSON encoding of a scalar without key reordering concerns.
    fn py_json(v: &Value) -> String {
        match v {
            Value::Null => "null".into(),
            other => other.to_string(),
        }
    }
}

fn py_repr_list(tuples: &[(String, String, String)]) -> String {
    let inner: Vec<String> = tuples
        .iter()
        .map(|(a, b, c)| format!("('{}', '{}', '{}')", a, b, c))
        .collect();
    format!("[{}]", inner.join(", "))
}

/// `dg_print` — upstream consumes the DependencyGraph, but its output is
/// fully determined by the saved dict, so the wire takes the dict.
pub fn dg_print(_sess: &mut Session, text: &str) -> Result<String, ProcessError> {
    let mut words = parse_ud(text)?;
    for w in words.iter_mut() {
        if w.id == 0 {
            w.form = "root".into();
        }
    }
    let names: std::collections::HashMap<i64, String> = words
        .iter()
        .map(|w| (w.id, format!("{}/{}", w.form, w.pos)))
        .collect();
    let mut out = Vec::new();
    for w in &words {
        if w.id >= 1 {
            let h = match w.head_id() {
                Some(h) => h,
                None => return Err(ProcessError("KeyError".into())),
            };
            let hd = names
                .get(&h)
                .ok_or_else(|| ProcessError("KeyError".into()))?;
            out.push((
                Word::py_str(&w.deprel),
                format!("{}/{}-{}", w.form, w.pos, w.id),
                format!("{}-{}", hd, h),
            ));
        }
    }
    Ok(py_repr_list(&out))
}

/// `dg_save` — on our wire the input already is a parse dict (the
/// DependencyGraph step is upstream-only); canonicalise to the exact
/// upstream wire shape: root node 0 inserted if absent, words in id order,
/// fixed key order `id,form,pos,head,deprel`, `sentence` first.
pub fn dg_save(_sess: &mut Session, text: &str) -> Result<String, ProcessError> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| ProcessError(format!("invalid ud json: {}", e)))?;
    let mut words: Vec<Word> = parse_ud(text)?;
    if !words.iter().any(|w| w.id == 0) {
        words.push(Word {
            id: 0,
            form: "root".into(),
            pos: "TOP".into(),
            head: Value::Null,
            deprel: Value::Null,
        });
    }
    words.sort_by_key(|w| w.id);
    let mut out = String::from("{\"sentence\":");
    out.push_str(&match v.get("sentence") {
        Some(Value::String(s)) => serde_json::to_string(s).unwrap(),
        Some(Value::Null) | None => "null".into(),
        Some(other) => other.to_string(),
    });
    out.push_str(",\"words\":[");
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"id\":{},\"form\":{},\"pos\":{},\"head\":{},\"deprel\":{}}}",
            w.id,
            serde_json::to_string(&w.form).unwrap(),
            serde_json::to_string(&w.pos).unwrap(),
            Word::py_json(&w.head),
            Word::py_json(&w.deprel),
        ));
    }
    out.push_str("]}");
    Ok(out)
}

fn scan_dtree_rec(words: &[Word], node: i64, level: usize, out: &mut Vec<(i64, String)>) {
    if node > 0
        && let Some(w) = words.iter().find(|w| w.id == node)
    {
        out.push((
            node,
            format!(
                "{}{} ({}, {} - {})",
                "--".repeat(level),
                w.form,
                w.pos,
                Word::py_str(&w.deprel),
                Word::py_str(&w.head),
            ),
        ));
    }
    for w in words {
        if w.head_id() == Some(node) {
            scan_dtree_rec(words, w.id, level + 1, out);
        }
    }
}

fn scan_and_render(text: &str) -> Result<String, ProcessError> {
    let words = parse_ud(text)?;
    let mut entries: Vec<(i64, String)> = Vec::new();
    scan_dtree_rec(&words, 0, 0, &mut entries);
    // upstream prints `sorted(PRN_TREE, key=lambda k: k['id'])` — numeric id sort
    entries.sort_by_key(|(id, _)| *id);
    Ok(entries
        .into_iter()
        .map(|(id, msg)| format!("{}:{}", id, msg))
        .collect::<Vec<_>>()
        .join("\n"))
}

/// `print_dtree` — joined rendered lines (no trailing newline on the wire).
pub fn print_dtree(_sess: &mut Session, text: &str) -> Result<String, ProcessError> {
    scan_and_render(text)
}

/// `scan_dtree` — upstream builds the same PRN_TREE list; identical wire
/// output to print_dtree.
pub fn scan_dtree(_sess: &mut Session, text: &str) -> Result<String, ProcessError> {
    scan_and_render(text)
}

/// `get_tree_depth(ud, word_id)` — word_id from the cutoff field
/// (default "0").
pub fn get_tree_depth(
    _sess: &mut Session,
    text: &str,
    cutoff: &str,
) -> Result<String, ProcessError> {
    let word_id: i64 = if cutoff.is_empty() {
        0
    } else {
        cutoff
            .parse()
            .map_err(|_| ProcessError(format!("invalid word id: {}", cutoff)))?
    };
    let words = parse_ud(text)?;
    let mut budget = 1000i64;
    Ok(dtree_depth(&words, word_id, &mut budget)?.to_string())
}

fn dtree_depth(words: &[Word], word_id: i64, budget: &mut i64) -> Result<i64, ProcessError> {
    *budget -= 1;
    if *budget < 0 {
        // upstream: RecursionError on cyclic heads
        return Err(ProcessError(RECURSION_MSG.into()));
    }
    if !words.iter().any(|w| w.id == word_id) {
        return Ok(0);
    }
    if !words.iter().any(|w| w.head_id() == Some(word_id)) {
        return Ok(1);
    }
    let mut max_depth = 0i64;
    for w in words {
        if w.head_id() == Some(word_id) {
            let d = dtree_depth(words, w.id, budget)?;
            if d > max_depth {
                max_depth = d;
            }
        }
    }
    Ok(max_depth + 1)
}

/// `tree_max_depth` — max of get_tree_depth over nodes with id > 0.
pub fn tree_max_depth(_sess: &mut Session, text: &str) -> Result<String, ProcessError> {
    let words = parse_ud(text)?;
    let mut max_depth = 0i64;
    let mut budget = 1000i64;
    for w in &words {
        if w.id > 0 {
            let d = dtree_depth(&words, w.id, &mut budget)?;
            if d > max_depth {
                max_depth = d;
            }
        }
    }
    Ok(max_depth.to_string())
}
