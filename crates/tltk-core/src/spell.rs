//! Spelling: edits1/candidates (Norvig), spell_variants, spoonerism.

use crate::g2p::pcompose;
use crate::prelude::*;
use crate::translit::phone_to_thai;
use crate::Session;

/// Thai alphabet used by edits1: chr(0x0E01)..=chr(0x0E4E).
fn letters() -> impl Iterator<Item = char> {
    (0x0E01u32..=0x0E4E).filter_map(char::from_u32)
}

/// Port of edits1 — order: deletes, transposes, replaces, inserts.
pub fn edits1(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    // deletes
    for i in 0..n {
        let mut s: String = chars[..i].iter().collect();
        s.extend(&chars[i + 1..]);
        out.push(s);
    }
    // transposes
    for i in 0..n.saturating_sub(1) {
        let mut s: String = chars[..i].iter().collect();
        s.push(chars[i + 1]);
        s.push(chars[i]);
        s.extend(&chars[i + 2..]);
        out.push(s);
    }
    // replaces
    for i in 0..n {
        for c in letters() {
            let mut s: String = chars[..i].iter().collect();
            s.push(c);
            s.extend(&chars[i + 1..]);
            out.push(s);
        }
    }
    // inserts
    for i in 0..=n {
        for c in letters() {
            let mut s: String = chars[..i].iter().collect();
            s.push(c);
            s.extend(&chars[i..]);
            out.push(s);
        }
    }
    out
}

pub fn spell_candidates(sess: &Session, word: &str) -> Vec<String> {
    edits1(word)
        .into_iter()
        .filter(|w| sess.m.thdict.contains(w))
        .collect()
}

/// Port of spell_variants(wrd, InDict='no', Karan='exclude') — server uses defaults.
pub fn spell_variants(sess: &mut Session, wrd: &str) -> String {
    let tmp = sess.g2p(wrd);
    let tmp = tmp.replace("<s/>", "");
    let tmp = tmp.replace('|', "~");
    let (syls, trans) = match tmp.split_once("<tr/>") {
        Some(p) => p,
        None => return "[]".to_string(),
    };
    let tran_lst: Vec<&str> = trans.split('~').collect();
    let syl_lst: Vec<&str> = syls.split('~').collect();
    let mut var: Vec<Vec<String>> = Vec::new();
    for i in 0..tran_lst.len() {
        if !tran_lst[i].is_empty() {
            // Karan == 'exclude'
            let raw = sess.m.sylvar.get_or_empty(tran_lst[i]);
            let filtered: Vec<String> = raw.into_iter().filter(|v| !v.contains('์')).collect();
            if syl_lst.len() > i && filtered.contains(&syl_lst[i].to_string()) {
                var.push(filtered);
            } else {
                var.push(vec![syl_lst.get(i).map(|s| s.to_string()).unwrap_or_default()]);
            }
        }
    }
    if var.is_empty() {
        return "[]".to_string();
    }
    // itertools.product: last list varies fastest
    let mut la: Vec<String> = var[0].clone();
    for next in &var[1..] {
        let mut combined: Vec<String> = Vec::with_capacity(la.len() * next.len());
        for a in &la {
            for b in next {
                let mut s = String::with_capacity(a.len() + b.len());
                s.push_str(a);
                s.push_str(b);
                combined.push(s);
            }
        }
        la = combined;
    }
    py_list_repr(&la)
}

/// Port of spoonerism. Python can crash inside PhoneToThai when the swapped
/// phone has an unknown final (server responds 500); that surfaces as Err here.
pub fn spoonerism(sess: &mut Session, word: &str) -> Result<String, crate::ProcessError> {
    static RE_KARAN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re_karan = RE_KARAN.get_or_init(|| regex::Regex::new(r"[ก-ฮุิ]์").unwrap());
    static RE_VOW: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static RE_TONE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re_vow = RE_VOW.get_or_init(|| regex::Regex::new(r".+?([aeiouxOU\@]+)").unwrap());
    let re_tone = RE_TONE.get_or_init(|| regex::Regex::new(r"[0-4]").unwrap());
    let crash = |k: String| crate::ProcessError(format!("KeyError: '{}'", k));

    let word = re_karan.replace_all(word, "").into_owned();
    let tmp = sess.g2p(&word);
    let tmp = tmp.replace("<s/>", "");
    let tmp = tmp.replace('|', "~");
    let (syls, trans) = match tmp.split_once("<tr/>") {
        Some(p) => p,
        None => return Ok("[]".to_string()),
    };
    let mut tran_lst: Vec<String> = trans.split('~').map(|s| s.to_string()).collect();
    if tran_lst.is_empty() {
        return Ok("[]".to_string());
    }
    tran_lst.pop(); // remove last empty
    let mut syl_lst: Vec<String> = syls.split('~').map(|s| s.to_string()).collect();

    let tranx0 = tran_lst.first().cloned().unwrap_or_default();
    let trany0 = tran_lst.last().cloned().unwrap_or_default();
    let (intc1, tone1) = pcompose(&tranx0);
    let (intc2, tone2) = pcompose(&trany0);

    // preserve initC: re.sub replaces ALL matches — both the real vowel run and
    // any later run after a "'" (leading-h phones like "ha1'kxxw0").
    let tranx = re_vow
        .replace_all(&tranx0, format!("{}$1", intc2))
        .into_owned();
    let trany = re_vow
        .replace_all(&trany0, format!("{}$1", intc1))
        .into_owned();
    let sylx = phone_to_thai(&tranx).map_err(crash)?;
    let syly = phone_to_thai(&trany).map_err(crash)?;

    let mut output: Vec<String> = Vec::new();
    if !syl_lst.is_empty() {
        syl_lst[0] = syly.clone();
        let last = syl_lst.len() - 1;
        syl_lst[last] = sylx.clone();
        let x = syl_lst.join("-");
        if !x.contains('?') {
            output.push(x);
        }
        // preserve initC and tone
        let tranx2 = re_tone.replace_all(&tranx, tone2.as_str()).into_owned();
        let trany2 = re_tone.replace_all(&trany, tone1.as_str()).into_owned();
        let sylx2 = phone_to_thai(&tranx2).map_err(crash)?;
        let syly2 = phone_to_thai(&trany2).map_err(crash)?;
        syl_lst[0] = syly2;
        let last = syl_lst.len() - 1;
        syl_lst[last] = sylx2;
        let x2 = syl_lst.join("-");
        if !output.contains(&x2) && !x2.contains('?') {
            output.insert(0, x2);
        }
    }
    Ok(py_list_repr(&output))
}
