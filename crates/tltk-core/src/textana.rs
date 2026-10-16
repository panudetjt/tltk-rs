//! TextAna — port of nlp.TextAna / WordAna / SylAna / ARF.
//!
//! The ONLY function where tltk 1.11 differs from 1.10 (LdvTTR uses ×100, not
//! ×1000), so this module is 1.11-semantics while the rest of the crate stays
//! 1.10. Only the default surface is ported: TextOption="par",
//! WordOption="colloc", UDParse="none" — the validator/toolbox call shape
//! (MaltParser UD depth is out of scope, SynDepth stays 0.0 like upstream's
//! default path).
//!
//! Wire format: the txtfeats dict serialized like
//! `json.dumps(txtfeats, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`
//! — keys sorted (Python text-dependent dict order is NOT reproducible), ints
//! as ints, floats in CPython repr form (`py_float_repr`), `wrd`/`wrd_arf` as
//! sorted sub-objects, `deprel` always `{}` (UDParse=none). Upstream crash
//! cases (empty input → StatisticsError from mean/stdev, single-utterance
//! input → stdev needs 2 points, zero analysable words → ZeroDivisionError at
//! LdvTTR) return Err here and `ok=false` in goldens, like the validator 500.

use crate::ner::{ner, pack_ner};
use crate::pos::{change_tag, perceptron_tag};
use crate::ProcessError;
use crate::Session;
use std::collections::{BTreeMap, HashMap, HashSet};

// ---- Python-representation helpers -----------------------------------------

/// CPython `repr(float)` / `json.dumps` float form: shortest round-trip digits,
/// fixed notation while -4 < decpt <= 16, else scientific with sign and >=2
/// exponent digits. Integral floats carry ".0" (the ADD_DOT_0 repr flag).
pub(crate) fn py_float_repr(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-Infinity" } else { "Infinity" }.to_string();
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let e = format!("{:e}", v); // shortest digits, e.g. "-7.058823529411765e2"
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.trim_start_matches('-').chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1; // value = 0.<digits> × 10^decpt
    let nd = digits.len() as i32;
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if decpt > -4 && decpt <= 16 {
        if decpt <= 0 {
            out.push_str("0.");
            for _ in 0..-decpt {
                out.push('0');
            }
            out.push_str(&digits);
        } else if decpt >= nd {
            out.push_str(&digits);
            for _ in 0..(decpt - nd) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..decpt as usize]);
            out.push('.');
            out.push_str(&digits[decpt as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if nd > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        let e = decpt - 1;
        if e < 0 {
            out.push_str(&format!("-{:02}", -e));
        } else {
            out.push_str(&format!("+{:02}", e));
        }
    }
    out
}

/// statistics.mean over ints: int result when the sum divides evenly (the
/// Fraction collapses to denominator 1), else the correctly-rounded float.
enum PyNum {
    Int(i64),
    Float(f64),
}

fn py_num_emit(n: &PyNum) -> String {
    match n {
        PyNum::Int(i) => i.to_string(),
        PyNum::Float(f) => py_float_repr(*f),
    }
}

fn py_mean_ints(data: &[i64]) -> PyNum {
    let n = data.len() as i64;
    let s: i64 = data.iter().sum();
    if s % n == 0 {
        PyNum::Int(s / n)
    } else {
        PyNum::Float(s as f64 / n as f64)
    }
}

/// statistics.stdev over ints: sqrt of the EXACT SS/(n-1) rational, evaluated
/// as fl(fl(num)/fl(den)) — the same two correctly-rounded steps as
/// `float(Fraction)` ** 0.5 inside the statistics module. Magnitudes here stay
/// far below 2^53, so the int→f64 casts are exact and this is bit-identical.
fn py_stdev_ints(data: &[i64]) -> Result<f64, ()> {
    let n = data.len();
    if n < 2 {
        return Err(());
    }
    let n = n as i128;
    let s1: i128 = data.iter().map(|&x| x as i128).sum();
    let s2: i128 = data.iter().map(|&x| x as i128 * x as i128).sum();
    let num = n * s2 - s1 * s1; // SS = num / n, exact
    let den = n * (n - 1); // sample variance = num / den
    Ok(((num as f64) / (den as f64)).sqrt())
}

/// nlp.ARF — average reduced frequency of w in the token stream.
fn arf(words: &[String], w: &str) -> f64 {
    let idxs: Vec<usize> = words
        .iter()
        .enumerate()
        .filter(|(_, x)| x.as_str() == w)
        .map(|(i, _)| i)
        .collect();
    if idxs.is_empty() {
        return 0.0;
    }
    let v = words.len() as f64 / idxs.len() as f64;
    let mut sum = 0.0f64;
    for i in 1..idxs.len() {
        sum += v.min((idxs[i] - idxs[i - 1]) as f64);
    }
    sum += v.min((idxs[0] + words.len() - idxs[idxs.len() - 1]) as f64);
    sum / v
}

// ---- shared regexes ---------------------------------------------------------

static RE_SYMBOL: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn re_symbol() -> &'static regex::Regex {
    // upstream class r'[$#:;^@&%\*\+\-\(\)\"\']'
    RE_SYMBOL.get_or_init(|| regex::Regex::new(r##"[$#:;^@&%*+\-()\"']"##).unwrap())
}

static RE_NE_TAG: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn re_ne_tag() -> &'static regex::Regex {
    RE_NE_TAG.get_or_init(|| regex::Regex::new(r"</?NE[lop]>").unwrap())
}

static RE_NL_RUN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn re_nl_run() -> &'static regex::Regex {
    RE_NL_RUN.get_or_init(|| regex::Regex::new(r"\n+").unwrap())
}

static RE_TONE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn re_tone() -> &'static regex::Regex {
    // upstream class [\u0E48|\u0E49|\u0E4A|\u0E4B] — the pipes are literal class members
    RE_TONE.get_or_init(|| regex::Regex::new("[\u{0E48}|\u{0E49}|\u{0E4A}|\u{0E4B}]").unwrap())
}

static RE_FINAL_C: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn re_final_c() -> &'static regex::Regex {
    RE_FINAL_C.get_or_init(|| regex::Regex::new(r"[กงดนมยว]$").unwrap())
}

static RE_NOT_FINAL_C: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn re_not_final_c() -> &'static regex::Regex {
    RE_NOT_FINAL_C.get_or_init(|| regex::Regex::new(r"[^กงดนมยว]$").unwrap())
}

static RE_FINAL_PH: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn re_final_ph() -> &'static regex::Regex {
    RE_FINAL_PH.get_or_init(|| regex::Regex::new(r"[ktnNjw][0-9]$").unwrap())
}

static RE_DOUB_V: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn re_doub_v() -> &'static regex::Regex {
    RE_DOUB_V.get_or_init(|| regex::Regex::new(r"[iuU]a").unwrap())
}

static RE_KARAN_SUB: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn re_karan_sub() -> &'static regex::Regex {
    RE_KARAN_SUB.get_or_init(|| regex::Regex::new(r"([ก-ฮ])?([ก-ฮ]?์[ก-ฮ]?)").unwrap())
}

static RE_RLW: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn re_rlw() -> &'static regex::Regex {
    RE_RLW.get_or_init(|| regex::Regex::new(r".[rlw]").unwrap())
}

static RE_PHONE_SPLIT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn re_phone_split() -> &'static regex::Regex {
    RE_PHONE_SPLIT
        .get_or_init(|| regex::Regex::new(r"^(.*[^aeioux@OU])([aeioux@OU]+)(.*)([0-9])").unwrap())
}

// ---- WordAna / SylAna -------------------------------------------------------

#[derive(Default)]
struct WordAnaOut {
    syl: i64,
    corrtone: i64,
    incorrtone: i64,
    corrfinal: i64,
    incorrfinal: i64,
    karan: i64,
    cluster: i64,
    lead: i64,
    doubvowel: i64,
}

/// nlp.SylAna, reduced to what WordAna counts: (leading, cluster).
/// Python raises AttributeError when the phone has no trailing tone digit —
/// mirrored as Err. Leading syllables return early, so cluster stays false.
fn syl_ana(sess: &Session, s_in: &str, p: &str) -> Result<(bool, bool), ProcessError> {
    let mut s = s_in.to_string();
    if p.is_empty() {
        return Ok((false, false)); // caller skips empty phones; upstream returns ''
    }
    if re_karan_sub().is_match(&s) {
        s = re_karan_sub().replace_all(&s, "${1}").into_owned();
    }
    if p.contains('\'') {
        return Ok((true, false));
    }
    let mt = re_phone_split()
        .captures(p)
        .ok_or_else(|| ProcessError("'NoneType' object has no attribute 'group'".into()))?;
    let init_ph = mt.get(1).map(|g| g.as_str()).unwrap_or("");
    for pat in &sess.m.pron {
        let full = match &pat.re_full {
            Some(f) => f,
            None => continue, // upstream: `if "(" not in f: continue`
        };
        if let Some(caps) = full.captures(&s) {
            if caps.get(3).is_some() {
                return Ok((false, false));
            }
            if caps.get(2).is_some() {
                let cluster = re_rlw().is_match(init_ph);
                return Ok((false, cluster));
            }
            return Ok((false, false));
        }
    }
    Ok((false, false))
}

/// nlp.WordAna — per-word syllable features; Ok(None) == upstream's '' (skip).
fn wordana(sess: &mut Session, word: &str) -> Result<Option<WordAnaOut>, ProcessError> {
    let w = word.replace(' ', "");
    if w.is_empty() || w == "\t" {
        return Ok(None);
    }
    let gp = sess.g2p(&w);
    if gp.contains("<Fail>") {
        return Ok(None);
    }
    let gp = match gp.strip_suffix("|<s/>") {
        Some(x) => x.to_string(),
        None => gp,
    };
    let gp = gp.replace('|', "~");
    let tr = gp.matches("<tr/>").count();
    if tr == 0 || tr > 1 {
        return Ok(None);
    }
    let (form, phone) = gp.split_once("<tr/>").unwrap();
    if form == phone {
        return Ok(None);
    }
    let mut out = WordAnaOut::default();
    if form.contains('์') {
        out.karan += 1;
    }
    let frmlst: Vec<&str> = form.split('~').collect();
    let phlst: Vec<&str> = phone.split('~').collect();
    out.syl = frmlst.len() as i64;
    for (i, &syl_form) in frmlst.iter().enumerate() {
        let phon_form = match phlst.get(i) {
            Some(x) => *x,
            None => return Err(ProcessError("list index out of range".into())),
        };
        if phon_form.is_empty() || syl_form == "ๆ" {
            continue;
        }
        let (leading, cluster) = syl_ana(sess, syl_form, phon_form)?;
        if let Some(m) = re_tone().captures(syl_form) {
            let t = m.get(0).unwrap().as_str();
            let corr = match t {
                "\u{0E48}" => phon_form.contains('1'),
                "\u{0E49}" => phon_form.contains('2'),
                "\u{0E4A}" => phon_form.contains('3'),
                "\u{0E4B}" => phon_form.contains('4'),
                _ => false,
            };
            if corr {
                out.corrtone += 1;
            } else {
                out.incorrtone += 1;
            }
        } else if phon_form.contains('0') {
            out.corrtone += 1;
        } else {
            out.incorrtone += 1;
        }
        if re_final_c().is_match(syl_form) && re_final_ph().is_match(phon_form) {
            out.corrfinal += 1;
        } else if re_not_final_c().is_match(syl_form) && re_final_ph().is_match(phon_form) {
            out.incorrfinal += 1;
        }
        if re_doub_v().is_match(phon_form) {
            out.doubvowel += 1;
        }
        if cluster {
            out.cluster += 1;
        }
        if leading {
            out.lead += 1;
        }
    }
    Ok(Some(out))
}

/// nlp.Duplicate_word
fn duplicate_word(pos: &str, lst1: &[(String, String)], lst2: &[(String, String)]) -> bool {
    for (w1, p1) in lst1 {
        for (w2, p2) in lst2 {
            if w1 == w2 && p1 == p2 && p1 == pos {
                return true;
            }
        }
    }
    false
}

// ---- TextAna ----------------------------------------------------------------

struct Feats {
    des_symb_c: i64,
    des_nel_c: i64,
    des_neo_c: i64,
    des_nep_c: i64,
    crf_cnl: i64,
    crf_cvl: i64,
    crf_cwl: i64,
    crf_ctl: i64,
    des_neg_c: i64,
    des_tnct1c: i64,
    des_tnct2c: i64,
    des_tnct3c: i64,
    des_tnct4c: i64,
    des_ttc1: i64,
    des_ttc2: i64,
    des_ttc3: i64,
    des_ttc4: i64,
    wrd_c: i64,
    wrd_f: i64,
    wrd_cf: i64,
    wrd_sem_simp: i64,
    wrd_sem_tran: i64,
    wrd_sem_semi: i64,
    wrd_sem_opaq: i64,
    wrd_frm_sing: i64,
    wrd_frm_tran: i64,
    wrd_frm_comp: i64,
    wrd_basic_m: i64,
    wrd_basic_t: i64,
    des_corr_tone_c: i64,
    des_in_corr_tone_c: i64,
    des_corr_final_c: i64,
    des_in_corr_final_c: i64,
    des_karan_c: i64,
    des_cluster_c: i64,
    des_lead_c: i64,
    des_doub_vowel_c: i64,
    des_tot_w: i64,
    wrd_sym: i64,
    wrd_punct: i64,
    wrd_noun: i64,
    wrd_verb: i64,
    wrd_adv: i64,
    wrd_det: i64,
    wrd_adj: i64,
    wrd_adp: i64,
    wrd_aux: i64,
    wrd_intj: i64,
    wrd_cconj: i64,
    wrd_propn: i64,
    wrd_num: i64,
    wrd_part: i64,
    wrd_pron: i64,
    wrd_sconj: i64,
}

fn wordtier_counts(sess: &Session, w: &str, f: &mut Feats) {
    let info = match sess.m.wordtier.as_ref().and_then(|t| t.get(w)) {
        Some(i) => i,
        None => return,
    };
    use crate::model::WordInfo;
    let WordInfo {
        tier,
        level,
        cont,
        seman,
        form,
        lvocab,
        bvocab,
        ..
    } = info;
    if let Some(t) = tier {
        match t.as_str() {
            "T1" => f.des_tnct1c += 1,
            "T2" => f.des_tnct2c += 1,
            "T3" => f.des_tnct3c += 1,
            "T4" => f.des_tnct4c += 1,
            _ => {}
        }
    }
    if let Some(l) = level {
        match l.as_str() {
            "L1" => f.des_ttc1 += 1,
            "L2" => f.des_ttc2 += 1,
            "L3" => f.des_ttc3 += 1,
            "L4" => f.des_ttc4 += 1,
            _ => {}
        }
    }
    if let Some(c) = cont {
        match c.as_str() {
            "CF" => f.wrd_cf += 1,
            "C" => f.wrd_c += 1,
            "F" => f.wrd_f += 1,
            _ => {}
        }
    }
    if let Some(s) = seman {
        match s.as_str() {
            "simp" => f.wrd_sem_simp += 1,
            "tranp" => f.wrd_sem_tran += 1,
            "semi" => f.wrd_sem_semi += 1,
            "opaq" => f.wrd_sem_opaq += 1,
            _ => {}
        }
    }
    if let Some(fm) = form {
        match fm.as_str() {
            "sing" => f.wrd_frm_sing += 1,
            "tran" => f.wrd_frm_tran += 1,
            "comp" => f.wrd_frm_comp += 1,
            _ => {}
        }
    }
    if lvocab.as_deref() == Some("L1") {
        f.wrd_basic_m += 1;
    }
    if bvocab.as_deref() == Some("2000") {
        f.wrd_basic_t += 1;
    }
}

fn stat_err(msg: &str) -> ProcessError {
    ProcessError(msg.to_string())
}

/// The textana dispatch body. `text` is analysed with upstream's default
/// options; see the module doc for the wire format and crash parity.
pub fn textana(sess: &mut Session, alltext: &str) -> Result<String, ProcessError> {
    compute_textana(sess, alltext).map(|o| o.json)
}

pub(crate) struct TextAnaOutput {
    pub json: String,
    pub features: HashMap<&'static str, f64>,
}

pub(crate) fn compute_textana(sess: &mut Session, alltext: &str) -> Result<TextAnaOutput, ProcessError> {
    if sess.m.wordtier.is_none() {
        return Err(stat_err("wordtier.bin missing (textana requires it)"));
    }

    // TextOption == 'par' (default); WordOption == 'colloc'
    let mut s = alltext.replace('\n', "<u/>");
    s = sess.word_segment(&s);
    s = s.replace("<s/>", "|<s/>|");
    let utext = s.replace("||", "|");
    let des_pc = re_nl_run().find_iter(&s).count() as i64;
    let des_sp_c = s.chars().filter(|c| *c == ' ' || *c == '\t').count() as i64;

    let des_ch_c = alltext.chars().count() as i64;
    let des_edu_c = utext.matches("<u/>").count() as i64 - 1;
    let utts: Vec<&str> = utext.split("<u/>").collect();
    let no_crf = std::cmp::max(utts.len() as i64 - 1, 1);

    let mut f = Feats {
        des_symb_c: 0,
        des_nel_c: 0,
        des_neo_c: 0,
        des_nep_c: 0,
        crf_cnl: 0,
        crf_cvl: 0,
        crf_cwl: 0,
        crf_ctl: 0,
        des_neg_c: 0,
        des_tnct1c: 0,
        des_tnct2c: 0,
        des_tnct3c: 0,
        des_tnct4c: 0,
        des_ttc1: 0,
        des_ttc2: 0,
        des_ttc3: 0,
        des_ttc4: 0,
        wrd_c: 0,
        wrd_f: 0,
        wrd_cf: 0,
        wrd_sem_simp: 0,
        wrd_sem_tran: 0,
        wrd_sem_semi: 0,
        wrd_sem_opaq: 0,
        wrd_frm_sing: 0,
        wrd_frm_tran: 0,
        wrd_frm_comp: 0,
        wrd_basic_m: 0,
        wrd_basic_t: 0,
        des_corr_tone_c: 0,
        des_in_corr_tone_c: 0,
        des_corr_final_c: 0,
        des_in_corr_final_c: 0,
        des_karan_c: 0,
        des_cluster_c: 0,
        des_lead_c: 0,
        des_doub_vowel_c: 0,
        des_tot_w: 0,
        wrd_sym: 0,
        wrd_punct: 0,
        wrd_noun: 0,
        wrd_verb: 0,
        wrd_adv: 0,
        wrd_det: 0,
        wrd_adj: 0,
        wrd_adp: 0,
        wrd_aux: 0,
        wrd_intj: 0,
        wrd_cconj: 0,
        wrd_propn: 0,
        wrd_num: 0,
        wrd_part: 0,
        wrd_pron: 0,
        wrd_sconj: 0,
    };

    let mut prev_tag_results: Vec<(String, String)> = Vec::new();
    let mut eduwrd_lst: Vec<i64> = Vec::new();
    let mut wrdsyl_lst: Vec<i64> = Vec::new();
    let mut parwrd: i64 = 0;
    let mut words_lst: Vec<String> = Vec::new();
    let mut wtype: HashSet<String> = HashSet::new();
    // txt.wrd in Python dict order: first-appearance order drives the float
    // summation order of WrdTfidf/TncDisp/TtcDisp/Arf — must be preserved.
    let mut wrd_order: Vec<String> = Vec::new();
    let mut wrd_counts: HashMap<String, i64> = HashMap::new();

    for (i, u0) in utts.iter().enumerate() {
        let u = *u0;
        if u.is_empty() || u == "|" {
            continue; // upstream also decrements no_Crf here
        }
        let sent: Vec<String> = u.split('|').filter(|x| !x.is_empty()).map(String::from).collect();
        let tag_results: Vec<(String, String)> = perceptron_tag(sess, &sent)
            .into_iter()
            .map(|(w, p)| {
                let p2 = change_tag(&w, &p);
                (w, p2)
            })
            .collect();
        let u_packed = pack_ner(&ner(sess, &tag_results));

        if duplicate_word("NOUN", &tag_results, &prev_tag_results) {
            f.crf_cnl += 1;
        }
        if duplicate_word("VERB", &tag_results, &prev_tag_results) {
            f.crf_cvl += 1;
        }
        prev_tag_results = tag_results;

        f.des_nel_c += u_packed.matches("<NEl>").count() as i64;
        f.des_neo_c += u_packed.matches("<NEo>").count() as i64;
        f.des_nep_c += u_packed.matches("<NEp>").count() as i64;
        let u = re_ne_tag().replace_all(&u_packed, "").into_owned();

        let wlst: Vec<&str> = u.split('|').collect();
        parwrd += wlst.len() as i64;
        eduwrd_lst.push(wlst.len() as i64);

        for wp in &wlst {
            if re_symbol().is_match(wp) {
                f.des_symb_c += 1;
                continue;
            } else if wp.contains("SYM") || wp.contains("PUNCT") {
                f.des_symb_c += 1;
                if wp.contains("SYM") {
                    f.wrd_sym += 1;
                } else if wp.contains("PUNCT") {
                    f.wrd_punct += 1;
                }
                continue;
            }
            if wp.is_empty() || !wp.contains('/') || wp.matches('/').count() > 1 {
                continue;
            }
            let (w, p) = wp.split_once('/').unwrap();
            wtype.insert(w.to_string());
            match wrd_counts.get_mut(w) {
                Some(c) => *c += 1,
                None => {
                    wrd_order.push(w.to_string());
                    wrd_counts.insert(w.to_string(), 1);
                }
            }
            match p {
                "NOUN" => f.wrd_noun += 1,
                "VERB" => f.wrd_verb += 1,
                "ADV" => f.wrd_adv += 1,
                "DET" => f.wrd_det += 1,
                "ADJ" => f.wrd_adj += 1,
                "ADP" => f.wrd_adp += 1,
                "AUX" => f.wrd_aux += 1,
                "INTJ" => f.wrd_intj += 1,
                "CCONJ" => f.wrd_cconj += 1,
                "PROPN" => f.wrd_propn += 1,
                "NUM" => f.wrd_num += 1,
                "PART" => f.wrd_part += 1,
                "PRON" => f.wrd_pron += 1,
                "SCONJ" => f.wrd_sconj += 1,
                _ => {}
            }
            words_lst.push(w.to_string());
            if w.contains("ไม่") {
                f.des_neg_c += 1;
            }
            wordtier_counts(sess, w, &mut f);

            match wordana(sess, w)? {
                None => continue,
                Some(wa) => {
                    wrdsyl_lst.push(wa.syl);
                    f.des_corr_tone_c += wa.corrtone;
                    f.des_in_corr_tone_c += wa.incorrtone;
                    f.des_corr_final_c += wa.corrfinal;
                    f.des_in_corr_final_c += wa.incorrfinal;
                    f.des_karan_c += wa.karan;
                    f.des_cluster_c += wa.cluster;
                    f.des_lead_c += wa.lead;
                    f.des_doub_vowel_c += wa.doubvowel;
                    f.des_tot_w += 1;
                }
            }
        }

        if (i as i64) < no_crf && utts.len() > i + 1 {
            let sent_set: HashSet<&str> = sent.iter().map(|s| s.as_str()).collect();
            let sentn: HashSet<&str> = utts[i + 1].split('|').filter(|x| !x.is_empty()).collect();
            let overlap = sentn.intersection(&sent_set).count();
            if overlap > 0 {
                f.crf_cwl += 1;
                f.crf_ctl += overlap as i64;
            }
        }
    }

    // post-loop, upstream line order
    let crf_cnl = f.crf_cnl as f64 / no_crf as f64;
    let crf_cwl = f.crf_cwl as f64 / no_crf as f64;
    let crf_ctl = f.crf_ctl as f64 / no_crf as f64;
    let syn_depth = 0.0f64 / no_crf as f64; // UDParse="none"

    let mut wrd_arf_map: BTreeMap<String, String> = BTreeMap::new();
    let mut wrd_arf_sum = 0.0f64;
    let mut tfidf_sum = 0.0f64;
    let mut tnc_disp_sum = 0.0f64;
    let mut ttc_disp_sum = 0.0f64;
    for w in &wrd_order {
        let a = arf(&words_lst, w);
        wrd_arf_map.insert(w.clone(), py_float_repr(a));
        wrd_arf_sum += a;
        if let Some(info) = sess.m.wordtier.as_ref().unwrap().get(w) {
            if let Some(v) = info.tnc_tfidf {
                tfidf_sum += v;
            }
            if let Some(v) = info.tnc_disp {
                tnc_disp_sum += v;
            }
            if let Some(v) = info.ttc_disp {
                ttc_disp_sum += v;
            }
        }
    }
    let _ = &f.crf_cvl; // counted, never re-assigned upstream

    let des_tot_t = wtype.len() as i64;
    if eduwrd_lst.is_empty() {
        return Err(stat_err("mean requires at least one data point"));
    }
    let des_edu_l = py_mean_ints(&eduwrd_lst);
    let des_edu_ld = py_stdev_ints(&eduwrd_lst)
        .map_err(|_| stat_err("variance requires at least two data points"))?;
    if wrdsyl_lst.is_empty() {
        return Err(stat_err("mean requires at least one data point"));
    }
    let des_wrd_l = py_mean_ints(&wrdsyl_lst);
    let des_wrd_ld = py_stdev_ints(&wrdsyl_lst)
        .map_err(|_| stat_err("variance requires at least two data points"))?;
    // upstream mean()s a single-element list of the accumulated parwrd
    let des_pl = PyNum::Int(parwrd);

    if f.des_tot_w == 0 {
        return Err(stat_err("division by zero")); // upstream: ZeroDivisionError at LdvTTR
    }
    // tltk 1.11: LdvTTR = DesTotT * 100 / DesTotW (1.10 had * 1000)
    let ldv_ttr = ((des_tot_t * 100) as f64) / f.des_tot_w as f64;
    let ldv_ttr_1000 = ((des_tot_t * 1000) as f64) / f.des_tot_w as f64;

    let totw = f.des_tot_w as f64;
    let ratio = |c: i64| -> String { py_float_repr(((c * 1000) as f64) / totw) };
    let wrd_noun = ratio(f.wrd_noun);
    let wrd_verb = ratio(f.wrd_verb);
    let wrd_adv = ratio(f.wrd_adv);
    let wrd_det = ratio(f.wrd_det);
    let wrd_adj = ratio(f.wrd_adj);
    let wrd_adp = ratio(f.wrd_adp);
    let wrd_aux = ratio(f.wrd_aux);
    let wrd_intj = ratio(f.wrd_intj);
    let wrd_cconj = ratio(f.wrd_cconj);
    let wrd_propn = ratio(f.wrd_propn);
    let wrd_num = ratio(f.wrd_num);
    let wrd_part = ratio(f.wrd_part);
    let wrd_pron = ratio(f.wrd_pron);
    let wrd_sconj = ratio(f.wrd_sconj);

    let tott = des_tot_t as f64;
    let wrd_tfidf = py_float_repr(tfidf_sum / tott);
    let wrd_tnc_disp = py_float_repr(tnc_disp_sum / tott);
    let wrd_ttc_disp = py_float_repr(ttc_disp_sum / tott);
    let wrd_arf = py_float_repr(wrd_arf_sum / tott);

    // sorted-key JSON emit (Python: json.dumps(..., sort_keys=True,
    // separators=(",", ":")))
    let mut features: HashMap<&'static str, f64> = HashMap::with_capacity(96);
    features.insert("CrfCNL", crf_cnl);
    features.insert("CrfCTL", crf_ctl);
    features.insert("CrfCWL", crf_cwl);
    features.insert("DesChC", des_ch_c as f64);
    features.insert("DesClusterC", f.des_cluster_c as f64);
    features.insert("DesCorrFinalC", f.des_corr_final_c as f64);
    features.insert("DesCorrToneC", f.des_corr_tone_c as f64);
    features.insert("DesDoubVowelC", f.des_doub_vowel_c as f64);
    features.insert("DesEduC", des_edu_c as f64);
    features.insert("DesEduL", match des_edu_l { PyNum::Int(i) => i as f64, PyNum::Float(f) => f });
    features.insert("DesEduLd", des_edu_ld);
    features.insert("DesInCorrFinalC", f.des_in_corr_final_c as f64);
    features.insert("DesInCorrToneC", f.des_in_corr_tone_c as f64);
    features.insert("DesKaranC", f.des_karan_c as f64);
    features.insert("DesLeadC", f.des_lead_c as f64);
    features.insert("DesNElC", f.des_nel_c as f64);
    features.insert("DesNEoC", f.des_neo_c as f64);
    features.insert("DesNEpC", f.des_nep_c as f64);
    features.insert("DesPC", des_pc as f64);
    features.insert("DesPL", match des_pl { PyNum::Int(i) => i as f64, PyNum::Float(f) => f });
    features.insert("DesSpC", des_sp_c as f64);
    features.insert("DesTotT", des_tot_t as f64);
    features.insert("DesTotW", f.des_tot_w as f64);
    features.insert("DesWrdL", match des_wrd_l { PyNum::Int(i) => i as f64, PyNum::Float(f) => f });
    features.insert("DesWrdLd", des_wrd_ld);
    features.insert("LdvTTR", ldv_ttr_1000); // 1.10 semantics for textclass/txt2feat
    features.insert("SynDepth", syn_depth);

    let ratio_f = |c: i64| -> f64 { ((c * 1000) as f64) / totw };
    features.insert("WrdADJ", ratio_f(f.wrd_adj));
    features.insert("WrdADP", ratio_f(f.wrd_adp));
    features.insert("WrdADV", ratio_f(f.wrd_adv));
    features.insert("WrdAUX", ratio_f(f.wrd_aux));
    features.insert("WrdArf", wrd_arf_sum / tott);
    features.insert("WrdBasicM", ratio_f(f.wrd_basic_m));
    features.insert("WrdBasicT", ratio_f(f.wrd_basic_t));
    features.insert("WrdC", ratio_f(f.wrd_c));
    features.insert("WrdCCONJ", ratio_f(f.wrd_cconj));
    features.insert("WrdCF", ratio_f(f.wrd_cf));
    features.insert("WrdCluster", ratio_f(f.des_cluster_c));
    features.insert("WrdCorrFinal", ratio_f(f.des_corr_final_c));
    features.insert("WrdCorrTone", ratio_f(f.des_corr_tone_c));
    features.insert("WrdDET", ratio_f(f.wrd_det));
    features.insert("WrdDoubVowel", ratio_f(f.des_doub_vowel_c));
    features.insert("WrdF", ratio_f(f.wrd_f));
    features.insert("WrdFrmComp", ratio_f(f.wrd_frm_comp));
    features.insert("WrdFrmSing", ratio_f(f.wrd_frm_sing));
    features.insert("WrdFrmTran", ratio_f(f.wrd_frm_tran));
    features.insert("WrdINTJ", ratio_f(f.wrd_intj));
    features.insert("WrdInCorrFinal", ratio_f(f.des_in_corr_final_c));
    features.insert("WrdInCorrTone", ratio_f(f.des_in_corr_tone_c));
    features.insert("WrdKaran", ratio_f(f.des_karan_c));
    features.insert("WrdLead", ratio_f(f.des_lead_c));
    features.insert("WrdNEl", ratio_f(f.des_nel_c));
    features.insert("WrdNEo", ratio_f(f.des_neo_c));
    features.insert("WrdNEp", ratio_f(f.des_nep_c));
    features.insert("WrdNeg", ratio_f(f.des_neg_c));
    features.insert("WrdNOUN", ratio_f(f.wrd_noun));
    features.insert("WrdNUM", ratio_f(f.wrd_num));
    features.insert("WrdPART", ratio_f(f.wrd_part));
    features.insert("WrdPRON", ratio_f(f.wrd_pron));
    features.insert("WrdPROPN", ratio_f(f.wrd_propn));
    features.insert("WrdPUNCT", ratio_f(f.wrd_punct));
    features.insert("WrdSCONJ", ratio_f(f.wrd_sconj));
    features.insert("WrdSYM", ratio_f(f.wrd_sym));
    features.insert("WrdSemOpaq", ratio_f(f.wrd_sem_opaq));
    features.insert("WrdSemSemi", ratio_f(f.wrd_sem_semi));
    features.insert("WrdSemSimp", ratio_f(f.wrd_sem_simp));
    features.insert("WrdSemTran", ratio_f(f.wrd_sem_tran));
    features.insert("WrdTfidf", tfidf_sum / tott);
    features.insert("WrdTNCt1", ratio_f(f.des_tnct1c));
    features.insert("WrdTNCt2", ratio_f(f.des_tnct2c));
    features.insert("WrdTNCt3", ratio_f(f.des_tnct3c));
    features.insert("WrdTNCt4", ratio_f(f.des_tnct4c));
    features.insert("WrdTTC1", ratio_f(f.des_ttc1));
    features.insert("WrdTTC2", ratio_f(f.des_ttc2));
    features.insert("WrdTTC3", ratio_f(f.des_ttc3));
    features.insert("WrdTTC4", ratio_f(f.des_ttc4));
    features.insert("WrdTncDisp", tnc_disp_sum / tott);
    features.insert("WrdTtcDisp", ttc_disp_sum / tott);
    features.insert("WrdVERB", ratio_f(f.wrd_verb));
    let mut parts: Vec<(&str, String)> = Vec::with_capacity(96);
    parts.push(("CrfCNL", py_float_repr(crf_cnl)));
    parts.push(("CrfCTL", py_float_repr(crf_ctl)));
    parts.push(("CrfCWL", py_float_repr(crf_cwl)));
    parts.push(("DesChC", des_ch_c.to_string()));
    parts.push(("DesClusterC", f.des_cluster_c.to_string()));
    parts.push(("DesCorrFinalC", f.des_corr_final_c.to_string()));
    parts.push(("DesCorrToneC", f.des_corr_tone_c.to_string()));
    parts.push(("DesDoubVowelC", f.des_doub_vowel_c.to_string()));
    parts.push(("DesEduC", des_edu_c.to_string()));
    parts.push(("DesEduL", py_num_emit(&des_edu_l)));
    parts.push(("DesEduLd", py_float_repr(des_edu_ld)));
    parts.push(("DesInCorrFinalC", f.des_in_corr_final_c.to_string()));
    parts.push(("DesInCorrToneC", f.des_in_corr_tone_c.to_string()));
    parts.push(("DesKaranC", f.des_karan_c.to_string()));
    parts.push(("DesLeadC", f.des_lead_c.to_string()));
    parts.push(("DesNElC", f.des_nel_c.to_string()));
    parts.push(("DesNEoC", f.des_neo_c.to_string()));
    parts.push(("DesNEpC", f.des_nep_c.to_string()));
    parts.push(("DesPC", des_pc.to_string()));
    parts.push(("DesPL", py_num_emit(&des_pl)));
    parts.push(("DesSpC", des_sp_c.to_string()));
    // fields whose Python `+=` only fires conditionally exist in vars(txt)
    // only when non-zero — CrfCVL additionally never gets the /no_Crf pass
    if f.crf_cvl > 0 {
        parts.push(("CrfCVL", f.crf_cvl.to_string()));
        features.insert("CrfCVL", f.crf_cvl as f64);
    }
    if f.des_symb_c > 0 {
        parts.push(("DesSymbC", f.des_symb_c.to_string()));
        features.insert("DesSymbC", f.des_symb_c as f64);
    }
    if f.des_tnct1c > 0 {
        parts.push(("DesTNCt1C", f.des_tnct1c.to_string()));
        features.insert("DesTNCt1C", f.des_tnct1c as f64);
    }
    if f.des_tnct2c > 0 {
        parts.push(("DesTNCt2C", f.des_tnct2c.to_string()));
        features.insert("DesTNCt2C", f.des_tnct2c as f64);
    }
    if f.des_tnct3c > 0 {
        parts.push(("DesTNCt3C", f.des_tnct3c.to_string()));
        features.insert("DesTNCt3C", f.des_tnct3c as f64);
    }
    if f.des_tnct4c > 0 {
        parts.push(("DesTNCt4C", f.des_tnct4c.to_string()));
        features.insert("DesTNCt4C", f.des_tnct4c as f64);
    }
    if f.des_ttc1 > 0 {
        parts.push(("DesTTC1", f.des_ttc1.to_string()));
        features.insert("DesTTC1", f.des_ttc1 as f64);
    }
    if f.des_ttc2 > 0 {
        parts.push(("DesTTC2", f.des_ttc2.to_string()));
        features.insert("DesTTC2", f.des_ttc2 as f64);
    }
    if f.des_ttc3 > 0 {
        parts.push(("DesTTC3", f.des_ttc3.to_string()));
        features.insert("DesTTC3", f.des_ttc3 as f64);
    }
    if f.des_ttc4 > 0 {
        parts.push(("DesTTC4", f.des_ttc4.to_string()));
        features.insert("DesTTC4", f.des_ttc4 as f64);
    }
    if f.des_neg_c > 0 {
        parts.push(("DesNegC", f.des_neg_c.to_string()));
        features.insert("DesNegC", f.des_neg_c as f64);
    }
    parts.push(("DesTotT", des_tot_t.to_string()));
    parts.push(("DesTotW", f.des_tot_w.to_string()));
    parts.push(("DesWrdL", py_num_emit(&des_wrd_l)));
    parts.push(("DesWrdLd", py_float_repr(des_wrd_ld)));
    parts.push(("LdvTTR", py_float_repr(ldv_ttr)));
    parts.push(("SynDepth", py_float_repr(syn_depth)));
    parts.push(("WrdADJ", wrd_adj));
    parts.push(("WrdADP", wrd_adp));
    parts.push(("WrdADV", wrd_adv));
    parts.push(("WrdAUX", wrd_aux));
    parts.push(("WrdArf", wrd_arf));
    parts.push(("WrdBasicM", ratio(f.wrd_basic_m)));
    parts.push(("WrdBasicT", ratio(f.wrd_basic_t)));
    parts.push(("WrdC", ratio(f.wrd_c)));
    parts.push(("WrdCCONJ", wrd_cconj));
    parts.push(("WrdCF", ratio(f.wrd_cf)));
    parts.push(("WrdCluster", ratio(f.des_cluster_c)));
    parts.push(("WrdCorrFinal", ratio(f.des_corr_final_c)));
    parts.push(("WrdCorrTone", ratio(f.des_corr_tone_c)));
    parts.push(("WrdDET", wrd_det));
    parts.push(("WrdDoubVowel", ratio(f.des_doub_vowel_c)));
    parts.push(("WrdF", ratio(f.wrd_f)));
    parts.push(("WrdFrmComp", ratio(f.wrd_frm_comp)));
    parts.push(("WrdFrmSing", ratio(f.wrd_frm_sing)));
    parts.push(("WrdFrmTran", ratio(f.wrd_frm_tran)));
    parts.push(("WrdINTJ", wrd_intj));
    parts.push(("WrdInCorrFinal", ratio(f.des_in_corr_final_c)));
    parts.push(("WrdInCorrTone", ratio(f.des_in_corr_tone_c)));
    parts.push(("WrdKaran", ratio(f.des_karan_c)));
    parts.push(("WrdLead", ratio(f.des_lead_c)));
    parts.push(("WrdNEl", ratio(f.des_nel_c)));
    parts.push(("WrdNEo", ratio(f.des_neo_c)));
    parts.push(("WrdNEp", ratio(f.des_nep_c)));
    parts.push(("WrdNeg", ratio(f.des_neg_c)));
    parts.push(("WrdNOUN", wrd_noun));
    parts.push(("WrdNUM", wrd_num));
    parts.push(("WrdPART", wrd_part));
    parts.push(("WrdPRON", wrd_pron));
    parts.push(("WrdPROPN", wrd_propn));
    parts.push(("WrdPUNCT", ratio(f.wrd_punct)));
    parts.push(("WrdSCONJ", wrd_sconj));
    parts.push(("WrdSYM", ratio(f.wrd_sym)));
    parts.push(("WrdSemOpaq", ratio(f.wrd_sem_opaq)));
    parts.push(("WrdSemSemi", ratio(f.wrd_sem_semi)));
    parts.push(("WrdSemSimp", ratio(f.wrd_sem_simp)));
    parts.push(("WrdSemTran", ratio(f.wrd_sem_tran)));
    parts.push(("WrdTfidf", wrd_tfidf));
    parts.push(("WrdTNCt1", ratio(f.des_tnct1c)));
    parts.push(("WrdTNCt2", ratio(f.des_tnct2c)));
    parts.push(("WrdTNCt3", ratio(f.des_tnct3c)));
    parts.push(("WrdTNCt4", ratio(f.des_tnct4c)));
    parts.push(("WrdTTC1", ratio(f.des_ttc1)));
    parts.push(("WrdTTC2", ratio(f.des_ttc2)));
    parts.push(("WrdTTC3", ratio(f.des_ttc3)));
    parts.push(("WrdTTC4", ratio(f.des_ttc4)));
    parts.push(("WrdTncDisp", wrd_tnc_disp));
    parts.push(("WrdTtcDisp", wrd_ttc_disp));
    parts.push(("WrdVERB", wrd_verb));
    parts.sort_by(|a, b| a.0.cmp(b.0));

    let mut wrd_json = String::from("{");
    let mut wrd_sorted: BTreeMap<&str, i64> = BTreeMap::new();
    for w in &wrd_order {
        wrd_sorted.insert(w, wrd_counts[w]);
    }
    for (i, (w, c)) in wrd_sorted.iter().enumerate() {
        if i > 0 {
            wrd_json.push(',');
        }
        wrd_json.push_str(&format!("\"{}\":{}", w, c));
    }
    wrd_json.push('}');

    let mut arf_json = String::from("{");
    for (i, (w, v)) in wrd_arf_map.iter().enumerate() {
        if i > 0 {
            arf_json.push(',');
        }
        arf_json.push_str(&format!("\"{}\":{}", w, v));
    }
    arf_json.push('}');

    parts.push(("deprel", "{}".to_string()));
    parts.push(("wrd", wrd_json));
    parts.push(("wrd_arf", arf_json));
    parts.sort_by(|a, b| a.0.cmp(b.0));

    let mut out = String::with_capacity(2048);
    out.push('{');
    for (i, (k, v)) in parts.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(k);
        out.push_str("\":");
        out.push_str(v);
    }
    out.push('}');
    Ok(TextAnaOutput { json: out, features })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_float_repr_matches_cpython() {
        // expected values produced by CPython 3.9 repr()
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (100.0, "100.0"),
            (7.0, "7.0"),
            (0.5, "0.5"),
            (-2.5, "-2.5"),
            (70.58823529411765, "70.58823529411765"),
            (33.333333333333336, "33.333333333333336"),
            (2.7058823529411766, "2.7058823529411766"),
            (19068.932691198414, "19068.932691198414"),
            (0.242535625036333, "0.242535625036333"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.2345e-5, "1.2345e-05"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (9.9e16, "9.9e+16"),
            (1.3186274509803921, "1.3186274509803921"),
        ];
        for (v, want) in cases {
            assert_eq!(&py_float_repr(*v), want, "py_float_repr({v})");
        }
    }

    #[test]
    fn mean_int_returns_int_only_when_divisible() {
        match py_mean_ints(&[6, 8, 7]) {
            PyNum::Int(i) => assert_eq!(i, 7),
            _ => panic!("expected int mean"),
        }
        match py_mean_ints(&[6, 7]) {
            PyNum::Float(f) => assert_eq!(f, 6.5),
            _ => panic!("expected float mean"),
        }
    }

    #[test]
    fn stdev_ints_exact_and_guards() {
        assert_eq!(py_stdev_ints(&[6, 8, 7]).unwrap(), 1.0);
        assert_eq!(py_stdev_ints(&[4, 4, 4, 4]).unwrap(), 0.0);
        assert!(py_stdev_ints(&[5]).is_err());
        assert!(py_stdev_ints(&[]).is_err());
    }

    #[test]
    fn arf_hapax_is_one() {
        let words: Vec<String> = ["แมว", "กิน", "ปลา"].iter().map(|s| s.to_string()).collect();
        assert_eq!(arf(&words, "แมว"), 1.0);
        assert_eq!(arf(&words, "หมู"), 0.0);
    }
}
