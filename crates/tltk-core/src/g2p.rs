//! Grapheme-to-phoneme pipeline (port of nlp.sylparse / wordparse / g2p and
//! their helpers: ReplaceSnd, NotExceptionSyl, ToneAssign, DeadSyl,
//! TransformSyl, SelectPhones, ProbPhone).

use crate::prelude::*;
use crate::sylseg::SpanMap;
use crate::Session;
use std::collections::HashMap;

/// stable['X'] — initial consonant sound map (hard-coded in read_sylpattern).
pub fn stable_x(c: &str) -> Option<&'static str> {
    Some(match c {
        "ก" => "k", "ข" => "kh", "ฃ" => "kh", "ค" => "kh", "ฅ" => "kh", "ฆ" => "kh",
        "ง" => "N", "จ" => "c", "ฉ" => "ch", "ช" => "ch", "ซ" => "s", "ฌ" => "ch",
        "ญ" => "j", "ฎ" => "d", "ฏ" => "t", "ฐ" => "th", "ฑ" => "th", "ฒ" => "th",
        "ณ" => "n", "ด" => "d", "ต" => "t", "ถ" => "th", "ท" => "th", "ธ" => "th",
        "น" => "n", "บ" => "b", "ป" => "p", "ผ" => "ph", "ฝ" => "f", "พ" => "ph",
        "ฟ" => "f", "ภ" => "ph", "ม" => "m", "ย" => "j", "ร" => "r", "ฤ" => "r",
        "ล" => "l", "ฦ" => "l", "ว" => "w", "ศ" => "s", "ษ" => "s", "ส" => "s",
        "ห" => "h", "ฬ" => "l", "อ" => "?", "ฮ" => "h",
        _ => return None,
    })
}

/// stable['Y'] — final consonant sound map.
pub fn stable_y(c: &str) -> Option<&'static str> {
    Some(match c {
        "ก" => "k", "ข" => "k", "ค" => "k", "ฆ" => "k", "ง" => "N", "จ" => "t",
        "ฉ" => "-", "ช" => "t", "ซ" => "t", "ฌ" => "-", "ญ" => "n", "ฎ" => "t",
        "ฏ" => "t", "ฐ" => "t", "ฑ" => "t", "ฒ" => "t", "ณ" => "n", "ด" => "t",
        "ต" => "t", "ถ" => "t", "ท" => "t", "ธ" => "t", "น" => "n", "บ" => "p",
        "ป" => "p", "ผ" => "-", "ฝ" => "-", "พ" => "p", "ฟ" => "p", "ภ" => "p",
        "ม" => "m", "ย" => "j", "ร" => "n", "ฤ" => "-", "ล" => "n", "ฦ" => "-",
        "ว" => "w", "ศ" => "t", "ษ" => "t", "ส" => "t", "ห" => "-", "ฬ" => "n",
        "อ" => "-", "ฮ" => "-",
        _ => return None,
    })
}

fn table_of(code: &str) -> Option<fn(&str) -> Option<&'static str>> {
    match code {
        "X" | "A" | "K" | "C" | "R" | "G" | "E" => Some(stable_x),
        "Y" | "D" | "Z" | "H" | "F" => Some(stable_y),
        _ => None,
    }
}

/// AK[leading consonant] = allowed spellable following consonants (hard-coded).
pub fn ak_allowed(lead: &str) -> Option<&'static str> {
    Some(match lead {
        "ก" => "นฎฐณตถบปมรลวษสพหฬ",
        "ข" => "จนณบมยรลฬดตทษภส",
        "ค" => "กชดตนณปมทฑหบ",
        "ฆ" => "กบรสช",
        "จ" => "ณตนมรลทยกด",
        "ฉ" => "กงนบพรลวทม",
        "ช" => "กญนยรลวฎคทบมอ",
        "ซ" => "ล",
        "ญ" => "กภลญ",
        "ณ" => "กร",
        "ด" => "นรมวบ",
        "ต" => "กตนภยรลฆงถบมวฤท",
        "ถ" => "กนลวศพงมร",
        "ท" => "กชนบมยรลวสหงศ",
        "ธ" => "นภมยรวกพช",
        "น" => "กทธฎภมยรวคขปลวห",
        "บ" => "ดรทพม",
        "ป" => "กฐฏณทนภรวศถฎตปยสหด",
        "ผ" => "งจชดนลสวกคณทยรอ",
        "ฝ" => "ร",
        "พ" => "กญนมยลวสหณจธ",
        "ภ" => "ยรคณมฤวน",
        "ม" => "กณตนยรลหศดธมสฆว",
        "ย" => "กดธภวสบศตมนถช",
        "ร" => "กจณดพภมยวสหชตถนบ",
        "ล" => "กคฆดตอบปพมลวห",
        "ว" => "ชณดนพภรลสมยกจฏตทธปฤศหคธ",
        "ศ" => "จณนบรลวศพดธกตมยส",
        "ษ" => "ณฎบภรนคธม",
        "ส" => "กคงดตถนบปพภมยรลวหอจฟสขทธฤ",
        "ห" => "กงพทนรภญนมยรลวบต",
        "อ" => "กงชณดตธนพภมยรลวสศคบฆจทปห",
        "ฑ" => "มสรนค",
        "ฐ" => "กจ",
        "ฏ" => "ก",
        "ฌ" => "ก",
        _ => return None,
    })
}

const ENG_ABBR: &[&str] = &[
    "เอ", "บี", "ซี", "ดี", "อี", "เอฟ", "จี", "เจ", "เอช", "ไอ", "เค", "แอล", "เอ็ม",
    "เอ็น", "โอ", "พี", "คิว", "อาร์", "เอส", "ที", "ยู", "วี", "เอ็กซ์", "เอ็ก", "วาย", "แซด",
];

pub fn is_eng_abbr(s: &str) -> bool {
    ENG_ABBR.contains(&s)
}

/// Port of ReplaceSnd.
pub fn replace_snd(phone: &str, codematch: &str, charmatch: &str) -> String {
    let chars: Vec<&str> = charmatch.split(' ').collect();
    let mut snd = phone.to_string();
    for (i, x) in codematch.chars().enumerate() {
        if i >= chars.len() {
            break;
        }
        let t = chars[i];
        if let Some(tbl) = table_of(&x.to_string()) {
            if let Some(s) = tbl(t) {
                snd = snd.replace(&x.to_string(), s);
            }
        }
    }
    snd.push('8');
    snd
}

/// Port of NotExceptionSyl.
pub fn not_exception_syl(codematch: &str, charmatch: &str, form: &str, phone: &str) -> bool {
    static RE_ACR: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static RE_KAP: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static RE_VOWEL: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re_acr = RE_ACR.get_or_init(|| {
        regex::Regex::new(r"ผ ร|ด ล|ต ล|ท ล|ด ว|ต ว|ท ว|บ ว|ป ว|พ ว|ฟ ว|ผ ว|ส ล|ส ว|ร ร|ศ ล|ศ ว").unwrap()
    });
    let re_kap = RE_KAP.get_or_init(|| regex::Regex::new(r"[ก-ฮ] ข").unwrap());
    let re_vowel = RE_VOWEL.get_or_init(|| regex::Regex::new(r"[\u{0E38}\u{0E31}\u{0E40}]").unwrap());

    if form.contains('.') {
        return true;
    }
    if codematch.contains("CR") && re_acr.is_match(charmatch) {
        return false;
    }
    if codematch.contains("AK") {
        let clst: Vec<&str> = charmatch.split(' ').collect();
        if clst.len() > 1 {
            match ak_allowed(clst[0]) {
                Some(allowed) => {
                    if !allowed.contains(clst[1]) {
                        return false;
                    }
                }
                None => return false, // Python AK[clst[0]] would KeyError
            }
        }
    }
    // Case 1: the source pattern r"\u0E31[\0E48-\u0E4B]?ว]" parses in Python as
    // ั + opt{NUL,'E','4',range('8'..0x0E4B)} + ว + literal ']' — it fires only
    // when the form contains a literal ']' right after ว. Ported verbatim.
    static RE_CASE1: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re_case1 =
        RE_CASE1.get_or_init(|| regex::Regex::new(r"\u{0E31}[\u{0000}E4\u{38}-\u{0E4B}]?ว\]").unwrap());
    if re_case1.is_match(form) && phone.contains("aw") {
        return false;
    }
    if re_kap.is_match(charmatch) && !re_vowel.is_match(form) {
        return false;
    }
    true
}

/// Port of DeadSyl — 'D' or 'L' from a phone string.
pub fn dead_syl(phone: &str) -> &'static str {
    let inx = phone.replace("ch", "C");
    let inx = strip_tone_digits(&inx);
    let chars: Vec<char> = inx.chars().collect();
    if ends_with_opt8(&["m", "n", "w", "j", "l", "N"], &chars) {
        return "L";
    }
    if ends_with_opt8(&["p", "k", "t", "f", "s", "c", "C"], &chars) {
        return "D";
    }
    if has_double_vowel(&inx) {
        return "L";
    }
    "D"
}

fn strip_tone_digits(s: &str) -> String {
    s.chars().filter(|c| !('0'..='4').contains(c)).collect()
}

fn ends_with_opt8(finals: &[&str], chars: &[char]) -> bool {
    let last = match chars.last() {
        Some(c) => *c,
        None => return false,
    };
    if finals.contains(&&*last.to_string()) {
        return true;
    }
    if last == '8' && chars.len() >= 2 {
        let prev = chars[chars.len() - 2];
        return finals.contains(&&*prev.to_string());
    }
    false
}

fn has_double_vowel(inx: &str) -> bool {
    const V: &str = "aeiouxOU@";
    let b: Vec<char> = inx.chars().collect();
    for w in b.windows(2) {
        if w[0] == w[1] && V.contains(w[0]) {
            return true;
        }
    }
    false
}

/// Port of DeadSylx — 'Dl'/'Ll'/'Ls'/'Ds'.
pub fn dead_sylx(phone: &str) -> &'static str {
    let inx = phone.replace("ch", "C");
    let inx = strip_tone_digits(&inx);
    if has_double_vowel(&inx) {
        let chars: Vec<char> = inx.chars().collect();
        if ends_with_opt8(&["p", "k", "t", "f", "s", "c", "C"], &chars) {
            "Dl"
        } else {
            "Ll"
        }
    } else {
        let chars: Vec<char> = inx.chars().collect();
        if ends_with_opt8(&["m", "n", "w", "j", "l", "N"], &chars) {
            "Ls"
        } else {
            "Ds"
        }
    }
}

/// Port of ToneAssign — returns (phone, tone) with Python string-compare
/// semantics for `tone < '5'` (tone is a 0/1-char string; "" and digits).
pub fn tone_assign(keymatch: &str, phone_in: &str, codematch: &str, charmatch: &str) -> (String, String) {
    if phone_in.is_empty() {
        return (String::new(), "9".to_string());
    }
    let mut phone = phone_in.to_string();
    static RE_T8: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re_t8 = RE_T8.get_or_init(|| regex::Regex::new(r"[0-4]8").unwrap());
    if re_t8.is_match(&phone) {
        // Python: phone = re.sub(r'([0-4])8', r'\1', phone) — drop digits before '8'
        let cs: Vec<char> = phone.chars().collect();
        let mut cleaned = String::new();
        for (i, c) in cs.iter().enumerate() {
            if ('0'..='4').contains(c) && i + 1 < cs.len() && cs[i + 1] == '8' {
                continue;
            }
            cleaned.push(*c);
        }
        return (cleaned, String::new());
    }
    let mut lead = String::new();
    let init;
    let has_x = codematch.contains('X') || codematch == "GH" || codematch == "EF";
    static RE_AKCR: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re_akcr = RE_AKCR.get_or_init(|| regex::Regex::new("AK|CR").unwrap());
    if has_x {
        let lx: Vec<&str> = charmatch.split(' ').collect();
        lead = String::new();
        init = lx[0].to_string();
    } else if re_akcr.is_match(codematch) {
        let lx: Vec<&str> = charmatch.split(' ').collect();
        lead = lx[0].to_string();
        if lx.len() > 1 {
            init = lx[1].to_string();
        } else {
            init = String::new();
        }
    } else {
        init = String::new();
    }
    let deadsyll = dead_syl(&phone);

    // leading syllable "+"
    if phone.contains("+\'") {
        if "ผฝถขสหฉศษ".contains(&lead) || "กจดตบปอ".contains(&lead) {
            phone = phone.replace('+', "1");
        } else {
            phone = phone.replace('+', "3");
        }
    }

    let m1 = keymatch.contains('\u{0E48}');
    let m2 = keymatch.contains('\u{0E49}');
    let m3 = keymatch.contains('\u{0E4A}');
    let m4 = keymatch.contains('\u{0E4B}');

    let tone: &str = if "กจดตฎฏบปอ".contains(&init) {
        if deadsyll == "L" {
            if m1 { "1" } else if m2 { "2" } else if m3 { "3" } else if m4 { "4" } else { "0" }
        } else {
            if m1 { "9" } else if m2 { "2" } else if m3 { "3" } else if m4 { "4" } else { "1" }
        }
    } else if "ขฃฉฐถผฝสศษห".contains(&init)
        || ("งญณนมยรลวฬ".contains(&init) && !lead.is_empty() && "ขฃฉฐถผฝสศษห".contains(&lead))
    {
        if deadsyll == "L" {
            if m1 { "1" } else if m2 { "2" } else if m3 || m4 { "9" } else { "4" }
        } else {
            if m1 { "9" } else if m2 { "2" } else if m3 || m4 { "9" } else { "1" }
        }
    } else if "งญณนมยรลวฬ".contains(&init) && !lead.is_empty() && "กจดตฎฏบปอ".contains(&lead) {
        if deadsyll == "L" {
            if m1 { "1" } else if m2 { "2" } else if m3 { "3" } else if m4 { "4" } else { "0" }
        } else {
            if m1 { "9" } else if m2 { "2" } else if m3 { "3" } else if m4 { "4" } else { "1" }
        }
    } else if "คฅฆชฌซฑฒทธพภฟฮงญณนมยรลวฬฤฦ".contains(&init) {
        if deadsyll == "L" {
            if m1 { "2" } else if m2 { "3" } else if m3 || m4 { "9" } else { "0" }
        } else if has_double_vowel(&phone) {
            if m1 { "9" } else if m2 { "3" } else if m3 { "9" } else if m4 { "4" } else { "2" }
        } else {
            if m1 { "2" } else if m2 || m3 { "9" } else if m4 { "4" } else { "3" }
        }
    } else {
        // Python falls through: returns None implicitly -> tuple unpack would
        // fail. In practice every phone reaches an arm; mirror by "no tone".
        return (phone, String::new());
    };
    (phone, tone.to_string())
}

/// Python `tone < '5'` string comparison; tone may be "", a digit, or "9".
pub fn tone_lt5(tone: &str) -> bool {
    tone < "5"
}

/// Port of TransformSyl — returns (form, phone).
pub fn transform_syl(form: &str, phone: &str) -> (String, String) {
    let mut phone = phone.to_string();
    static RE1: std::sync::OnceLock<[regex::Regex; 3]> = std::sync::OnceLock::new();
    static RE_CASE1: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static RE_CASE3_F: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static RE_CASE3_P: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static RE_CASE3_NA: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static RE_HN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();

    let r = RE1.get_or_init(|| {
        [
            regex::Regex::new(r"xx[nmN][12]").unwrap(),
            regex::Regex::new(r"ee[nmN][12]").unwrap(),
            regex::Regex::new(r"\@\@[nmN][12]").unwrap(),
        ]
    });
    let re_case1 = RE_CASE1.get_or_init(|| regex::Regex::new(r"^อย่า$|^อยู่$|^อย่าง$|^อยาก$").unwrap());
    let re_c3f = RE_CASE3_F.get_or_init(|| regex::Regex::new(r"[จซศส]ร").unwrap());
    let re_c3p = RE_CASE3_P.get_or_init(|| regex::Regex::new(r"[cs]r").unwrap());
    let re_c3n = RE_CASE3_NA.get_or_init(|| regex::Regex::new(r"[^']").unwrap());
    let re_hn = RE_HN.get_or_init(|| regex::Regex::new(r"หนุ$|หก|หท|หพ|หฤ|หโ").unwrap());

    if r[0].is_match(&phone) {
        phone = phone.replace("xx", "x");
    } else if r[1].is_match(&phone) {
        phone = phone.replace("ee", "e");
    } else if r[2].is_match(&phone) {
        phone = phone.replace("@@", "@");
    }
    if (re_case1.is_match(form) && phone.contains('\''))
        || (form.contains('ห') && phone.contains("ha1") && !re_hn.is_match(form))
    {
        if let Some(pos) = phone.rfind('\'') {
            phone = phone[pos + 1..].to_string();
        }
    } else if re_c3f.is_match(form) && re_c3p.is_match(&phone) && re_c3n.is_match(&phone) {
        phone = phone.replace('r', "");
    }
    (form.to_string(), phone)
}

/// Port of ProbPhone.
pub fn prob_phone(sess: &Session, p: &str, pw: &str, w: &str, nw: &str) -> f64 {
    let dbg = debug_enabled();
    let m = &sess.m.phs;
    let id = |s: &str| -> u32 { m.syl_id(s) };
    let (iw, iww, inw, ip) = (id(w), id(pw), id(nw), id(p));
    let k4 = [iww, iw, inw, ip];
    let k3a = [iww, iw, ip];
    let k3b = [iw, inw, ip];
    let k2 = [iw, ip];
    let _k1 = [iw];

    let mut p3 = 0.0;
    let mut p2 = 0.0;
    let mut p1 = 0.0;
    let mut p0 = 0.0;
    let tri = m.phs_tri_get(&k4);
    if tri > 0.0 {
        p3 = (1.0 + tri.ln()) / (1.0 + m.frm_tri_get(&[iww, iw, inw]).ln());
    }
    let bi1 = m.phs_bi_get(&k3a);
    if bi1 > 0.0 {
        p2 = (1.0 + bi1.ln()) / (1.0 + m.frm_bi_get(&[iww, iw]).ln()) * 0.25;
    }
    let bi2 = m.phs_bi_get(&k3b);
    if bi2 > 0.0 {
        p2 += (1.0 + bi2.ln()) / (1.0 + m.frm_bi_get(&[iw, inw]).ln()) * 0.75;
    }
    let uni = m.phs_uni_get(&k2);
    if uni > 0.0 {
        p1 = (1.0 + uni.ln()) / (1.0 + m.frm_uni_get(&[iw]).ln());
    }

    // abstract forms
    let abs_w: String = w
        .chars()
        .filter(|c| !matches!(c, '\u{0E48}' | '\u{0E49}' | '\u{0E4A}' | '\u{0E4B}'))
        .map(|c| if ('\u{0E01}'..='\u{0E2E}').contains(&c) { 'C' } else { c })
        .collect();
    let abs_p: String = p
        .chars()
        .filter(|c| !c.is_ascii_digit())
        .map(|c| if "aeio@OuxU".contains(c) { c } else { 'C' })
        .collect();
    let aw = m.syl_id(&abs_w);
    let ap = m.syl_id(&abs_p);
    if aw != u32::MAX && ap != u32::MAX {
        let a = m.abs_uni_get(&[aw, ap]);
        if a > 0.0 {
            p0 = (1.0 + a.ln()) / (1.0 + m.abs_frm_get(&[aw]).ln());
        }
    }
    let prob = 0.8 * p3 + 0.16 * p2 + 0.03 * p1 + 0.00001 * p0 + 0.00000000001;
    if dbg {
        eprintln!(
            "prob_phone p={:?} pw={:?} w={:?} nw={:?} -> p3={} p2={} p1={} p0={} prob={}",
            p, pw, w, nw, p3, p2, p1, p0, prob
        );
    }
    prob
}

impl<'a> Session<'a> {
    /// Port of g2p.
    pub fn g2p(&mut self, input: &str) -> String {
        let mut output = String::new();
        let input = preprocess(input);
        for s in input.split(SEG_SEP) {
            for inp in s.split(SSEG_SEP) {
                if inp.is_empty() {
                    continue;
                }
                let out = if starts_non_thai(inp) {
                    format!("{}<tr/>{}", inp, inp)
                } else {
                    let y = self.sylparse(inp);
                    self.wordparse(&y)
                };
                output.push_str(&out);
                output.push(WORD_SEP);
            }
            output.push_str("<s/>");
        }
        output
    }

    /// Port of sylparse — returns syllable forms joined by '~', with PRONUN
    /// populated in self.pronun (cleared per call, like the Python global).
    pub fn sylparse(&mut self, input: &str) -> String {
        self.pronun.clear();
        let dbg = debug_enabled();
        let offs: Vec<usize> = input.char_indices().map(|(b, _)| b).collect();
        let n = offs.len();
        let mut schart: Vec<SpanMap<Vec<String>>> = (0..=n).map(|_| SpanMap::new()).collect();
        let mut prob_end: HashMap<(usize, usize), f64> = HashMap::new();
        static RE_THORR: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_THR: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re_thorr = RE_THORR.get_or_init(|| regex::Regex::new(r"^ทร").unwrap());
        let re_thr = RE_THR.get_or_init(|| regex::Regex::new(r"^thr").unwrap());

        for pp in &self.m.pron {
            for i in 0..n {
                let rest = &input[offs[i]..];
                let m = match pp.re.captures(rest) {
                    Some(m) => m,
                    None => continue,
                };
                let whole = m.get(0).unwrap();
                let keymatch = whole.as_str().to_string();
                let ng = pp.n_groups;
                let charmatch: String = if ng >= 3 {
                    format!("{} {} {}", g1(&m), g2(&m), g3(&m))
                } else if ng == 2 {
                    format!("{} {}", g1(&m), g2(&m))
                } else if ng == 1 {
                    g1(&m)
                } else {
                    self.pronun
                        .entry(keymatch.clone())
                        .or_default()
                        .extend(pp.codes.iter().cloned());
                    String::new()
                };
                let k = i + whole.as_str().chars().count();
                schart[i].set(k, vec![keymatch.clone()]);
                let dbg2 = dbg;
                for pronf in &pp.codes {
                    let codematch: String = pronf
                        .chars()
                        .filter(|c| "AKYDZCRX".contains(*c))
                        .collect();
                    if !codematch.is_empty() {
                        let phone = replace_snd(pronf, &codematch, &charmatch);
                        if not_exception_syl(&codematch, &charmatch, &keymatch, &phone) {
                            let (mut phone, tone) =
                                tone_assign(&keymatch, &phone, &codematch, &charmatch);
                            let (_, new_phone) = transform_syl(&keymatch, &phone);
                            phone = new_phone;
                            let keymatch2 = keymatch.clone();
                            if tone_lt5(&tone) {
                                let t = &tone;
                                let replaced: String = if t.is_empty() {
                                    phone.clone()
                                } else {
                                    phone.replace('8', t)
                                };
                                phone = replaced;
                                let form_key: String =
                                    schart[i].get(k).unwrap().concat();
                                if dbg2 {
                                    eprintln!("pronun append {:?} <- {:?} (pat={:?} code={:?} tone={:?})", form_key, phone, pp.pattern, pronf, tone);
                                }
                                self.pronun.entry(form_key).or_default().push(phone.clone());
                            } else if dbg2 {
                                eprintln!("pronun SKIP (tone {:?}) pat={:?} code={:?}", tone, pp.pattern, pronf);
                            }
                            // ทร -> s variant
                            let km: String = keymatch2;
                            if re_thorr.is_match(&km) && re_thr.is_match(&phone) {
                                let phone2 = phone.replace("thr", "s");
                                let form_key: String =
                                    schart[i].get(k).unwrap().concat();
                                self.pronun.entry(form_key).or_default().push(phone2);
                            }
                        } else if dbg2 {
                            eprintln!("pronun EXCEPT pat={:?} code={:?}", pp.pattern, pronf);
                        }
                    } else {
                        let form_key: String = schart[i].get(k).unwrap().concat();
                        if dbg2 {
                            eprintln!("pronun append(raw) {:?} <- {:?} pat={:?}", form_key, pronf, pp.pattern);
                        }
                        self.pronun.entry(form_key).or_default().push(pronf.clone());
                    }
                    prob_end.insert(
                        (i, k),
                        self.prob_trisyl(schart[i].get(k).unwrap()),
                    );
                }
            }
        }

        for j in 0..n {
            if let Some(s1) = schart[0].get(j).cloned() {
                let mut row0 = std::mem::take(&mut schart[0]);
                for &(k, ref s2) in &schart[j].items {
                    let mut merged: Vec<String> = Vec::with_capacity(s1.len() + s2.len());
                    merged.extend_from_slice(&s1);
                    merged.extend_from_slice(s2);
                    let tmp = mergekaran1(self, &merged);
                    if !row0.contains(k) {
                        row0.set(k, tmp.clone());
                        prob_end.insert((0, k), self.prob_trisyl(&tmp));
                    } else {
                        let p = self.prob_trisyl(&tmp);
                        let cur = prob_end.get(&(0, k)).copied().unwrap_or(0.0);
                        if p > cur {
                            row0.set(k, tmp.clone());
                            prob_end.insert((0, k), p);
                        }
                    }
                }
                schart[0] = row0;
            }
        }

        if let Some(final_syls) = schart[0].get(n) {
            return final_syls.join("~");
        }
        format!("<Fail>{}</Fail>", input)
    }
}

fn g1(m: &regex::Captures) -> String {
    m.get(1).map(|g| g.as_str().to_string()).unwrap_or_default()
}
fn g2(m: &regex::Captures) -> String {
    m.get(2).map(|g| g.as_str().to_string()).unwrap_or_default()
}
fn g3(m: &regex::Captures) -> String {
    m.get(3).map(|g| g.as_str().to_string()).unwrap_or_default()
}

/// Port of mergekaran1 — also merges PRONUN entries of the karan syllable.
pub fn mergekaran1(sess: &mut Session, lst: &[String]) -> Vec<String> {
    static RE_KARAN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE_KARAN.get_or_init(|| regex::Regex::new(r"(.+)์").unwrap());
    let mut rs: Vec<String> = Vec::new();
    let mut found = false;
    let mut x = String::new();
    for s in lst.iter().rev() {
        if re.is_match(s) && s.chars().count() < 4 {
            found = true;
            x = s.clone();
            continue;
        }
        if found {
            // merge pronunciations of (s+x) from PRONUN[s]
            let pron_s = sess.pronun.get(s).cloned().unwrap_or_default();
            let target = format!("{}{}", s, x);
            let entry = sess.pronun.entry(target.clone()).or_default();
            for ph in pron_s {
                if !entry.contains(&ph) {
                    entry.push(ph);
                }
            }
            let mut t = s.clone();
            t.push_str(&x);
            rs.push(t);
            found = false;
        } else {
            rs.push(s.clone());
        }
    }
    rs.reverse();
    rs
}

/// Port of Pcompose — initial consonant(s) + tone digit from a phone string.
pub fn pcompose(ph: &str) -> (String, String) {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"^(.+?)[aeiouxOU\@]+.*([0-4])$").unwrap());
    match re.captures(ph) {
        Some(m) => (
            m.get(1).map(|g| g.as_str().to_string()).unwrap_or_default(),
            m.get(2).map(|g| g.as_str().to_string()).unwrap_or_default(),
        ),
        None => (String::new(), String::new()), // Python would AttributeError
    }
}

/// Port of SelectPhones — pick the most probable pronunciation per syllable of a word.
pub fn select_phones(sess: &Session, slst_in: &[String]) -> String {
    let mut out: Vec<String> = Vec::new();
    let bar = "|".to_string();
    let mut slst: Vec<String> = Vec::with_capacity(slst_in.len() + 2);
    slst.push(bar.clone());
    slst.extend(slst_in.iter().cloned());
    slst.push(bar);

    for i in 1..slst.len() - 1 {
        let cands = sess.pronun.get(&slst[i]).cloned().unwrap_or_default();
        if cands.len() == 1 {
            out.push(cands[0].clone());
            continue;
        }
        let mut prmax = 0.0f64;
        let mut outp = String::new();
        for p in &cands {
            let pr = prob_phone(sess, p, &slst[i - 1], &slst[i], &slst[i + 1]);
            if pr > prmax || (pr == prmax && p.contains('\'') && p.chars().count() > outp.chars().count()) {
                prmax = pr;
                outp = p.clone();
            }
        }
        out.push(outp);
    }
    out.join("~")
}

impl<'a> Session<'a> {
    /// Port of wordparse — word segmentation over parsed syllables + phone selection.
    pub fn wordparse(&mut self, input: &str) -> String {
        static RE_MY: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re_my = RE_MY.get_or_init(|| {
            regex::Regex::new(r"(<tr/>|\|)([?a-zENOU0-9~'@^]+?)[|~]ๆ").unwrap()
        });
        let parts: Vec<&str> = input.split(SEG_SEP).collect();
        if let Some(&inx) = parts.first() {
            let syl_lst: Vec<String> = inx.split(SYL_SEP).map(|s| s.to_string()).collect();
            let n = syl_lst.len();
            let mut chart: Vec<SpanMap<Vec<String>>> = (0..=n).map(|_| SpanMap::new()).collect();
            let mut colloc_st: HashMap<(usize, usize), f64> = HashMap::new();

            for i in 0..n {
                chart[i].set(i + 1, vec![syl_lst[i].clone()]);
            }
            for i in 0..n {
                for j in i..=n {
                    chart[i].set(j, vec![syl_lst[i..j].join("~")]);
                    if j > i + 1 {
                        let mut st = 0.0f64;
                        for ii in i..(j - 1) {
                            st += self.compute_colloc_mi(&syl_lst[ii], &syl_lst[ii + 1]);
                        }
                        colloc_st.insert((i, j), st);
                    } else {
                        colloc_st.insert((i, j), 0.0);
                    }
                }
            }
            if self.chart_parse(&mut chart, &mut colloc_st, n) {
                let words = chart[0].get(n).unwrap();
                let mut outx: String = words.join("|");
                outx = outx.replace("~ๆ", "|ๆ");
                outx.push_str("<tr/>");
                let mut outp: Vec<String> = Vec::new();
                for wx in words {
                    let tmp: Vec<String> = wx.split(SYL_SEP).map(|s| s.to_string()).collect();
                    outp.push(select_phones(self, &tmp));
                }
                outx.push_str(&outp.join("|"));
                // ๆ duplication: (<tr/>|\|)([?a-zENOU0-9~'@^]+?)[|~]ๆ -> \1\2|\2
                let s = outx;
                outx = re_my.replace_all(&s, "${1}${2}|${2}").into_owned();
                outx
            } else {
                format!("<Fail>{}</Fail>", input)
            }
        } else {
            String::new()
        }
    }

    /// Port of chart_parse (maximal collocation chain). Returns true when a full
    /// 0..EndOfSent chain exists; leaves the best chain in chart[0][EndOfSent].
    pub fn chart_parse(
        &mut self,
        chart: &mut [SpanMap<Vec<String>>],
        colloc_st: &mut HashMap<(usize, usize), f64>,
        end_of_sent: usize,
    ) -> bool {
        for j in 0..end_of_sent {
            // Python: `if j in chart[0]` only — CollocSt is a defaultdict(float)
            if let Some(s1) = chart[0].get(j).cloned() {
                let mut row0 = std::mem::take(&mut chart[0]);
                let st0j = colloc_st.get(&(0, j)).copied().unwrap_or(0.0);
                for &(k, ref s2) in &chart[j].items {
                    let stjk = colloc_st.get(&(j, k)).copied().unwrap_or(0.0);
                    let combined = st0j + stjk;
                    let mut newlist: Vec<String> = Vec::with_capacity(s1.len() + s2.len());
                    newlist.extend_from_slice(&s1);
                    newlist.extend_from_slice(s2);
                    if !row0.contains(k) {
                        row0.set(k, newlist);
                        colloc_st.insert((0, k), combined);
                    } else if combined > colloc_st.get(&(0, k)).copied().unwrap_or(0.0) {
                        colloc_st.insert((0, k), combined);
                        row0.set(k, newlist);
                    }
                }
                chart[0] = row0;
            }
        }
        chart[0].contains(end_of_sent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedded;
    use crate::Tltk;

    #[test]
    fn test_prob_phone_abs_w_vowel_range() {
        let tltk = Tltk::new(embedded::models());
        let sess = tltk.session();

        // In Python 1.10 ProbPhone:
        // abs_w = re.sub(r"[ก-ฮ]", "C", abs_w) maps only consonants ก-ฮ (0E01..=0E2E).
        // For w="การ" (ก + า + ร), consonants ก and ร become 'C', preserving vowel 'า' -> "CาC".
        // abs_p for "kaan0" becomes "CaaC".
        // AbsUnigram[("CาC", "CaaC")] = 4135.0, yielding p0 ~ 0.9628.
        // Under the buggy range 0E01..=0E4A, vowel 'า' was also mapped to 'C' -> "CCC",
        // where AbsUnigram[("CCC", "CaaC")] was 0, incorrectly zeroing p0.
        let prob = prob_phone(&sess, "kaan0", "|", "การ", "|");
        // Oracle ProbPhone('kaan0', '|', 'การ', '|') == 0.988481885347311
        assert!((prob - 0.988481885347311).abs() < 1e-6);
    }
}
