//! Transliteration: th2ipa, th2roman, th2read + PhoneToThai (ports of the
//! same-named nlp.py functions).

use crate::g2p::dead_sylx;
use crate::Session;

const IPA_NORMALIZE: &[(&str, &str)] = &[
    ("O", "\u{1D10}"),
    ("x", "\u{025B}"),
    ("@", "\u{0264}"),
    ("N", "\u{014B}"),
    ("?", "\u{0294}"),
    ("U", "\u{026F}"),
    ("|", " "),
    ("~", "."),
    ("^", "."),
    ("'", "."),
    ("4", "5"),
    ("3", "4"),
    ("2", "3"),
    ("1", "2"),
    ("0", "1"),
];

static RE_ASP: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
fn asp_re() -> &'static regex::Regex {
    RE_ASP.get_or_init(|| regex::Regex::new(r"([ptkc])h").unwrap())
}

/// Python re.sub(r"([aeiouUxO@])\1", repl_with_backref, tran) — the regex crate
/// has no backreferences, so scan for doubled vowel chars manually.
/// `suffix` is what follows the retained vowel ("ː" for IPA, "" for roman).
fn sub_double_vowel(tran: &str, suffix: &str) -> String {
    const V: &[char] = &['a', 'e', 'i', 'o', 'u', 'U', 'x', 'O', '@'];
    let chars: Vec<char> = tran.chars().collect();
    let mut out = String::with_capacity(tran.len());
    let mut i = 0;
    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == chars[i + 1] && V.contains(&chars[i]) {
            out.push(chars[i]);
            out.push_str(suffix);
            i += 2;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn split_g2p_seg(seg: &str) -> Option<(String, String)> {
    let idx = seg.find("<tr/>")?;
    Some((seg[..idx].to_string(), seg[idx + 5..].to_string()))
}

impl<'a> Session<'a> {
    /// Port of th2ipa.
    pub fn th2ipa(&mut self, txt: &str) -> String {
        let mut out = String::new();
        let inx = self.g2p(txt);
        for seg in inx.split("<s/>") {
            if seg.is_empty() {
                continue;
            }
            let (_th, tran) = match split_g2p_seg(seg) {
                Some(p) => p,
                None => continue, // Python would ValueError; skip defensively
            };
            let mut tran = sub_double_vowel(&tran, "ː");
            tran = asp_re().replace_all(&tran, "${1}ʰ").into_owned();
            for (k, v) in IPA_NORMALIZE {
                tran = tran.replace(k, v);
            }
            out.push_str(&tran);
            out.push_str("<s/>");
        }
        out
    }

    /// Port of th2roman.
    pub fn th2roman(&mut self, txt: &str) -> String {
        const NORMALIZE_ROM: &[(&str, &str)] = &[
            ("O", "o"),
            ("x", "ae"),
            ("@", "oe"),
            ("N", "ng"),
            ("U", "ue"),
            ("aw", "ao"),
            ("iw", "io"),
            ("ew", "eo"),
            ("?", ""),
            ("|", " "),
            ("~", "-"),
            ("^", "-"),
            ("'", "-"),
        ];
        static RE_J: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_CH: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_DASH: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re_j = RE_J.get_or_init(|| regex::Regex::new(r"([aeiou])j").unwrap());
        let re_ch = RE_CH.get_or_init(|| regex::Regex::new(r"c([^h])").unwrap());
        let re_dash = RE_DASH.get_or_init(|| regex::Regex::new(r"\-([^aeiou])").unwrap());
        static RE_DIGIT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re_digit = RE_DIGIT.get_or_init(|| regex::Regex::new(r"[0-9]").unwrap());

        let mut out = String::new();
        let inx = self.g2p(txt);
        for seg in inx.split("<s/>") {
            if seg.is_empty() {
                continue;
            }
            let (_th, tran) = match split_g2p_seg(seg) {
                Some(p) => p,
                None => continue,
            };
            let mut tran = sub_double_vowel(&tran, "");
            tran = re_digit.replace_all(&tran, "").into_owned();
            for (k, v) in NORMALIZE_ROM {
                tran = tran.replace(k, v);
            }
            tran = re_j.replace_all(&tran, "${1}i").into_owned();
            tran = tran.replace('j', "y");
            tran = re_ch.replace_all(&tran, "ch${1}").into_owned();
            tran = re_dash.replace_all(&tran, "${1}").into_owned();
            out.push_str(&tran);
            out.push_str("<s/>");
        }
        out
    }

    /// Port of th2read.
    pub fn th2read(&mut self, txt: &str) -> Result<String, crate::ProcessError> {
        let mut out = String::new();
        static RE_SPL: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re_spl = RE_SPL.get_or_init(|| regex::Regex::new(r"~|\||\^|'").unwrap());
        let inx = self.g2p(txt);
        for seg in inx.split("<s/>") {
            if seg.is_empty() {
                continue;
            }
            let (_th, tran) = match split_g2p_seg(seg) {
                Some(p) => p,
                None => continue,
            };
            for syl in re_spl.split(&tran) {
                if !syl.is_empty() {
                    let thread = phone_to_thai(syl)
                        .map_err(|k| crate::ProcessError(format!("'{}'", k)))?;
                    out.push_str(&thread);
                    out.push('-');
                }
            }
        }
        Ok(out)
    }
}

fn init_ct(s: &str) -> Option<&'static str> {
    Some(match s {
        "w" => "ว", "r" => "ร", "t" => "ต", "th" => "ท", "p" => "ป", "ph" => "พ",
        "s" => "ซ", "d" => "ด", "f" => "ฟ", "h" => "ฮ", "j" => "ย", "k" => "ก",
        "kh" => "ค", "l" => "ล", "c" => "จ", "ch" => "ช", "b" => "บ", "n" => "น",
        "N" => "ง", "m" => "ม", "?" => "อ",
        "br" => "บ\u{0E3A}ร", "bl" => "บ\u{0E3A}ล", "dr" => "ด\u{0E3A}ร",
        "dl" => "ด\u{0E3A}ล", "kr" => "ก\u{0E3A}ร", "kl" => "ก\u{0E3A}ล",
        "kw" => "ก\u{0E3A}ว", "khr" => "ค\u{0E3A}ร", "khl" => "ค\u{0E3A}ล",
        "khw" => "ค\u{0E3A}ว", "tr" => "ต\u{0E3A}ร", "thr" => "ท\u{0E3A}ร",
        "thl" => "ท\u{0E3A}ล", "pr" => "ป\u{0E3A}ร", "pl" => "ป\u{0E3A}ล",
        "phr" => "พ\u{0E3A}ร", "phl" => "พ\u{0E3A}ล", "fr" => "ฟ\u{0E3A}ร",
        "fl" => "ฟ\u{0E3A}ล", "sr" => "ส\u{0E3A}ร",
        _ => return None,
    })
}

fn final_ct(s: &str) -> Option<&'static str> {
    Some(match s {
        "w" => "ว", "t" => "ด", "p" => "บ", "s" => "ส", "f" => "ฟ", "j" => "ย",
        "k" => "ก", "l" => "ล", "n" => "น", "N" => "ง", "m" => "ม", "c" => "จ",
        "ch" => "ช",
        _ => return None,
    })
}

fn vowel_t(s: &str) -> Option<&'static str> {
    Some(match s {
        "a" => "xTะ", "aa" => "xTา", "i" => "xิT", "ii" => "xีT", "e" => "เxTะ",
        "ee" => "เxT", "u" => "xุT", "uu" => "xูT", "o" => "โxTะ", "oo" => "โxT",
        "x" => "แxTะ", "xx" => "แxT", "O" => "เxTาะ", "OO" => "xTอ", "@" => "เxTอะ",
        "@@" => "เxTอ", "U" => "xึT", "UU" => "xือT", "ia" => "เxีTยะ",
        "iia" => "เxีTย", "ua" => "xัTวะ", "uua" => "xัTว", "Ua" => "เxืTอะ",
        "UUa" => "เxืTอ",
        _ => return None,
    })
}

/// Port of PhoneToThai — phone string to Thai spelling. Unknown initial/vowel/
/// final keys crash Python with KeyError (a 500 on the server), so they carry
/// an Err here; the tone-0 dead-syllable cases legitimately produce "?".
pub fn phone_to_thai(syl: &str) -> Result<String, String> {
    static RE_MT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re_mt = RE_MT.get_or_init(|| {
        regex::Regex::new(r"^(.+?)([aeioux@OU]+)(.*)([0-4])").unwrap()
    });
    let m = match re_mt.captures(syl) {
        Some(m) => m,
        None => return Ok("?".to_string()), // Python mt.group() would AttributeError
    };
    let g1 = m.get(1).map(|g| g.as_str()).unwrap_or("");
    let g2 = m.get(2).map(|g| g.as_str()).unwrap_or("");
    let g3 = m.get(3).map(|g| g.as_str()).unwrap_or("");
    let g4 = m.get(4).map(|g| g.as_str()).unwrap_or("");

    let it0 = match init_ct(g1) {
        Some(v) => v.to_string(),
        None => return Err(g1.to_string()), // Python KeyError
    };
    let vt0 = match vowel_t(g2) {
        Some(v) => v.to_string(),
        None => return Err(g2.to_string()),
    };
    let ft = if !g3.is_empty() {
        match final_ct(g3) {
            Some(v) => v.to_string(),
            None => return Err(g3.to_string()),
        }
    } else {
        String::new()
    };
    let mut it = it0;
    let mut vt = vt0;
    let tone = g4;
    let deadsyl = dead_sylx(syl);
    // Python re.search(r"[กจดตบปอ]", it) — ANY char of the mapped initial in the class
    let mid_cons = |s: &str| s.chars().any(|c| "กจดตบปอ".contains(c));
    let _high_cons = |s: &str| s.chars().any(|c| "ขฃฉฐถผฝสศษห".contains(c));

    let high_help = |it: &mut String| {
        *it = it.replace('ค', "ข");
        *it = it.replace('ฟ', "ฝ");
        *it = it.replace('พ', "ผ");
        *it = it.replace('ซ', "ส");
        *it = it.replace('ฮ', "ห");
        *it = it.replace('ช', "ฉ");
        *it = it.replace('ท', "ถ");
        *it = it.replace('ย', "หฺย");
        *it = it.replace('น', "หฺน");
        *it = it.replace('ง', "หฺง");
        *it = it.replace('ม', "หฺม");
        *it = it.replace('ว', "หฺว");
        *it = it.replace('ล', "หฺล");
        *it = it.replace('ร', "หฺร");
    };

    if deadsyl.contains('L') {
        if mid_cons(&it) {
            match tone {
                "1" => vt = vt.replace('T', "่"),
                "2" => vt = vt.replace('T', "้"),
                "3" => vt = vt.replace('T', "๊"),
                "4" => vt = vt.replace('T', "๋"),
                "0" => vt = vt.replace('T', ""),
                _ => {}
            }
        } else {
            match tone {
                "1" => {
                    high_help(&mut it);
                    vt = vt.replace('T', "่");
                }
                "2" => vt = vt.replace('T', "่"),
                "3" => vt = vt.replace('T', "้"),
                "4" => {
                    high_help(&mut it);
                    vt = vt.replace('T', "");
                }
                "0" => vt = vt.replace('T', ""),
                _ => {}
            }
        }
    } else if mid_cons(&it) {
        match tone {
            "1" => vt = vt.replace('T', ""),
            "2" => vt = vt.replace('T', "้"),
            "3" => vt = vt.replace('T', "๊"),
            "4" => vt = vt.replace('T', "๋"),
            "0" => return Ok("?".to_string()),
            _ => {}
        }
    } else if deadsyl == "Dl" {
        match tone {
            "1" => {
                high_help(&mut it);
                vt = vt.replace('T', "");
            }
            "2" => vt = vt.replace('T', ""),
            "3" => vt = vt.replace('T', "้"),
            "4" => vt = vt.replace('T', "๋"),
            "0" => return Ok("?".to_string()),
            _ => {}
        }
    } else {
        match tone {
            "1" => {
                high_help(&mut it);
                vt = vt.replace('T', "");
            }
            "2" => vt = vt.replace('T', "่"),
            "3" => vt = vt.replace('T', ""),
            "4" => vt = vt.replace('T', "๋"),
            "0" => return Ok("?".to_string()),
            _ => {}
        }
    }
    vt = vt.replace('x', &it);
    let mut tsyl = vt + &ft;
    if !ft.is_empty() {
        static RE_AE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_O: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_E_OE_A: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_E_OE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_E_AW: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_UE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_E: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_A_Y: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_A_W: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_A: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_UA: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_TONE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();

        let re_ae = RE_AE.get_or_init(|| regex::Regex::new(r"แ([ก-ฮ]?\u{0E3A}?[ก-ฮ])([่-๋]?)ะ").unwrap());
        let re_o = RE_O.get_or_init(|| regex::Regex::new(r"โ([ก-ฮ]?\u{0E3A}?[ก-ฮ])([่-๋]?)ะ").unwrap());
        let re_e_oe_a = RE_E_OE_A.get_or_init(|| regex::Regex::new(r"เ([ก-ฮ]?\u{0E3A}?[ก-ฮ])([่-๋]?)อะ").unwrap());
        let re_e_oe = RE_E_OE.get_or_init(|| regex::Regex::new(r"เ([ก-ฮ]?\u{0E3A}?[ก-ฮ])([่-๋]?)อ").unwrap());
        let re_e_aw = RE_E_AW.get_or_init(|| regex::Regex::new(r"เ([ก-ฮ]?\u{0E3A}?[ก-ฮ])([่-๋]?)าะ").unwrap());
        let re_ue = RE_UE.get_or_init(|| regex::Regex::new(r"^([ก-ฮ]?\u{0E3A}?[ก-ฮ])ื([่-๋]?)อ").unwrap());
        let re_e = RE_E.get_or_init(|| regex::Regex::new(r"เ([ก-ฮ]?\u{0E3A}?[ก-ฮ])([่-๋]?)ะ").unwrap());
        let re_a_y = RE_A_Y.get_or_init(|| regex::Regex::new(r"([ก-ฮ]?\u{0E3A}?[ก-ฮ])([่-๋]?)ะย").unwrap());
        tsyl = re_ae.replace_all(&tsyl, "แ${1}็${2}/").into_owned();
        tsyl = re_o.replace_all(&tsyl, "${1}${2}").into_owned();
        tsyl = re_e_oe_a.replace_all(&tsyl, "เ${1}ิ${2}").into_owned();
        tsyl = re_e_oe.replace_all(&tsyl, "เ${1}ิ${2}").into_owned();
        tsyl = re_e_aw.replace_all(&tsyl, "${1}็${2}อ").into_owned();
        tsyl = re_ue.replace_all(&tsyl, "${1}ื${2}").into_owned();
        tsyl = re_e.replace_all(&tsyl, "เ${1}็${2}").into_owned();
        tsyl = re_a_y.replace_all(&tsyl, "ไ${1}${2}").into_owned();
        if syl.contains("aw") {
            let re_a_w = RE_A_W.get_or_init(|| regex::Regex::new(r"([ก-ฮ]?\u{0E3A}?[ก-ฮ])([่-๋]?)ะว").unwrap());
            tsyl = re_a_w.replace_all(&tsyl, "เ${1}${2}า").into_owned();
        } else {
            let re_a = RE_A.get_or_init(|| regex::Regex::new(r"([ก-ฮ]?\u{0E3A}?[ก-ฮ])([่-๋]?)ะ").unwrap());
            tsyl = re_a.replace_all(&tsyl, "${1}ั${2}").into_owned();
        }
        let re_ua = RE_UA.get_or_init(|| regex::Regex::new(r"([ก-ฮ]?\u{0E3A}?[ก-ฮ])ั([่-๋]?)ว([กดบมนยง])").unwrap());
        let re_tone = RE_TONE.get_or_init(|| regex::Regex::new(r"็([่-๋])").unwrap());
        tsyl = re_ua.replace_all(&tsyl, "${1}${2}ว${3}").into_owned();
        tsyl = re_tone.replace_all(&tsyl, "${1}").into_owned();
    }
    Ok(tsyl)
}
