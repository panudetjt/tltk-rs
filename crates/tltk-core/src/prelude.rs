//! Shared text preprocessing and character-class helpers (port of nlp.py preprocess).

pub const SYL_SEP: char = '~'; // chr(126)
pub const WORD_SEP: char = '|'; // chr(124)
pub const SSEG_SEP: char = '\u{1E}'; // chr(30)
pub const SEG_SEP: char = '\u{1F}'; // chr(31)

pub const SEG_SEP_STR: &str = "\u{1F}";
pub const SSEG_SEP_STR: &str = "\u{1E}";

/// Thai range used by `re.match(r"[^ก-์]+", inp)` — ก (U+0E01) .. ์ (U+0E4E)
pub fn is_thai_char(c: char) -> bool {
    ('\u{0E01}'..='\u{0E4E}').contains(&c)
}

/// Python: re.match(r"[^ก-์]+", inp) — some match iff first char is NOT in [ก-์].
pub fn starts_non_thai(inp: &str) -> bool {
    match inp.chars().next() {
        None => false, // re.match on empty string fails for `+`
        Some(c) => !is_thai_char(c),
    }
}

pub struct Res {
    pub p_sp_yamok: regex::Regex,        // " +ๆ" -> "ๆ"
    pub p_yamok_nonsp: regex::Regex,     // ๆ([^ ]) -> ๆ<SEG>\1
    pub p_ch1: regex::Regex,             // char-1-char-2 space collapse
    pub p_seg1: regex::Regex,            // ([^\s\t\u00A0]{3,})[space]+([^\s\t\u00A0]+?)
    pub p_seg2: regex::Regex,            // ([^\s\t\u00A0]+)[space]+([0-9]+)
    pub p_seg3: regex::Regex,            // ([^\s\t\u00A0]+?)[space]+([^\s\t\u00A0]{3,})
    pub p_ss1: regex::Regex,             // ([ก-์][ฯๆ]) -> \1<SSEG>
    pub p_ss2: regex::Regex,             // Thai. -> Thai<SSEG>non
    pub p_ss3: regex::Regex,             // non Thai -> <SSEG>Thai
    pub p_ss4: regex::Regex,             // (<.+?>) -> <SSEG>\1
    pub p_ss5: regex::Regex,             // ([0-9a-zA-Z.\-]{2,})([Thai]+)
    pub p_ss6: regex::Regex,             // (\.\.\.+)
}

static RES: std::sync::OnceLock<Res> = std::sync::OnceLock::new();

pub fn res() -> &'static Res {
    RES.get_or_init(|| Res {
        p_sp_yamok: regex::Regex::new(r" +ๆ").unwrap(),
        p_yamok_nonsp: regex::Regex::new(r"ๆ([^ ])").unwrap(),
        p_ch1: regex::Regex::new(concat!(
            r"([ก-ฮเแาำะไใโฯๆ][\u{0E31}\u{0E34}\u{0E35}\u{0E36}\u{0E37}\u{0E38}\u{0E39}\u{0E47}\u{0E48}\u{0E49}\u{0E4A}\u{0E4B}\u{0E4C}]*) +",
            r"([ก-ฮเแาำะไใโฯๆ][\u{0E31}\u{0E34}\u{0E35}\u{0E36}\u{0E37}\u{0E38}\u{0E39}\u{0E47}\u{0E48}\u{0E49}\u{0E4A}\u{0E4B}\u{0E4C}]*) +|",
            r"([ก-ฮเแาำะไใโฯๆ][\u{0E31}\u{0E34}\u{0E35}\u{0E36}\u{0E37}\u{0E38}\u{0E39}\u{0E47}\u{0E48}\u{0E49}\u{0E4A}\u{0E4B}\u{0E4C}]*) +",
            r"([ก-ฮเแาำะไใโฯๆ][\u{0E31}\u{0E34}\u{0E35}\u{0E36}\u{0E37}\u{0E38}\u{0E39}\u{0E47}\u{0E48}\u{0E49}\u{0E4A}\u{0E4B}\u{0E4C}]*)$"
        ))
        .unwrap(),
        p_seg1: regex::Regex::new(r"([^\s\t\u{A0}]{3,})[\s\t\u{A0}]+([^\s\t\u{A0}]+?)").unwrap(),
        p_seg2: regex::Regex::new(r"([^\s\t\u{A0}]+)[\s\t\u{A0}]+([0-9]+)").unwrap(),
        p_seg3: regex::Regex::new(r"([^\s\t\u{A0}]+?)[\s\t\u{A0}]+([^\s\t\u{A0}]{3,})").unwrap(),
        p_ss1: regex::Regex::new(r"([ก-์][ฯๆ])").unwrap(),
        p_ss2: regex::Regex::new(r"([\u{0E01}-\u{0E5B}]+\.?)([^\.\u{0E01}-\u{0E5B}\u{1F}]+)").unwrap(),
        p_ss3: regex::Regex::new(r"([^\.\u{0E01}-\u{0E5B}\u{1F}]+)([\u{0E01}-\u{0E5B}]+)").unwrap(),
        p_ss4: regex::Regex::new(r"(<.+?>)").unwrap(),
        p_ss5: regex::Regex::new(r"([0-9a-zA-Z\.\-]{2,})([\u{0E01}-\u{0E5B}]+)").unwrap(),
        p_ss6: regex::Regex::new(r"(\.\.\.+)").unwrap(),
    })
}

/// Port of nlp.preprocess — mutates text into SegSep/SSegSep-chunked form.
pub fn preprocess(input: &str) -> String {
    let r = res();
    let mut s = r.p_sp_yamok.replace_all(input, "ๆ").into_owned();
    s = r
        .p_yamok_nonsp
        .replace_all(&s, format!("ๆ{}$1", SEG_SEP))
        .into_owned();

    // NORMALIZE_DICT
    s = s.replace("\u{0E40}\u{0E40}", "\u{0E41}");
    s = s.replace("\u{0E4D}\u{0E32}", "\u{0E33}");
    s = s.replace("\u{0E24}\u{0E32}", "\u{0E24}\u{0E45}");
    s = s.replace("\u{0E26}\u{0E32}", "\u{0E26}\u{0E45}");

    s = r.p_ch1.replace_all(&s, "$1$2").into_owned();
    s = r.p_seg1.replace_all(&s, format!("$1{}$2", SEG_SEP)).into_owned();
    s = r.p_seg2.replace_all(&s, format!("$1{}$2", SEG_SEP)).into_owned();
    s = r.p_seg3.replace_all(&s, format!("$1{}$2", SEG_SEP)).into_owned();

    s = r.p_ss1.replace_all(&s, format!("$1{}", SSEG_SEP)).into_owned();
    s = r.p_ss2.replace_all(&s, format!("$1{}$2", SSEG_SEP)).into_owned();
    s = r.p_ss3.replace_all(&s, format!("$1{}$2", SSEG_SEP)).into_owned();
    s = r.p_ss4.replace_all(&s, format!("{}$1", SSEG_SEP)).into_owned();
    s = r.p_ss5.replace_all(&s, format!("$1{}$2", SSEG_SEP)).into_owned();
    s = r.p_ss6
        .replace_all(&s, format!("{}$1{}", SSEG_SEP, SSEG_SEP))
        .into_owned();
    s
}

/// Python float repr (shortest round-trip with ".0" and exponent rules) — used
/// where the server renders Python values into strings. For the functions this
/// crate serves, only `%.4f` (see fmt4) appears; repr is provided for parity
/// tooling.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let ax = x.abs();
    if x != 0.0 && (ax >= 1e16 || ax < 1e-4) {
        // Python repr switches to exponent notation; e.g. 1e-05, 1.234e+16
        let s = format!("{:e}", x);
        // Rust: "1.234e-5" ; Python: "1.234e-05" — pad exponent to 2 digits
        if let Some(pos) = s.find('e') {
            let (mant, exp) = s.split_at(pos);
            let exp = &exp[1..];
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exp),
            };
            if digits.len() < 2 {
                return format!("{}e{}0{}", mant, sign, digits);
            }
            return format!("{}e{}{}", mant, sign, digits);
        }
        s
    } else {
        let s = format!("{}", x);
        if s.contains('.') || s.contains('e') {
            s
        } else {
            format!("{}.0", s)
        }
    }
}

/// Python "%.4f" formatting (round-half-even on the exact binary value —
/// identical output to Rust's {:.4}).
pub fn fmt4(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    format!("{:.4}", x)
}

/// Python str() of a list of strings: `['a', 'b']` (repr of each item).
pub fn py_list_repr(items: &[String]) -> String {
    let mut out = String::from("[");
    for (i, s) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&py_str_repr(s));
    }
    out.push(']');
    out
}

/// Python repr() of a str (single quotes preferred, escape the quote kind).
pub fn py_str_repr(s: &str) -> String {
    let has_sq = s.contains('\'');
    let has_dq = s.contains('"');
    let (quote, esc) = if has_sq && !has_dq { ('"', true) } else { ('\'', has_sq) };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if esc && c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}
