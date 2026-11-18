//! Pure-Rust forward pass for `iapp/OpenThai2.0-LATTE-WordSeg-Lite` v0.1.2.
//!
//! Upstream: <https://huggingface.co/iapp/OpenThai2.0-LATTE-WordSeg-Lite>.
//! The port surface is the wheel's `tokenize(text, keep_whitespace=True, unit)`
//! (Python list[str]); the wire returns that list as compact JSON
//! (`serde_json`, UTF-8 unescaped), matching `json.dumps(..., ensure_ascii=False)`.
//!
//! The 35-node ONNX graph (opset 17 + `com.microsoft` 1) is reproduced in f32:
//!   gather char/lex embeddings -> 3 conv1d+ReLU -> 2 x dynamic-quantized BiLSTM
//!   -> DynamicQuantizeLinear -> MatMulInteger -> bias -> transpose.
//! Pre/post-processing (64-bit poly hash lexicon bits, Thai orthography rules,
//! whitespace-as-token, ASCII/digit runs) follows `_runtime.py` verbatim.
//! See `docs/latte-onnx-analysis.md` for the graph and the empirically pinned
//! `DynamicQuantizeLSTM` semantics (round-to-nearest-even, dynamic scales for
//! the input and for the step's hidden state, per-direction i/o/f/c weights).
//!
//! Goldens are captured from the iApp wheel, never from this code.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use crate::{ProcessError, Session};

const MAGIC_W: &[u8; 4] = b"LT01";
const MAGIC_L: &[u8; 4] = b"LX01";
const HASH_MUL: u64 = 0x9E37_79B9_7F4A_7C15;
const HIDDEN: usize = 128;
const UNITS: [&str; 3] = ["long", "short", "minimal"];
const UNNESTED_MINIMAL: &str = "minimal";
/// `UNITS` is coarse to fine, so index 1 is `short`; that order is
/// load-bearing — the nested-unit `max` fold in `unit_logit` depends on it.
const SHORT_UNIT: usize = 1;
const NOISE_CLASS: i16 = 12;

const EMB_BYTES: &[u8] = include_bytes!("../data/latte_weights.bin.gz");
const LEX_BYTES: &[u8] = include_bytes!("../data/latte_lex.bin.gz");
const CHAR_BYTES: &[u8] = include_bytes!("../data/latte_charclass.bin.gz");

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0 }
    }
    fn uvar(&mut self) -> usize {
        let mut v = 0usize;
        let mut s = 0;
        loop {
            let x = self.b[self.p];
            self.p += 1;
            v |= ((x & 0x7F) as usize) << s;
            if x & 0x80 == 0 {
                return v;
            }
            s += 7;
        }
    }
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
    fn str(&mut self) -> String {
        let n = self.uvar();
        String::from_utf8_lossy(self.take(n)).into_owned()
    }
}

struct Tensor {
    data: Vec<u8>,
}
impl Tensor {
    fn f32s(&self) -> &[f32] {
        assert_eq!(self.data.len() % 4, 0);
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, self.data.len() / 4) }
    }
    fn i8s(&self) -> &[i8] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const i8, self.data.len()) }
    }
    fn scalar_f32(&self) -> f32 {
        self.f32s()[0]
    }
}

struct Model {
    chars: Vec<i16>,
    lex: Vec<(usize, Vec<u64>)>,
    t: HashMap<String, Tensor>,
}

fn gunzip(bytes: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut d = flate2::read::GzDecoder::new(bytes);
    let mut out = Vec::new();
    d.read_to_end(&mut out).expect("latte: corrupt gzip blob");
    out
}

static MODEL: LazyLock<Model> = LazyLock::new(|| {
    let wb = gunzip(EMB_BYTES);
    assert_eq!(&wb[..4], MAGIC_W);
    let mut r = Reader::new(&wb[4..]);
    let n = r.uvar();
    let mut t = HashMap::with_capacity(n);
    for _ in 0..n {
        let name = r.str();
        let _dtype = r.uvar();
        let rank = r.uvar();
        let dims: Vec<usize> = (0..rank).map(|_| r.uvar()).collect();
        let len = r.uvar();
        let data = r.take(len).to_vec();
        let _ = dims; // validated by the converter; shapes are baked into the kernels
        t.insert(name, Tensor { data });
    }

    let lb = gunzip(LEX_BYTES);
    assert_eq!(&lb[..4], MAGIC_L);
    let mut r = Reader::new(&lb[4..]);
    let n = r.uvar();
    let mut lex = Vec::with_capacity(n);
    for _ in 0..n {
        let len = r.uvar();
        let cnt = r.uvar();
        let raw = r.take(cnt * 8);
        let (chunks, _rem) = raw.as_chunks::<8>();
        let vals: Vec<u64> = chunks.iter().map(|c| u64::from_le_bytes(*c)).collect();
        lex.push((len, vals));
    }

    let cb = gunzip(CHAR_BYTES);
    let (chunks, _rem) = cb.as_chunks::<2>();
    let chars: Vec<i16> = chunks.iter().map(|c| i16::from_le_bytes(*c)).collect();

    Model { chars, lex, t }
});

macro_rules! f32v {
    ($m:expr, $name:expr) => {
        $m.t.get($name).unwrap_or_else(|| panic!("latte: missing {}", $name)).f32s()
    };
}
macro_rules! i8v {
    ($m:expr, $name:expr) => {
        $m.t.get($name).unwrap_or_else(|| panic!("latte: missing {}", $name)).i8s()
    };
}
macro_rules! scalarf {
    ($m:expr, $name:expr) => {
        $m.t.get($name).unwrap_or_else(|| panic!("latte: missing {}", $name)).scalar_f32()
    };
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// DynamicQuantizeLinear: whole-tensor uint8 quantization with ties-to-even.
fn dql(x: &[f32]) -> (Vec<u8>, f32, i32) {
    let mut mn = f32::MAX;
    let mut mx = f32::MIN;
    for &v in x {
        if v < mn { mn = v; }
        if v > mx { mx = v; }
    }
    let scale = if mx > mn { (mx - mn) / 255.0 } else { 1.0 };
    let zp = ((-mn / scale).round_ties_even() as i32).clamp(0, 255);
    let q: Vec<u8> = x
        .iter()
        .map(|&v| ((v / scale).round_ties_even() as i32 + zp).clamp(0, 255) as u8)
        .collect();
    (q, scale, zp)
}

/// Conv1D over [channels, time] -> [out_channels, time] with replicate-free
/// zero padding, matching ONNX Conv pads=[p,p] (k=2p+1).
fn conv1d(x: &[f32], w: &[f32], b: &[f32], c: usize, t: usize, o: usize, kp: (usize, usize)) -> Vec<f32> {
    let (k, p) = kp;
    let mut out = vec![0.0f32; o * t];
    // Accumulate in the same order as the reference: for each kernel offset,
    // for each input channel, fold into the output channels.
    for kk in 0..k {
        for cc in 0..c {
            for oo in 0..o {
                let weight = w[oo * c * k + cc * k + kk];
                for tt in 0..t {
                    let xi = tt + kk;
                    if xi >= p && xi < p + t {
                        out[oo * t + tt] += x[cc * t + (xi - p)] * weight;
                    }
                }
            }
        }
    }
    for oo in 0..o {
        for tt in 0..t {
            out[oo * t + tt] += b[oo];
            if out[oo * t + tt] < 0.0 {
                out[oo * t + tt] = 0.0; // ReLU
            }
        }
    }
    out
}

/// One BiLSTM layer; X is [T, input], result is [T, 2*HIDDEN].
fn lstm(m: &Model, x: &[f32], t: usize, input: usize, layer: usize) -> Vec<f32> {
    let (xq, xs, xz) = dql(x);
    let mut out = vec![0.0f32; t * 2 * HIDDEN];
    for dir in 0..2 {
        let wq = i8v!(m, &format!("lstm{layer}_W_quantized"));
        let rq = i8v!(m, &format!("lstm{layer}_R_quantized"));
        let wscale = {
            let t = m.t.get(&format!("lstm{layer}_W_scale")).unwrap();
            t.f32s()[dir]
        };
        let rscale = {
            let t = m.t.get(&format!("lstm{layer}_R_scale")).unwrap();
            t.f32s()[dir]
        };
        let b = f32v!(m, &format!("lstm{layer}_B"));
        let wb = &b[dir * 8 * HIDDEN..dir * 8 * HIDDEN + 4 * HIDDEN];
        let rb = &b[dir * 8 * HIDDEN + 4 * HIDDEN..dir * 8 * HIDDEN + 8 * HIDDEN];
        let wqd = &wq[dir * input * 4 * HIDDEN..(dir + 1) * input * 4 * HIDDEN];
        let rqd = &rq[dir * HIDDEN * 4 * HIDDEN..(dir + 1) * HIDDEN * 4 * HIDDEN];

        let mut h = vec![0.0f32; HIDDEN];
        let mut c = vec![0.0f32; HIDDEN];
        let mut z = vec![0.0f32; 4 * HIDDEN];
        for step in 0..t {
            let ti = if dir == 0 { step } else { t - 1 - step };
            for g in 0..4 * HIDDEN {
                let mut acc: i32 = 0;
                for i in 0..input {
                    acc += (xq[ti * input + i] as i32 - xz) * (wqd[i * 4 * HIDDEN + g] as i32);
                }
                z[g] = (acc as f32) * (xs * wscale) + wb[g];
            }
            // DynamicQuantizeLSTM quantizes the hidden state as well, per step and
            // per direction, with a dynamic scale over its 128 values; without it
            // the recurrence drifts by ~1e-3 and the logits by ~5e-2.
            let (hq, hs, hz) = dql(&h);
            for g in 0..4 * HIDDEN {
                let mut acc: i32 = 0;
                for i in 0..HIDDEN {
                    acc += (hq[i] as i32 - hz) * (rqd[i * 4 * HIDDEN + g] as i32);
                }
                z[g] += (acc as f32) * (hs * rscale) + rb[g];
            }
            for i in 0..HIDDEN {
                let ii = sigmoid(z[i]);
                let oo = sigmoid(z[HIDDEN + i]);
                let ff = sigmoid(z[2 * HIDDEN + i]);
                let gg = z[3 * HIDDEN + i].tanh();
                c[i] = ff * c[i] + ii * gg;
                h[i] = oo * c[i].tanh();
            }
            out[ti * 2 * HIDDEN + dir * HIDDEN..ti * 2 * HIDDEN + dir * HIDDEN + HIDDEN]
                .copy_from_slice(&h);
        }
    }
    out
}

/// Per-call user-dictionary overlay: sorted 64-bit word hashes per codepoint
/// length, laid over the immutable embedded tables. Hints only (the model
/// still decides the cuts), built and dropped with one call; never global.
#[derive(Default)]
struct LexOverlay {
    tables: Vec<(usize, Vec<u64>)>,
}

/// Longest word the dictionary can hold, in characters (`_runtime.Lexicon`).
const MAX_WORD_LEN: usize = 16;

impl LexOverlay {
    /// Strips each word and silently drops what upstream drops: shorter than
    /// 2 or longer than 16 characters, or containing whitespace.
    fn new(words: &[String]) -> Self {
        let mut by_len: HashMap<usize, Vec<u64>> = HashMap::new();
        for word in words {
            let word = word.trim();
            let n = word.chars().count();
            if !(2..=MAX_WORD_LEN).contains(&n) || word.chars().any(char::is_whitespace) {
                continue;
            }
            let mut h = 0u64;
            for (i, c) in word.chars().enumerate() {
                h = if i == 0 {
                    c as u64
                } else {
                    h.wrapping_mul(HASH_MUL).wrapping_add(c as u64)
                };
            }
            by_len.entry(n).or_default().push(h);
        }
        let mut tables: Vec<(usize, Vec<u64>)> = by_len
            .into_iter()
            .map(|(n, mut hashes)| {
                hashes.sort_unstable();
                hashes.dedup();
                (n, hashes)
            })
            .collect();
        tables.sort_unstable_by_key(|(n, _)| *n);
        Self { tables }
    }
}

/// Shared empty overlay for calls without user words.
static NO_USER_WORDS: LexOverlay = LexOverlay { tables: Vec::new() };

/// Marks the starts/ends/inside bits of one lex table's hits in `codes`.
fn mark_table(codes: &[u32], n: usize, table: &[u64], bits: &mut [i64], cover: &mut [i64]) {
    let w = codes.len();
    for i in 0..=w - n {
        let mut h = codes[i] as u64;
        for j in 1..n {
            h = h.wrapping_mul(HASH_MUL).wrapping_add(codes[i + j] as u64);
        }
        if table.binary_search(&h).is_ok() {
            let bucket = len_bucket(n);
            bits[i] |= 1i64 << bucket;
            bits[i + n - 1] |= 1i64 << (6 + bucket);
            if n > 2 {
                cover[i + 1] += 1;
                cover[i + n - 1] -= 1;
            }
        }
    }
}

/// 64-bit rolling-hash lexicon bits for the codepoints of one line, embedded
/// tables plus the call's user-word overlay.
fn lex_bits(m: &Model, codes: &[u32], overlay: &LexOverlay) -> Vec<i64> {
    let w = codes.len();
    let mut bits = vec![0i64; w];
    let mut cover = vec![0i64; w + 1];
    for (n, table) in &m.lex {
        if *n > w { break; }
        mark_table(codes, *n, table, &mut bits, &mut cover);
    }
    for (n, table) in &overlay.tables {
        if *n > w { break; }
        mark_table(codes, *n, table, &mut bits, &mut cover);
    }
    let mut cur = 0i64;
    for i in 0..w {
        cur += cover[i];
        if cur > 0 { bits[i] |= 1i64 << 12; }
    }
    bits
}

fn len_bucket(n: usize) -> u32 {
    match n {
        3 => 1,
        4 => 2,
        5..=6 => 3,
        7..=9 => 4,
        10.. => 5,
        _ => 0,
    }
}

fn no_start(c: u32) -> bool {
    matches!(c, 0x0E30..=0x0E3A | 0x0E45 | 0x0E47..=0x0E4E)
}
fn leading(c: u32) -> bool {
    (0x0E40..=0x0E44).contains(&c)
}
fn same_run(a: char, b: char) -> bool {
    let da = a.is_ascii_digit() || ('\u{0E50}'..='\u{0E59}').contains(&a);
    let db = b.is_ascii_digit() || ('\u{0E50}'..='\u{0E59}').contains(&b);
    if da && db { return true; }
    a.is_ascii_alphabetic() && b.is_ascii_alphabetic()
}

/// One whitespace-free line through the network: for every character, the
/// logits of all three units (coarse to fine: `long`, `short`, `minimal`).
fn forward(m: &Model, codes: &[u32], overlay: &LexOverlay) -> Vec<[f32; 3]> {
    let w = codes.len();
    if w == 0 { return Vec::new(); }

    let emb = f32v!(m, "emb");
    let lex_table = f32v!(m, "lex_table");
    let bits = lex_bits(m, codes, overlay);
    let mut x = vec![0.0f32; w * 96];
    for (i, &code) in codes.iter().enumerate() {
        let cls = m.chars.get(code as usize).copied().unwrap_or(NOISE_CLASS) as usize;
        let base = cls * 64;
        for c in 0..64 {
            x[c * w + i] = emb[base + c];
        }
        let bi = bits[i] as usize;
        for c in 0..32 {
            x[(64 + c) * w + i] = lex_table[bi * 32 + c];
        }
    }
    let mut convs = Vec::with_capacity(w * 384);
    for (layer, k, p) in [(0, 3, 1), (1, 5, 2), (2, 7, 3)] {
        let cw = f32v!(m, &format!("conv{layer}_w"));
        let cb = f32v!(m, &format!("conv{layer}_b"));
        convs.extend(conv1d(&x, cw, cb, 96, w, 128, (k, p)));
    }
    // conv1d returns [out,time]; the graph concatenates along channels then
    // transposes to [time, 384].
    let mut seq0 = vec![0.0f32; w * 384];
    for t in 0..w {
        for c in 0..384 {
            seq0[t * 384 + c] = convs[c * w + t];
        }
    }
    let s0 = lstm(m, &seq0, w, 384, 0);
    let s1 = lstm(m, &s0, w, 256, 1);

    let (q, scale, zp) = dql(&s1);
    let ow = i8v!(m, "out_w_quantized");
    let ozp = i8v!(m, "out_w_zero_point")[0] as i32;
    let out_b = f32v!(m, "out_b");
    let wscale = scalarf!(m, "out_w_scale");
    let mut logits = vec![[0.0f32; 3]; w];
    for (t, slot) in logits.iter_mut().enumerate() {
        for (g, l) in slot.iter_mut().enumerate() {
            let mut acc: i32 = 0;
            for i in 0..256 {
                acc += (q[t * 256 + i] as i32 - zp) * (ow[i * 3 + g] as i32 - ozp);
            }
            *l = (acc as f32) * (scale * wscale) + out_b[g];
        }
    }
    logits
}

/// The logit one unit ranks a position by: `minimal` follows its own
/// convention, the nested units take the max over their coarser units.
fn unit_logit(l: &[f32; 3], unit_idx: usize) -> f32 {
    if UNITS[unit_idx] == UNNESTED_MINIMAL {
        l[unit_idx]
    } else {
        l[..=unit_idx].iter().copied().fold(f32::MIN, f32::max)
    }
}

/// Word starts of one unit, from the logits `forward` returned for the same
/// line; Thai orthography rules some starts out.
fn unit_starts(codes: &[u32], logits: &[[f32; 3]], unit_idx: usize) -> Vec<bool> {
    let mut starts = vec![true; codes.len()];
    for (t, start) in starts.iter_mut().enumerate() {
        let sel = unit_logit(&logits[t], unit_idx);
        let allowed = !no_start(codes[t]) && (t == 0 || !leading(codes[t - 1]));
        *start = t == 0 || (sel >= 0.0 && allowed);
    }
    starts
}

/// Runs one whitespace-free line through the network, returning word starts.
fn line_starts(m: &Model, line: &str, unit_idx: usize, overlay: &LexOverlay) -> Vec<bool> {
    let codes: Vec<u32> = line.chars().map(|c| c as u32).collect();
    let logits = forward(m, &codes, overlay);
    unit_starts(&codes, &logits, unit_idx)
}

/// Wire entry: `latte_segment(text, cutoff)` where `cutoff` is the unit
/// (`long`/`short`/`minimal`; unknown defaults to `short`).
pub fn latte_segment(_sess: &mut Session, text: &str, unit: &str) -> Result<String, ProcessError> {
    let m = &*MODEL;
    let unit_idx = UNITS.iter().position(|u| *u == unit).unwrap_or(SHORT_UNIT);
    let mut all_tokens: Vec<String> = Vec::new();
    for (li, raw_line) in text.split('\n').enumerate() {
        if li > 0 { all_tokens.push("\n".to_string()); }
        // Non-whitespace characters form the model input; their byte offsets
        // map word starts back onto the original line.
        let mut codes: Vec<u32> = Vec::new();
        let mut idx: Vec<usize> = Vec::new();
        for (j, ch) in raw_line.char_indices() {
            if !ch.is_whitespace() {
                codes.push(ch as u32);
                idx.push(j);
            }
        }
        if codes.is_empty() {
            all_tokens.push(raw_line.to_string());
            continue;
        }
        let starts = line_starts(m, &codes_to_str(&codes), unit_idx, &NO_USER_WORDS);
        let mut cuts: Vec<usize> = Vec::new();
        for (i, &s) in starts.iter().enumerate() {
            if i > 0 && s { cuts.push(idx[i]); }
        }
        let chars: Vec<(usize, char)> = raw_line.char_indices().collect();
        for &(j, ch) in &chars {
            if ch.is_whitespace() {
                cuts.push(j);
                cuts.push(j + ch.len_utf8());
            }
        }
        cuts.sort_unstable();
        cuts.dedup();
        let chars: Vec<char> = raw_line.chars().collect();
        let byte_at: Vec<usize> = raw_line.char_indices().map(|(j, _)| j).collect();
        let mut bounds = vec![0usize];
        for cut in cuts {
            if cut > 0 && cut < raw_line.len() {
                let ci = byte_at.binary_search(&cut).ok();
                if let Some(ci) = ci
                    && !(same_run(chars[ci - 1], chars[ci]))
                {
                    bounds.push(cut);
                }
            }
        }
        bounds.push(raw_line.len());
        bounds.dedup();
        for pair in bounds.windows(2) {
            let tok = &raw_line[pair[0]..pair[1]];
            if !tok.is_empty() {
                all_tokens.push(tok.to_string());
            }
        }
    }
    serde_json::to_string(&all_tokens).map_err(|e| ProcessError(e.to_string()))
}

fn codes_to_str(codes: &[u32]) -> String {
    codes.iter().filter_map(|&c| char::from_u32(c)).collect()
}

// ---------------------------------------------------------------- n-best readings

const MAX_NBEST_CHARS: usize = 10_000;
const MAX_NBEST: usize = 100;

#[derive(serde::Serialize)]
struct Reading {
    tokens: Vec<String>,
    probability: f64,
}

/// `ln(1 + e^x)` in f32, stable for large `|x|`: numpy's `logaddexp(0.0, x)`.
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Parses the n-best `cutoff`: `<unit>` or `<unit>:<n>`, where `""` and
/// `"medium"` mean unit `short` and `n = 10`.
fn nbest_options(cutoff: &str) -> Result<(usize, usize), ProcessError> {
    if cutoff.is_empty() || cutoff == "medium" {
        return Ok((SHORT_UNIT, 10));
    }
    let (unit, n) = match cutoff.split_once(':') {
        Some((unit, n)) => (unit, Some(n)),
        None => (cutoff, None),
    };
    let unit_idx = UNITS
        .iter()
        .position(|u| *u == unit)
        .ok_or_else(|| ProcessError("invalid unit".to_string()))?;
    let n = match n {
        // `<n>` is a base-10 integer: ASCII digits only. Rust's `usize` parser
        // alone would also accept a leading `+`.
        Some(s) if !s.bytes().all(|b| b.is_ascii_digit()) => {
            return Err(ProcessError("n must be between 1 and 100".to_string()));
        }
        Some(s) => s
            .parse::<usize>()
            .map_err(|_| ProcessError("n must be between 1 and 100".to_string()))?,
        None => 10,
    };
    if !(1..=MAX_NBEST).contains(&n) {
        return Err(ProcessError("n must be between 1 and 100".to_string()));
    }
    Ok((unit_idx, n))
}

/// `(start, end)` of every dictionary word in a whitespace-free line.
fn lattice(m: &Model, codes: &[u32]) -> HashSet<(usize, usize)> {
    let mut spans = HashSet::new();
    for (len, table) in &m.lex {
        let len = *len;
        if len > codes.len() {
            break;
        }
        for i in 0..=codes.len() - len {
            let mut h = codes[i] as u64;
            for j in 1..len {
                h = h.wrapping_mul(HASH_MUL).wrapping_add(codes[i + j] as u64);
            }
            if table.binary_search(&h).is_ok() {
                spans.insert((i, i + len));
            }
        }
    }
    spans
}

/// Token spans (codepoint indices) of `text` for one unit, mirroring upstream
/// `tokenize(text, keep_whitespace=True, unit)`: one model call per `\n` line,
/// every whitespace character is a token of its own and cuts inside a run of
/// ASCII letters or decimal digits are dropped.
fn tokenize_spans(
    m: &Model,
    text: &str,
    unit_idx: usize,
    overlay: &LexOverlay,
) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut cut = vec![false; n + 1];
    let mut start = 0usize;
    for line in text.split('\n') {
        let mut idx: Vec<usize> = Vec::new();
        for (j, ch) in line.chars().enumerate() {
            if !ch.is_whitespace() {
                idx.push(start + j);
            }
        }
        if !idx.is_empty() {
            let codes: Vec<u32> = idx.iter().map(|&j| chars[j] as u32).collect();
            let logits = forward(m, &codes, overlay);
            let starts = unit_starts(&codes, &logits, unit_idx);
            for (s, &word_start) in starts.iter().enumerate().skip(1) {
                if word_start {
                    cut[idx[s]] = true;
                }
            }
        }
        start += line.chars().count() + 1;
    }
    for j in 0..n {
        if chars[j].is_whitespace() {
            cut[j] = true;
            cut[j + 1] = true;
        }
    }
    let mut bounds = Vec::with_capacity(n + 2);
    bounds.push(0usize);
    for j in 1..n {
        if cut[j] && !same_run(chars[j - 1], chars[j]) {
            bounds.push(j);
        }
    }
    bounds.push(n);
    bounds
        .windows(2)
        .map(|w| (w[0], w[1]))
        .filter(|(a, b)| b > a)
        .collect()
}

/// The word ends of one n-best reading, first to last; readings are linked
/// nodes (`(end, previous)`) shared through `pool`, so memory grows with
/// `size x n`, not `size^2 x n`.
fn unroll(mut node: Option<u32>, pool: &[(u32, Option<u32>)]) -> Vec<usize> {
    let mut ends = Vec::new();
    while let Some(i) = node {
        let (end, prev) = pool[i as usize];
        ends.push(end as usize);
        node = prev;
    }
    ends.reverse();
    ends
}

/// Wire entry: `latte_nbest(text, cutoff)` — the `n` most probable readings of
/// `text`, best first, as `[{"tokens":["..."],"probability":<number>}, ...]`.
///
/// `cutoff` is `<unit>` or `<unit>:<n>` (`long`/`short`/`minimal`, `n` in
/// 1..=100); `""` and `"medium"` mean unit `short` and `n = 10`. The whole text
/// is one whitespace-free line (no windowing); candidates are the lexicon
/// words plus the word spans of all three units, and whitespace is a hard
/// boundary in every reading. This mirrors the wheel's
/// `_runtime.py::tokenize_nbest`; goldens come from the wheel, never from here.
pub fn latte_nbest(text: &str, cutoff: &str) -> Result<String, ProcessError> {
    let (unit_idx, n) = nbest_options(cutoff)?;
    let chars: Vec<char> = text.chars().collect();
    let keep: Vec<usize> = chars
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_whitespace())
        .map(|(j, _)| j)
        .collect();
    if keep.is_empty() {
        // upstream returns the characters themselves as one sure reading
        let reading = Reading {
            tokens: chars.iter().map(|c| c.to_string()).collect(),
            probability: 1.0,
        };
        return serde_json::to_string(&[reading]).map_err(|e| ProcessError(e.to_string()));
    }
    if keep.len() > MAX_NBEST_CHARS {
        return Err(ProcessError(
            "text too long (max 10000 non-whitespace characters)".to_string(),
        ));
    }

    let m = &*MODEL;
    let line: Vec<char> = keep.iter().map(|&j| chars[j]).collect();
    let codes: Vec<u32> = line.iter().map(|&c| c as u32).collect();
    let size = codes.len();
    let logits = forward(m, &codes, &NO_USER_WORDS);
    let sel: Vec<f32> = logits.iter().map(|l| unit_logit(l, unit_idx)).collect();
    // whitespace between two characters: always a cut, never inside a piece
    let mut gap = vec![false; size];
    for i in 1..size {
        gap[i] = keep[i] - keep[i - 1] > 1;
    }
    let mut log_start = vec![f32::NEG_INFINITY; size];
    let mut log_inside = vec![0.0f32; size];
    for i in 0..size {
        let free = !no_start(codes[i])
            && (i == 0 || (!leading(codes[i - 1]) && !same_run(line[i - 1], line[i])));
        if free {
            log_start[i] = -softplus(-sel[i]);
            log_inside[i] = -softplus(sel[i]);
        }
        if gap[i] {
            log_start[i] = 0.0;
            log_inside[i] = 0.0;
        }
    }
    let mut inside = vec![0.0f32; size + 1];
    for i in 0..size {
        inside[i + 1] = inside[i] + log_inside[i];
    }
    let mut gaps = vec![0i64; size + 1];
    for i in 0..size {
        gaps[i + 1] = gaps[i] + i64::from(gap[i]);
    }

    let mut pieces = lattice(m, &codes);
    // text codepoint -> index in the whitespace-free line
    let mut where_: Vec<usize> = vec![0; chars.len() + 1];
    for (j, ch) in chars.iter().enumerate() {
        where_[j + 1] = where_[j] + usize::from(!ch.is_whitespace());
    }
    let mut best: Option<Vec<usize>> = None;
    for v in 0..UNITS.len() {
        let mut bounds = vec![0usize];
        for (a, b) in tokenize_spans(m, text, v, &NO_USER_WORDS) {
            if chars[a..b].iter().all(|c| c.is_whitespace()) {
                continue;
            }
            pieces.insert((where_[a], where_[b]));
            bounds.push(where_[b]);
        }
        if v == unit_idx {
            best = Some(bounds);
        }
    }
    let mut ending: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(a, b) in &pieces {
        ending.entry(b).or_default().push(a);
    }
    for ends in ending.values_mut() {
        ends.sort_unstable();
    }

    // paths[b]: the <= n best scores to end a word at b, each with its node
    let mut pool: Vec<(u32, Option<u32>)> = vec![(0, None)];
    let mut paths: Vec<Vec<(f32, u32)>> = vec![Vec::new(); size + 1];
    paths[0].push((0.0, 0));
    for b in 1..=size {
        let mut found: Vec<(f32, u32)> = Vec::new();
        if let Some(ends) = ending.get(&b) {
            for &a in ends {
                if gaps[b] > gaps[a + 1] {
                    continue;
                }
                let score =
                    (if a == 0 { 0.0 } else { log_start[a] }) + inside[b] - inside[a + 1];
                if !score.is_finite() {
                    continue;
                }
                for &(s, node) in &paths[a] {
                    pool.push((b as u32, Some(node)));
                    found.push((s + score, (pool.len() - 1) as u32));
                }
            }
        }
        if !found.is_empty() {
            found.sort_by(|x, y| y.0.total_cmp(&x.0));
            found.truncate(n);
            paths[b] = found;
        }
    }
    let mut ranked: Vec<(f32, Vec<usize>)> = paths[size]
        .iter()
        .map(|&(s, node)| (s, unroll(Some(node), &pool)))
        .collect();
    if let Some(best) = best.as_ref() {
        // the plain `latte_segment` reading first, also on exact score ties
        ranked.sort_by(|x, y| {
            (x.1 != *best)
                .cmp(&(y.1 != *best))
                .then_with(|| y.0.total_cmp(&x.0))
        });
    }

    let space: Vec<usize> = (0..chars.len())
        .filter(|&j| chars[j].is_whitespace())
        .collect();
    let mut out: Vec<Reading> = Vec::with_capacity(ranked.len());
    for (score, path) in ranked.iter().take(n) {
        let mut cut: Vec<usize> = path[1..path.len().saturating_sub(1)]
            .iter()
            .map(|&p| keep[p])
            .collect();
        for &j in &space {
            cut.push(j);
            cut.push(j + 1);
        }
        cut.retain(|&j| j > 0 && j < chars.len());
        cut.sort_unstable();
        cut.dedup();
        let mut marks = Vec::with_capacity(cut.len() + 2);
        marks.push(0usize);
        marks.extend(cut);
        marks.push(chars.len());
        let tokens = marks
            .windows(2)
            .filter(|w| w[1] > w[0])
            .map(|w| chars[w[0]..w[1]].iter().collect())
            .collect();
        out.push(Reading {
            tokens,
            probability: score.exp() as f64,
        });
    }
    serde_json::to_string(&out).map_err(|e| ProcessError(e.to_string()))
}



// ---------------------------------------------------------------- segment opts

const MAX_TEXTS: usize = 100;
const MAX_USER_WORDS: usize = 1000;

fn invalid_options() -> ProcessError {
    ProcessError("invalid options".to_string())
}

/// One `latte_segment_opts` envelope. The custom visitor below rejects unknown
/// and duplicate keys and enforces the list caps while reading, so the
/// untrusted JSON cannot make us materialise more entries than the wire allows.
#[derive(Default)]
struct SegmentOptions {
    text: Option<String>,
    texts: Option<Vec<String>>,
    unit: Option<String>,
    keep_whitespace: Option<bool>,
    user_words: Option<Vec<String>>,
}

/// A JSON array of strings that errors as soon as it would exceed `MAX` entries.
struct CappedStrings<const MAX: usize>(Vec<String>);

impl<'de, const MAX: usize> serde::Deserialize<'de> for CappedStrings<MAX> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V<const MAX: usize>;
        impl<'de, const MAX: usize> serde::de::Visitor<'de> for V<MAX> {
            type Value = CappedStrings<MAX>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("an array of strings")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                // honour a size hint only up to the cap: never pre-allocate the
                // attacker's claimed length
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(MAX));
                while let Some(s) = seq.next_element::<String>()? {
                    if out.len() == MAX {
                        return Err(serde::de::Error::custom("too many entries"));
                    }
                    out.push(s);
                }
                Ok(CappedStrings(out))
            }
        }
        deserializer.deserialize_seq(V)
    }
}

impl<'de> serde::Deserialize<'de> for SegmentOptions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = SegmentOptions;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let dup = || serde::de::Error::custom("duplicate key");
                let mut out = SegmentOptions::default();
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "text" => {
                            if out.text.is_some() {
                                return Err(dup());
                            }
                            out.text = Some(map.next_value()?);
                        }
                        "texts" => {
                            if out.texts.is_some() {
                                return Err(dup());
                            }
                            out.texts =
                                Some(map.next_value::<CappedStrings<MAX_TEXTS>>()?.0);
                        }
                        "unit" => {
                            if out.unit.is_some() {
                                return Err(dup());
                            }
                            out.unit = Some(map.next_value()?);
                        }
                        "keep_whitespace" => {
                            if out.keep_whitespace.is_some() {
                                return Err(dup());
                            }
                            out.keep_whitespace = Some(map.next_value()?);
                        }
                        "user_words" => {
                            if out.user_words.is_some() {
                                return Err(dup());
                            }
                            out.user_words = Some(
                                map.next_value::<CappedStrings<MAX_USER_WORDS>>()?.0,
                            );
                        }
                        _ => return Err(serde::de::Error::custom("unknown key")),
                    }
                }
                Ok(out)
            }
        }
        deserializer.deserialize_map(V)
    }
}

/// Parses and validates an options envelope. Every check runs before the model
/// is touched; syntax maps to `malformed JSON`, schema/type/caps to
/// `invalid options`, then `invalid unit`, then the per-text length cap.
fn parse_segment_options(
    envelope: &str,
) -> Result<(Vec<String>, usize, bool, LexOverlay), ProcessError> {
    let opts: SegmentOptions = match serde_json::from_str(envelope) {
        Ok(opts) => opts,
        Err(e) => {
            return Err(match e.classify() {
                serde_json::error::Category::Syntax | serde_json::error::Category::Eof => {
                    ProcessError("malformed JSON".to_string())
                }
                _ => invalid_options(),
            });
        }
    };
    let texts = match (opts.text, opts.texts) {
        (Some(text), None) => vec![text],
        (None, Some(texts)) => texts,
        _ => return Err(invalid_options()),
    };
    let unit_idx = match opts.unit.as_deref() {
        None => SHORT_UNIT,
        Some(unit) => UNITS
            .iter()
            .position(|u| *u == unit)
            .ok_or_else(|| ProcessError("invalid unit".to_string()))?,
    };
    for text in &texts {
        let non_ws = text.chars().filter(|c| !c.is_whitespace()).count();
        if non_ws > MAX_NBEST_CHARS {
            return Err(ProcessError(
                "text too long (max 10000 non-whitespace characters)".to_string(),
            ));
        }
    }
    let user_words = opts.user_words.as_deref().unwrap_or(&[]);
    Ok((
        texts,
        unit_idx,
        opts.keep_whitespace.unwrap_or(true),
        LexOverlay::new(user_words),
    ))
}

/// Wire entry: `latte_segment_opts(text, cutoff)` — `text` is a compact JSON
/// envelope (`{"text":...}` or `{"texts":[...]}`, optional `unit`,
/// `keep_whitespace`, `user_words`), `cutoff` is ignored.
///
/// Returns a JSON array with one token array per input text, mirroring the
/// wheel's `tokenize_batch` (`{"text":""}` => `[[]]`, `{"texts":[]}` => `[]`).
/// Texts are processed independently (the accepted upstream batch departure);
/// user words are per-call lexicon hints overlaid on the embedded tables.
pub fn latte_segment_opts(text: &str) -> Result<String, ProcessError> {
    let (texts, unit_idx, keep_whitespace, overlay) = parse_segment_options(text)?;
    if texts.is_empty() {
        return Ok("[]".to_string());
    }
    let m = &*MODEL;
    let mut out: Vec<Vec<String>> = Vec::with_capacity(texts.len());
    for text in &texts {
        let chars: Vec<char> = text.chars().collect();
        let mut tokens: Vec<String> = tokenize_spans(m, text, unit_idx, &overlay)
            .into_iter()
            .map(|(a, b)| chars[a..b].iter().collect::<String>())
            .collect();
        if !keep_whitespace {
            // every whitespace character is its own token upstream; drop those
            tokens.retain(|tok| !tok.chars().all(char::is_whitespace));
        }
        out.push(tokens);
    }
    serde_json::to_string(&out).map_err(|e| ProcessError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(result: Result<String, ProcessError>) -> String {
        result.expect_err("expected an error").0
    }

    #[test]
    fn opts_shapes_and_units_match_the_wheel() {
        // a single `text` still produces one token array, and `short` is the default
        assert_eq!(
            latte_segment_opts(r#"{"text":"ตากลม"}"#).unwrap(),
            r#"[["ตา","กลม"]]"#
        );
        assert_eq!(
            latte_segment_opts(r#"{"text":"ตากลม","unit":"short"}"#).unwrap(),
            r#"[["ตา","กลม"]]"#
        );
        assert_eq!(
            latte_segment_opts(r#"{"text":"ตากลม","unit":"long"}"#).unwrap(),
            r#"[["ตากลม"]]"#
        );
        assert_eq!(
            latte_segment_opts(r#"{"text":"ตากลม","unit":"minimal"}"#).unwrap(),
            r#"[["ตากลม"]]"#
        );
        assert_eq!(latte_segment_opts(r#"{"text":""}"#).unwrap(), "[[]]");
        assert_eq!(latte_segment_opts(r#"{"texts":[]}"#).unwrap(), "[]");
    }

    #[test]
    fn opts_whitespace_toggle_and_multiline() {
        let text = "สวัสดีครับ วันนี้อากาศดีมาก";
        let kept = format!(r#"{{"text":"{text}"}}"#);
        let dropped = format!(r#"{{"text":"{text}","keep_whitespace":false}}"#);
        assert_eq!(
            latte_segment_opts(&kept).unwrap(),
            r#"[["สวัสดี","ครับ"," ","วัน","นี้","อากาศ","ดี","มาก"]]"#
        );
        assert_eq!(
            latte_segment_opts(&dropped).unwrap(),
            r#"[["สวัสดี","ครับ","วัน","นี้","อากาศ","ดี","มาก"]]"#
        );
        // one model call per line, newline kept as its own token by default
        assert_eq!(
            latte_segment_opts(r#"{"text":"ตา\nกลม"}"#).unwrap(),
            r#"[["ตา","\n","กลม"]]"#
        );
        assert_eq!(
            latte_segment_opts(r#"{"text":"ตา\nกลม","keep_whitespace":false}"#).unwrap(),
            r#"[["ตา","กลม"]]"#
        );
    }

    #[test]
    fn opts_texts_are_independent() {
        assert_eq!(
            latte_segment_opts(r#"{"texts":["ฉันกินข้าวที่บ้าน","แมวกินปลา"]}"#).unwrap(),
            r#"[["ฉัน","กิน","ข้าว","ที่","บ้าน"],["แมว","กิน","ปลา"]]"#
        );
        assert_eq!(
            latte_segment_opts(
                r#"{"texts":["ตา กลม","คุณปุณยวีร์ไปกรุงเทพ"],"keep_whitespace":false}"#
            )
            .unwrap(),
            r#"[["ตา","กลม"],["คุณ","ปุณยวีร์","ไป","กรุงเทพ"]]"#
        );
        // each text alone equals its slot in the batch
        let alone = latte_segment_opts(r#"{"text":"ฉันกินข้าวที่บ้าน"}"#).unwrap();
        assert_eq!(alone, r#"[["ฉัน","กิน","ข้าว","ที่","บ้าน"]]"#);
    }

    #[test]
    fn opts_user_words_are_hints() {
        // the control split, then the same text with a user word that joins it
        assert_eq!(
            latte_segment_opts(r#"{"text":"แมวหมา"}"#).unwrap(),
            r#"[["แมว","หมา"]]"#
        );
        assert_eq!(
            latte_segment_opts(r#"{"text":"แมวหมา","user_words":["แมวหมา"]}"#).unwrap(),
            r#"[["แมวหมา"]]"#
        );
        // words are stripped like upstream `str.strip()`
        assert_eq!(
            latte_segment_opts(r#"{"text":"แมวหมา","user_words":[" แมวหมา "]}"#).unwrap(),
            r#"[["แมวหมา"]]"#
        );
        // too short, whitespace-bearing and too long words are silently dropped
        assert_eq!(
            latte_segment_opts(
                r#"{"text":"แมวหมา","user_words":["ก","ตา กลม","12345678901234567"]}"#
            )
            .unwrap(),
            r#"[["แมว","หมา"]]"#
        );
    }

    #[test]
    fn opts_errors_are_contractual() {
        // malformed JSON (including the empty envelope)
        assert_eq!(err(latte_segment_opts("")), "malformed JSON");
        assert_eq!(err(latte_segment_opts("{")), "malformed JSON");
        assert_eq!(err(latte_segment_opts("not json")), "malformed JSON");
        // schema violations
        assert_eq!(err(latte_segment_opts("{}")), "invalid options");
        assert_eq!(
            err(latte_segment_opts(r#"{"text":"a","texts":["b"]}"#)),
            "invalid options"
        );
        assert_eq!(
            err(latte_segment_opts(r#"{"text":"a","extra":1}"#)),
            "invalid options"
        );
        assert_eq!(err(latte_segment_opts(r#"{"text":5}"#)), "invalid options");
        assert_eq!(err(latte_segment_opts(r#"{"texts":"x"}"#)), "invalid options");
        assert_eq!(
            err(latte_segment_opts(r#"{"texts":["x",1]}"#)),
            "invalid options"
        );
        assert_eq!(
            err(latte_segment_opts(r#"{"text":"a","keep_whitespace":"yes"}"#)),
            "invalid options"
        );
        assert_eq!(
            err(latte_segment_opts(r#"{"text":"a","user_words":[1]}"#)),
            "invalid options"
        );
        assert_eq!(
            err(latte_segment_opts(r#"{"text":"a","text":"b"}"#)),
            "invalid options"
        );
        // bad unit is the schema-valid value with an unknown name
        assert_eq!(
            err(latte_segment_opts(r#"{"text":"a","unit":"bogus"}"#)),
            "invalid unit"
        );
        assert_eq!(
            err(latte_segment_opts(r#"{"text":"a","unit":5}"#)),
            "invalid options"
        );
        // caps are validated before the model is touched
        let texts = serde_json::json!({"texts": vec!["x"; 101]}).to_string();
        assert_eq!(err(latte_segment_opts(&texts)), "invalid options");
        let words = serde_json::json!({"text": "a", "user_words": vec!["ab"; 1001]}).to_string();
        assert_eq!(err(latte_segment_opts(&words)), "invalid options");
        let long = serde_json::json!({"text": "ก".repeat(10_001)}).to_string();
        assert_eq!(
            err(latte_segment_opts(&long)),
            "text too long (max 10000 non-whitespace characters)"
        );
        // the cap counts non-whitespace characters, and bad unit wins over length
        let spaces = serde_json::json!({"text": " ".repeat(10_001)}).to_string();
        assert!(latte_segment_opts(&spaces).is_ok());
        let spaced = serde_json::json!({"text": "ก ".repeat(10_001)}).to_string();
        assert_eq!(
            err(latte_segment_opts(&spaced)),
            "text too long (max 10000 non-whitespace characters)"
        );
        let both = serde_json::json!({"text": "ก".repeat(10_001), "unit": "bogus"}).to_string();
        assert_eq!(err(latte_segment_opts(&both)), "invalid unit");
    }

    #[test]
    fn nbest_cutoff_errors_are_contractual() {
        assert_eq!(err(latte_nbest("ตากลม", "bogus")), "invalid unit");
        assert_eq!(err(latte_nbest("ตากลม", "medium:5")), "invalid unit");
        assert_eq!(
            err(latte_nbest("ตากลม", "short:x")),
            "n must be between 1 and 100"
        );
        assert_eq!(err(latte_nbest("ตากลม", "short:0")), "n must be between 1 and 100");
        assert_eq!(err(latte_nbest("ตากลม", "short:101")), "n must be between 1 and 100");
        assert_eq!(
            err(latte_nbest("ตากลม", "short:3:4")),
            "n must be between 1 and 100"
        );
        // `<n>` is a base-10 integer: a sign, a non-ASCII digit or an empty
        // value is not one (Rust's usize parser alone would take `+5`)
        assert_eq!(
            err(latte_nbest("ตากลม", "short:+5")),
            "n must be between 1 and 100"
        );
        assert_eq!(
            err(latte_nbest("ตากลม", "short:٥")),
            "n must be between 1 and 100"
        );
        assert_eq!(
            err(latte_nbest("ตากลม", "short:")),
            "n must be between 1 and 100"
        );
        assert_eq!(
            err(latte_nbest(&"ก".repeat(10_001), "short")),
            "text too long (max 10000 non-whitespace characters)"
        );
        // the cap counts non-whitespace characters, not bytes or chars
        assert!(latte_nbest(&format!("ก{}", " ".repeat(10_001)), "short:1").is_ok());
    }

    #[test]
    fn nbest_empty_and_whitespace_text() {
        assert_eq!(
            latte_nbest("", "short:3").unwrap(),
            r#"[{"tokens":[],"probability":1.0}]"#
        );
        assert_eq!(
            latte_nbest("  ", "short:3").unwrap(),
            r#"[{"tokens":[" "," "],"probability":1.0}]"#
        );
    }

    #[test]
    fn nbest_default_cutoff_is_short_ten() {
        assert_eq!(
            latte_nbest("ตากลม", "medium").unwrap(),
            latte_nbest("ตากลม", "short:10").unwrap()
        );
        assert_eq!(
            latte_nbest("ตากลม", "").unwrap(),
            latte_nbest("ตากลม", "short").unwrap()
        );
        // leading zeros are base-10 digits, so they name the same n
        assert_eq!(
            latte_nbest("ตากลม", "short:05").unwrap(),
            latte_nbest("ตากลม", "short:5").unwrap()
        );
    }

    #[test]
    fn nbest_short_three_readings() {
        let out = latte_nbest("ตากลม", "short:3").unwrap();
        // compact JSON, `tokens` before `probability`, nothing escaped
        assert!(
            out.starts_with(r#"[{"tokens":["ตา","กลม"],"probability":"#),
            "unexpected wire shape: {out}"
        );
        let readings: serde_json::Value = serde_json::from_str(&out).unwrap();
        let readings = readings.as_array().unwrap();
        assert_eq!(readings.len(), 3);
        let tokens: Vec<serde_json::Value> =
            readings.iter().map(|r| r["tokens"].clone()).collect();
        assert_eq!(
            tokens,
            [
                serde_json::json!(["ตา", "กลม"]),
                serde_json::json!(["ตากลม"]),
                serde_json::json!(["ตาก", "ลม"]),
            ]
        );
        for reading in readings {
            let obj = reading.as_object().unwrap();
            assert_eq!(obj.len(), 2, "each reading is exactly tokens + probability");
            assert!(obj.contains_key("tokens") && obj.contains_key("probability"));
        }
        let probs: Vec<f64> = readings
            .iter()
            .map(|r| r["probability"].as_f64().unwrap())
            .collect();
        assert!(
            probs.windows(2).all(|w| w[0] > w[1]),
            "readings must be ranked, got {probs:?}"
        );
        // the wheel's top probability; values are informational, so a bounded check
        assert!((probs[0] - 0.417_839_8).abs() < 1e-4, "got {}", probs[0]);
    }
}
