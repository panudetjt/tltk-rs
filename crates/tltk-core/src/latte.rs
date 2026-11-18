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

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::{ProcessError, Session};

const MAGIC_W: &[u8; 4] = b"LT01";
const MAGIC_L: &[u8; 4] = b"LX01";
const HASH_MUL: u64 = 0x9E37_79B9_7F4A_7C15;
const HIDDEN: usize = 128;
const UNITS: [&str; 3] = ["long", "short", "minimal"];
const UNNESTED_MINIMAL: &str = "minimal";
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

/// 64-bit rolling-hash lexicon bits for the codepoints of one line.
fn lex_bits(m: &Model, codes: &[u32]) -> Vec<i64> {
    let w = codes.len();
    let mut bits = vec![0i64; w];
    let mut cover = vec![0i64; w + 1];
    for (n, table) in &m.lex {
        let n = *n;
        if n > w { break; }
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

/// Runs one whitespace-free line through the network, returning word starts.
fn line_starts(m: &Model, line: &str, unit_idx: usize) -> Vec<bool> {
    let codes: Vec<u32> = line.chars().map(|c| c as u32).collect();
    let w = codes.len();
    if w == 0 { return Vec::new(); }

    let emb = f32v!(m, "emb");
    let lex_table = f32v!(m, "lex_table");
    let bits = lex_bits(m, &codes);
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
    let mut starts = vec![true; w];
    for t in 0..w {
        let mut logits = [0.0f32; 3];
        for g in 0..3 {
            let mut acc: i32 = 0;
            for i in 0..256 {
                acc += (q[t * 256 + i] as i32 - zp) * (ow[i * 3 + g] as i32 - ozp);
            }
            logits[g] = (acc as f32) * (scale * wscale) + out_b[g];
        }
        let sel = if UNITS[unit_idx] == UNNESTED_MINIMAL {
            logits[unit_idx]
        } else {
            logits[..=unit_idx].iter().copied().fold(f32::MIN, f32::max)
        };
        let code = codes[t];
        let allowed = !no_start(code) && (t == 0 || !leading(codes[t - 1]));
        starts[t] = t == 0 || (sel >= 0.0 && allowed);
    }
    starts
}

/// Wire entry: `latte_segment(text, cutoff)` where `cutoff` is the unit
/// (`long`/`short`/`minimal`; unknown defaults to `short`).
pub fn latte_segment(_sess: &mut Session, text: &str, unit: &str) -> Result<String, ProcessError> {
    let m = &*MODEL;
    let unit_idx = UNITS.iter().position(|u| *u == unit).unwrap_or(1);
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
        let starts = line_starts(m, &codes_to_str(&codes), unit_idx);
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

