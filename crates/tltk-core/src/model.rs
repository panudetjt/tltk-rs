//! Model file formats and loaders (see tools/convert/convert.py for writers).

use std::collections::HashMap;

pub struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Reader { b, pos: 0 }
    }
    fn byte(&mut self) -> u8 {
        let v = self.b[self.pos];
        self.pos += 1;
        v
    }
    pub fn uvarint(&mut self) -> usize {
        let mut n: usize = 0;
        let mut shift = 0;
        loop {
            let b = self.byte();
            n |= ((b & 0x7f) as usize) << shift;
            if b & 0x80 == 0 {
                return n;
            }
            shift += 7;
        }
    }
    pub fn str_(&mut self) -> &'a str {
        let len = self.uvarint();
        let s = std::str::from_utf8(&self.b[self.pos..self.pos + len]).unwrap();
        self.pos += len;
        s
    }
    pub fn f64_(&mut self) -> f64 {
        let v = f64::from_le_bytes(self.b[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        v
    }
    pub fn f32_(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.b[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    pub fn i32_(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.b[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    pub fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        s
    }
    pub fn at_end(&self) -> bool {
        self.pos >= self.b.len()
    }
}

fn try_gunzip(data: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut d = flate2::read::GzDecoder::new(data);
    let mut out = Vec::new();
    d.read_to_end(&mut out)
        .map_err(|e| format!("gunzip failed: {}", e))?;
    Ok(out)
}


fn check_magic(r: &mut Reader, magic: &[u8; 4]) -> Result<(), String> {
    if r.pos + 4 > r.b.len() {
        return Err(format!("unexpected EOF reading magic: want {:?}", magic));
    }
    let got = r.bytes(4);
    if got != magic {
        return Err(format!("bad model magic: got {:?} want {:?}", got, magic));
    }
    Ok(())
}


/// TD01 — thdict word set
pub struct Thdict {
    pub words: Vec<String>,
    set: std::collections::HashSet<String>,
}

impl Thdict {
    fn decode(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(data);
        check_magic(&mut r, b"TD01")?;
        let n = r.uvarint();
        let mut words = Vec::with_capacity(n);
        let mut set = std::collections::HashSet::with_capacity(n);
        for _ in 0..n {
            let w = r.str_().to_string();
            set.insert(w.clone());
            words.push(w);
        }
        Ok(Thdict { words, set })
    }
    pub fn contains(&self, w: &str) -> bool {
        self.set.contains(w)
    }
}
/// Minimal perfect hash over a set of distinct string keys using CHD (compress, hash and displace).
/// Maps each key -> its assigned u32 value (e.g. sorted index or block start).
/// A 32-bit guard residue ensures bogus keys are rejected with ~100% probability before
/// any string comparison.
#[derive(Clone, Debug, Default)]
pub struct PerfHash {
    num_keys: usize,
    num_buckets: usize,
    seeds: Vec<u32>,
    entries: Vec<PerfEntry>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PerfEntry {
    pub value: u32,
    pub guard: u32,
}

#[inline(always)]
fn perf_hash_str(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let mut hash = 0u64;
    const K: u64 = 0x517cc1b727220a95;
    let (chunks, rem) = bytes.as_chunks::<8>();
    for chunk in chunks {
        let v = u64::from_le_bytes(*chunk);
        hash = hash.rotate_left(5) ^ v;
        hash = hash.wrapping_mul(K);
    }
    if !rem.is_empty() {
        let mut buf = [0u8; 8];
        buf[..rem.len()].copy_from_slice(rem);
        let v = u64::from_le_bytes(buf);
        hash = hash.rotate_left(5) ^ v;
        hash = hash.wrapping_mul(K);
    }
    hash ^= hash >> 32;
    hash = hash.wrapping_mul(0xd6e8feb86659fd93);
    hash ^= hash >> 32;
    hash = hash.wrapping_mul(0xd6e8feb86659fd93);
    hash ^= hash >> 32;
    hash
}

#[inline(always)]
fn perf_fast_reduce(x: u64, n: usize) -> usize {
    ((x as u128 * n as u128) >> 64) as usize
}

#[inline(always)]
fn perf_get_slot(h: u64, d: u32, n: usize) -> usize {
    let x = h.wrapping_add((d as u64).wrapping_mul(0x9e3779b97f4a7c15));
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    perf_fast_reduce(z, n)
}

#[inline(always)]
fn perf_guard(h: u64, s: &str) -> u32 {
    ((h >> 32) as u32) ^ (h as u32) ^ (s.len() as u32)
}

impl PerfHash {
    pub fn build(items: &[(&str, u32)]) -> Self {
        let n = items.len();
        if n == 0 {
            return Self::default();
        }
        let m = (n * 13) / 10 + 1;

        let mut key_hashes: Vec<u64> = Vec::with_capacity(n);
        let mut bucket_sizes = vec![0u32; m];
        for &(k, _) in items {
            let h = perf_hash_str(k);
            key_hashes.push(h);
            let b = perf_fast_reduce(h, m);
            bucket_sizes[b] += 1;
        }

        let mut offsets = vec![0usize; m + 1];
        let mut cursor = 0;
        for b in 0..m {
            offsets[b] = cursor;
            cursor += bucket_sizes[b] as usize;
        }
        offsets[m] = cursor;

        let mut bucket_items = vec![(0u64, 0u32, 0u32); n];
        let mut write_pos = offsets.clone();
        for (idx, &(k, val)) in items.iter().enumerate() {
            let h = key_hashes[idx];
            let b = perf_fast_reduce(h, m);
            let pos = write_pos[b];
            write_pos[b] += 1;
            bucket_items[pos] = (h, val, k.len() as u32);
        }

        let mut non_empty: Vec<(usize, u32)> = bucket_sizes
            .iter()
            .enumerate()
            .filter(|&(_, &size)| size > 0)
            .map(|(b, &size)| (b, size))
            .collect();
        non_empty.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut occupied = vec![false; n];
        let mut seeds = vec![0u32; m];
        let mut entries = vec![PerfEntry::default(); n];
        let mut temp_slots = Vec::with_capacity(32);

        for &(b, _size) in &non_empty {
            let start = offsets[b];
            let end = offsets[b + 1];
            let b_items = &bucket_items[start..end];

            let mut d = 0u32;
            loop {
                temp_slots.clear();
                let mut collision = false;
                for &(h, _, _) in b_items {
                    let slot = perf_get_slot(h, d, n);
                    if occupied[slot] || temp_slots.contains(&slot) {
                        collision = true;
                        break;
                    }
                    temp_slots.push(slot);
                }

                if !collision {
                    seeds[b] = d;
                    for (idx, &(h, val, len)) in b_items.iter().enumerate() {
                        let slot = temp_slots[idx];
                        occupied[slot] = true;
                        let guard = ((h >> 32) as u32) ^ (h as u32) ^ len;
                        entries[slot] = PerfEntry { value: val, guard };
                    }
                    break;
                }

                d += 1;
            }
        }

        PerfHash {
            num_keys: n,
            num_buckets: m,
            seeds,
            entries,
        }
    }

    #[inline(always)]
    pub fn get(&self, s: &str) -> Option<u32> {
        if self.num_keys == 0 {
            return None;
        }
        let h = perf_hash_str(s);
        let b = perf_fast_reduce(h, self.num_buckets);
        let d = self.seeds[b];
        let slot = perf_get_slot(h, d, self.num_keys);
        let entry = &self.entries[slot];
        let guard = perf_guard(h, s);
        if entry.guard == guard {
            Some(entry.value)
        } else {
            None
        }
    }
}


/// SG01 — syllable trigram counts + derived maps (from sylseg.3g)
pub struct Sylseg {
    pub syls: Vec<String>,
    pub syl_hash: PerfHash,
    /// (x, y, z, count) sorted by (x,y,z) indices
    pub tri: Vec<[u32; 4]>,
    /// (x, y, count) sorted
    pub bi_count: Vec<[u32; 4]>,
    /// (x, y, type-count) sorted
    pub bi_type: Vec<[u32; 4]>,
    /// (x, count) sorted
    pub count: Vec<[u32; 2]>,
    /// (x, type-count) sorted
    pub type_: Vec<[u32; 2]>,
    pub total_word: usize,
    pub total_lex: usize,
}

impl Sylseg {
    fn decode(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(data);
        check_magic(&mut r, b"SG01")?;
        let n = r.uvarint();
        let mut syls = Vec::with_capacity(n);
        for _ in 0..n {
            syls.push(r.str_().to_string());
        }
        let rows4 = |r: &mut Reader| -> Vec<[u32; 4]> {
            let n = r.uvarint();
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push([r.uvarint() as u32, r.uvarint() as u32, r.uvarint() as u32, r.uvarint() as u32]);
            }
            v
        };
        let rows3 = |r: &mut Reader| -> Vec<[u32; 4]> {
            let n = r.uvarint();
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push([r.uvarint() as u32, r.uvarint() as u32, r.uvarint() as u32, 0]);
            }
            v
        };
        let tri = rows4(&mut r);
        let bi_count = rows3(&mut r);
        let bi_type = rows3(&mut r);
        let rows2 = |r: &mut Reader| -> Vec<[u32; 2]> {
            let n = r.uvarint();
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push([r.uvarint() as u32, r.uvarint() as u32]);
            }
            v
        };
        let count = rows2(&mut r);
        let type_ = rows2(&mut r);
        let total_word = r.uvarint();
        let total_lex = r.uvarint();
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        let items: Vec<(&str, u32)> = syls
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as u32))
            .collect();
        let syl_hash = PerfHash::build(&items);
        #[cfg(not(target_arch = "wasm32"))]
        let syl_build_ms = t0.elapsed().as_millis();
        #[cfg(target_arch = "wasm32")]
        let syl_build_ms = 0;
        if std::env::var("TLTK_DEBUG").is_ok() {
            eprintln!(
                "sylseg decoded: syls={} (hash in {}ms) tri={} bicount={} bitype={} count={} type={} TW={} TL={}",
                syls.len(), syl_build_ms, tri.len(), bi_count.len(), bi_type.len(), count.len(), type_.len(), total_word, total_lex
            );
        }
        Ok(Sylseg { syls, syl_hash, tri, bi_count, bi_type, count, type_, total_word, total_lex })
    }
    pub fn syl_id(&self, s: &str) -> Option<u32> {
        if let Some(idx) = self.syl_hash.get(s)
            && self.syls.get(idx as usize).map(|p| p.as_str()) == Some(s) {
                return Some(idx);
            }
        None
    }
    pub fn tri_get(&self, x: u32, y: u32, z: u32) -> u32 {
        self.tri
            .binary_search_by(|p| (p[0], p[1], p[2]).cmp(&(x, y, z)))
            .ok()
            .map(|i| self.tri[i][3])
            .unwrap_or(0)
    }
    pub fn bi_count_get(&self, x: u32, y: u32) -> u32 {
        self.bi_count
            .binary_search_by(|p| (p[0], p[1]).cmp(&(x, y)))
            .ok()
            .map(|i| self.bi_count[i][2])
            .unwrap_or(0)
    }
    pub fn bi_type_get(&self, x: u32, y: u32) -> u32 {
        self.bi_type
            .binary_search_by(|p| (p[0], p[1]).cmp(&(x, y)))
            .ok()
            .map(|i| self.bi_type[i][2])
            .unwrap_or(0)
    }
    pub fn count_get(&self, x: u32) -> u32 {
        self.count
            .binary_search_by(|p| p[0].cmp(&x))
            .ok()
            .map(|i| self.count[i][1])
            .unwrap_or(0)
    }
    pub fn type_get(&self, x: u32) -> u32 {
        self.type_
            .binary_search_by(|p| p[0].cmp(&x))
            .ok()
            .map(|i| self.type_[i][1])
            .unwrap_or(0)
    }
}

/// SV01 — phone -> spelling variants (list order preserved from the pickle)
pub struct Sylvar {
    pub entries: Vec<(String, Vec<String>)>,
    idx: HashMap<String, usize>,
}

impl Sylvar {
    fn decode(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(data);
        check_magic(&mut r, b"SV01")?;
        let n = r.uvarint();
        let mut entries = Vec::with_capacity(n);
        let mut idx = HashMap::with_capacity(n);
        for _ in 0..n {
            let phone = r.str_().to_string();
            let m = r.uvarint();
            let mut forms = Vec::with_capacity(m);
            for _ in 0..m {
                forms.push(r.str_().to_string());
            }
            idx.insert(phone.clone(), entries.len());
            entries.push((phone, forms));
        }
        Ok(Sylvar { entries, idx })
    }
    pub fn get(&self, phone: &str) -> Option<&Vec<String>> {
        self.idx.get(phone).map(|&i| &self.entries[i].1)
    }
    pub fn get_or_empty(&self, phone: &str) -> Vec<String> {
        self.idx.get(phone).map(|&i| self.entries[i].1.clone()).unwrap_or_default()
    }
}

/// PH01 — phone/form n-gram tables (from PhSTrigram.sts), f64 counts, sorted entries.
pub struct Phs {
    pub syls: Vec<String>,
    /// (x1,x2,x3,p) -> count
    pub phs_tri: Vec<[u32; 4]>,
    pub phs_tri_v: Vec<f64>,
    /// (x1,x2,x3) -> count
    pub frm_tri: Vec<[u32; 3]>,
    pub frm_tri_v: Vec<f64>,
    /// (a,b,p) -> count
    pub phs_bi: Vec<[u32; 3]>,
    pub phs_bi_v: Vec<f64>,
    /// (a,b) -> count
    pub frm_bi: Vec<[u32; 2]>,
    pub frm_bi_v: Vec<f64>,
    /// (w,p) -> count
    pub phs_uni: Vec<[u32; 2]>,
    pub phs_uni_v: Vec<f64>,
    /// (w,) -> count
    pub frm_uni: Vec<[u32; 1]>,
    pub frm_uni_v: Vec<f64>,
    /// (abs_w, abs_p) -> count
    pub abs_uni: Vec<[u32; 2]>,
    pub abs_uni_v: Vec<f64>,
    /// (abs_w,) -> count
    pub abs_frm: Vec<[u32; 1]>,
    pub abs_frm_v: Vec<f64>,
    idx: HashMap<String, u32>,
}

macro_rules! table_methods {
    ($get:ident, $rows:ident, $vals:ident, $arity:expr_2021) => {
        pub fn $get(&self, k: &[u32; $arity]) -> f64 {
            match self.$rows.binary_search_by(|p| {
                let mut a = [0u32; $arity];
                for i in 0..$arity {
                    a[i] = p[i];
                }
                a.cmp(k)
            }) {
                Ok(i) => self.$vals[i],
                Err(_) => 0.0,
            }
        }
    };
}

impl Phs {
    table_methods!(phs_tri_get, phs_tri, phs_tri_v, 4);
    table_methods!(frm_tri_get, frm_tri, frm_tri_v, 3);
    table_methods!(phs_bi_get, phs_bi, phs_bi_v, 3);
    table_methods!(frm_bi_get, frm_bi, frm_bi_v, 2);
    table_methods!(phs_uni_get, phs_uni, phs_uni_v, 2);
    table_methods!(frm_uni_get, frm_uni, frm_uni_v, 1);
    table_methods!(abs_uni_get, abs_uni, abs_uni_v, 2);
    table_methods!(abs_frm_get, abs_frm, abs_frm_v, 1);

    fn decode(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(data);
        check_magic(&mut r, b"PH01")?;
        let n = r.uvarint();
        let mut syls = Vec::with_capacity(n);
        let mut idx = HashMap::with_capacity(n);
        for i in 0..n {
            let s = r.str_().to_string();
            idx.insert(s.clone(), i as u32);
            syls.push(s);
        }
        macro_rules! table {
            ($arity:expr_2021) => {{
                let n = r.uvarint();
                let mut rows = Vec::with_capacity(n);
                let mut vals = Vec::with_capacity(n);
                for _ in 0..n {
                    let mut k = [0u32; $arity];
                    for i in 0..$arity {
                        k[i] = r.uvarint() as u32;
                    }
                    rows.push(k);
                    vals.push(r.f64_());
                }
                (rows, vals)
            }};
        }
        let (phs_tri, phs_tri_v) = table!(4);
        let (frm_tri, frm_tri_v) = table!(3);
        let (phs_bi, phs_bi_v) = table!(3);
        let (frm_bi, frm_bi_v) = table!(2);
        let (phs_uni, phs_uni_v) = table!(2);
        let (frm_uni, frm_uni_v) = table!(1);
        let (abs_uni, abs_uni_v) = table!(2);
        let (abs_frm, abs_frm_v) = table!(1);
        Ok(Phs {
            syls,
            idx,
            phs_tri,
            phs_tri_v,
            frm_tri,
            frm_tri_v,
            phs_bi,
            phs_bi_v,
            frm_bi,
            frm_bi_v,
            phs_uni,
            phs_uni_v,
            frm_uni,
            frm_uni_v,
            abs_uni,
            abs_uni_v,
            abs_frm,
            abs_frm_v,
        })
    }
    pub fn syl_id(&self, s: &str) -> u32 {
        *self.idx.get(s).unwrap_or(&u32::MAX)
    }
    pub fn has(&self, s: &str) -> bool {
        self.idx.contains_key(s)
    }
}

/// PT01 — nltk PerceptronTagger
pub struct PosModel {
    pub classes: Vec<String>,
    /// word -> tag
    pub tagdict: HashMap<String, String>,
    /// (feature, tag) -> weight; sorted by (feat, tag) for binary search
    pub weights: Vec<(String, String, f64)>,
    pub feat_hash: PerfHash,
}

impl PosModel {
    fn decode(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(data);
        check_magic(&mut r, b"PT01")?;
        let n = r.uvarint();
        let mut classes = Vec::with_capacity(n);
        for _ in 0..n {
            classes.push(r.str_().to_string());
        }
        let n = r.uvarint();
        let mut tagdict = HashMap::with_capacity(n);
        for _ in 0..n {
            let w = r.str_().to_string();
            let t = r.str_().to_string();
            tagdict.insert(w, t);
        }
        let n = r.uvarint();
        let mut weights = Vec::with_capacity(n);
        for _ in 0..n {
            let f = r.str_().to_string();
            let t = r.str_().to_string();
            let w = r.f64_();
            weights.push((f, t, w));
        }
        let mut unique_feats: Vec<(&str, u32)> = Vec::new();
        for (idx, (f, _, _)) in weights.iter().enumerate() {
            if unique_feats.last().map(|(prev, _)| *prev) != Some(f.as_str()) {
                unique_feats.push((f.as_str(), idx as u32));
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        let feat_hash = PerfHash::build(&unique_feats);
        #[cfg(not(target_arch = "wasm32"))]
        let feat_build_ms = t0.elapsed().as_millis();
        #[cfg(target_arch = "wasm32")]
        let feat_build_ms = 0;
        if std::env::var("TLTK_DEBUG").is_ok() {
            eprintln!(
                "pos decoded: classes={} tagdict={} weights={} unique_feats={} (hash in {}ms)",
                classes.len(), tagdict.len(), weights.len(), unique_feats.len(), feat_build_ms
            );
        }
        Ok(PosModel { classes, tagdict, weights, feat_hash })
    }
}

/// ED01 — DictVectorizer vocab + RandomForest trees
pub struct EduModel {
    /// feature name -> column index (in index order)
    pub vocab: Vec<(String, u32)>,
    pub feat_index: HashMap<String, u32>,
    /// classes as stored (["", "<u/>"])
    pub classes: Vec<String>,
    pub trees: Vec<EduTree>,
}

pub struct EduTree {
    /// left, right, feature per node
    pub nodes: Vec<[i32; 3]>,
    pub thresholds: Vec<f64>,
    /// per node, per class value
    pub values: Vec<Vec<f64>>,
}

impl EduModel {
    fn decode(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(data);
        check_magic(&mut r, b"ED01")?;
        let n = r.uvarint();
        let mut vocab = Vec::with_capacity(n);
        let mut feat_index = HashMap::with_capacity(n);
        for _ in 0..n {
            let name = r.str_().to_string();
            let idx = r.uvarint() as u32;
            feat_index.insert(name.clone(), idx);
            vocab.push((name, idx));
        }
        let n = r.uvarint();
        let mut classes = Vec::with_capacity(n);
        for _ in 0..n {
            classes.push(r.str_().to_string());
        }
        let nt = r.uvarint();
        let mut trees = Vec::with_capacity(nt);
        for _ in 0..nt {
            let n = r.uvarint();
            let mut nodes = Vec::with_capacity(n);
            let mut thresholds = Vec::with_capacity(n);
            let mut values = Vec::with_capacity(n);
            for _ in 0..n {
                let l = r.i32_();
                let ri = r.i32_();
                let f = r.i32_();
                let t = r.f64_();
                let nc = r.uvarint();
                let mut vals = Vec::with_capacity(nc);
                for _ in 0..nc {
                    vals.push(r.f64_());
                }
                nodes.push([l, ri, f]);
                thresholds.push(t);
                values.push(vals);
            }
            trees.push(EduTree { nodes, thresholds, values });
        }
        Ok(EduModel { vocab, feat_index, classes, trees })
    }
}

/// WV01 — gensim KeyedVectors
pub struct W2v {
    pub words: Vec<String>,
    pub dim: usize,
    /// row-major f32 [vocab][dim] normalized: each row = row / norm.
    pub vectors_unit: Vec<f32>,
    /// per-row L2 norm (gensim fill_norms)
    pub norms: Vec<f32>,
    idx: HashMap<String, usize>,
}

impl W2v {
    fn decode(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(data);
        check_magic(&mut r, b"WV01")?;
        let n = r.uvarint();
        let dim = r.uvarint();
        let mut words = Vec::with_capacity(n);
        let mut idx = HashMap::with_capacity(n);
        for i in 0..n {
            let w = r.str_().to_string();
            idx.insert(w.clone(), i);
            words.push(w);
        }
        let cnt = n * dim;
        let raw: &[u8] = r.bytes(cnt * 4);
        let mut norms = Vec::with_capacity(n);
        let mut vectors_unit = Vec::with_capacity(cnt);
        let row_bytes_len = dim * 4;
        for i in 0..n {
            let row_raw = &raw[i * row_bytes_len..(i + 1) * row_bytes_len];
            let s: f32 = row_raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| {
                    let x = f32::from_le_bytes(*c);
                    x * x
                })
                .sum();
            let norm = s.sqrt();
            norms.push(norm);
            for c in row_raw.as_chunks::<4>().0 {
                let x = f32::from_le_bytes(*c);
                vectors_unit.push(x / norm);
            }
        }
        Ok(W2v {
            words,
            dim,
            vectors_unit,
            norms,
            idx,
        })
    }
    pub fn contains(&self, w: &str) -> bool {
        self.idx.contains_key(w)
    }
    pub fn index(&self, w: &str) -> Option<usize> {
        self.idx.get(w).copied()
    }
    pub fn vector(&self, i: usize) -> &[f32] {
        &self.vectors_unit[i * self.dim..(i + 1) * self.dim]
    }
    pub fn vector_unit(&self, i: usize) -> &[f32] {
        &self.vectors_unit[i * self.dim..(i + 1) * self.dim]
    }
}
/// NR01 — CRF state/transition weights (from a crfsuite model via pycrfsuite info())
pub struct NerModel {
    pub labels: Vec<String>,
    pub transitions_idx: HashMap<(String, String), f64>,
    pub attributes_idx: HashMap<(String, String), f64>,
}

impl NerModel {
    fn decode(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(data);
        check_magic(&mut r, b"NR01")?;
        let n = r.uvarint();
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            labels.push(r.str_().to_string());
        }
        let n = r.uvarint();
        let mut transitions_idx = HashMap::with_capacity(n);
        for _ in 0..n {
            let a = r.str_().to_string();
            let b = r.str_().to_string();
            let w = r.f64_();
            transitions_idx.insert((a, b), w);
        }
        let n = r.uvarint();
        let mut attributes_idx = HashMap::with_capacity(n);
        for _ in 0..n {
            let a = r.str_().to_string();
            let b = r.str_().to_string();
            let w = r.f64_();
            attributes_idx.insert((a, b), w);
        }
        Ok(NerModel {
            labels,
            transitions_idx,
            attributes_idx,
        })
    }
}

/// WT01 — TextAna word-tier lexicon (from word.pklz; identical in tltk 1.10/1.11).
#[derive(Default)]
pub struct WordInfo {
    pub tier: Option<String>,
    pub level: Option<String>,
    pub cont: Option<String>,
    pub seman: Option<String>,
    pub form: Option<String>,
    pub lvocab: Option<String>,
    pub bvocab: Option<String>,
    pub tnc_tfidf: Option<f64>,
    pub tnc_disp: Option<f64>,
    pub ttc_disp: Option<f64>,
}

pub struct WordTier {
    entries: Vec<(String, WordInfo)>,
    idx: HashMap<String, usize>,
}

impl WordTier {
    fn decode(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(data);
        check_magic(&mut r, b"WT01")?;
        let n = r.uvarint();
        let mut entries = Vec::with_capacity(n);
        let mut idx = HashMap::with_capacity(n);
        for _ in 0..n {
            let word = r.str_().to_string();
            let flags = r.uvarint();
            // value stream order matches the converter: 7 strings then 3 f64s,
            // bit i set = field i present
            let mut info = WordInfo::default();
            let str_fields: [&mut Option<String>; 7] = [
                &mut info.tier,
                &mut info.level,
                &mut info.cont,
                &mut info.seman,
                &mut info.form,
                &mut info.lvocab,
                &mut info.bvocab,
            ];
            for (bit, slot) in str_fields.into_iter().enumerate() {
                if flags & (1 << bit) != 0 {
                    *slot = Some(r.str_().to_string());
                }
            }
            if flags & (1 << 7) != 0 {
                info.tnc_tfidf = Some(r.f64_());
            }
            if flags & (1 << 8) != 0 {
                info.tnc_disp = Some(r.f64_());
            }
            if flags & (1 << 9) != 0 {
                info.ttc_disp = Some(r.f64_());
            }
            idx.insert(word.clone(), entries.len());
            entries.push((word, info));
        }
        Ok(WordTier { entries, idx })
    }
    pub fn get(&self, word: &str) -> Option<&WordInfo> {
        self.idx.get(word).map(|&i| &self.entries[i].1)
    }
}

/// TC01 — level and genre RandomForest models
pub struct TextClassModel {
    pub level: RandomForestModel,
    pub genre: RandomForestModel,
}

pub struct RandomForestModel {
    pub feature_names: Vec<String>,
    pub feat_index: HashMap<String, usize>,
    pub classes: Vec<String>,
    pub trees: Vec<RfTree>,
}

pub struct RfTree {
    pub left: Vec<i32>,
    pub right: Vec<i32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f64>,
    /// Leaf values indexed by leaf_offset[node]
    pub leaf_values: Vec<f64>,
    pub leaf_offset: Vec<i32>,
}

impl TextClassModel {
    pub fn decode(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(data);
        check_magic(&mut r, b"TC01")?;
        let level = RandomForestModel::decode(&mut r)?;
        let genre = RandomForestModel::decode(&mut r)?;
        Ok(TextClassModel { level, genre })
    }
}

impl RandomForestModel {
    fn decode(r: &mut Reader) -> Result<Self, String> {
        let n_feats = r.uvarint();
        let mut feature_names = Vec::with_capacity(n_feats);
        let mut feat_index = HashMap::with_capacity(n_feats);
        for i in 0..n_feats {
            let name = r.str_().to_string();
            feat_index.insert(name.clone(), i);
            feature_names.push(name);
        }

        let n_classes = r.uvarint();
        let mut classes = Vec::with_capacity(n_classes);
        for _ in 0..n_classes {
            classes.push(r.str_().to_string());
        }

        let n_trees = r.uvarint();
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            let n_nodes = r.uvarint();
            let mut left = Vec::with_capacity(n_nodes);
            let mut right = Vec::with_capacity(n_nodes);
            let mut feature = Vec::with_capacity(n_nodes);
            let mut threshold = Vec::with_capacity(n_nodes);
            let mut leaf_values = Vec::new();
            let mut leaf_offset = Vec::with_capacity(n_nodes);

            for _ in 0..n_nodes {
                let l = r.i32_();
                let ri = r.i32_();
                let f = r.i32_();
                let t = r.f64_();
                left.push(l);
                right.push(ri);
                feature.push(f);
                threshold.push(t);
                if l == -1 {
                    leaf_offset.push(leaf_values.len() as i32);
                    for _ in 0..n_classes {
                        leaf_values.push(r.f64_());
                    }
                } else {
                    leaf_offset.push(-1);
                }
            }
            trees.push(RfTree {
                left,
                right,
                feature,
                threshold,
                leaf_values,
                leaf_offset,
            });
        }
        Ok(RandomForestModel {
            feature_names,
            feat_index,
            classes,
            trees,
        })
    }

    /// Predict class label for a feature vector x.
    /// In case of a tie, the lowest class index wins (matching numpy argmax).
    pub fn predict<'a>(&'a self, x: &[f64]) -> &'a str {
        let n_classes = self.classes.len();
        assert!(n_classes <= 16);
        let mut total_prob = [0.0f64; 16];
        for tree in &self.trees {
            let mut node = 0usize;
            while tree.left[node] != -1 {
                let feat = tree.feature[node] as usize;
                if x[feat] <= tree.threshold[node] {
                    node = tree.left[node] as usize;
                } else {
                    node = tree.right[node] as usize;
                }
            }
            let offset = tree.leaf_offset[node] as usize;
            let leaf = &tree.leaf_values[offset..offset + n_classes];
            let mut sum = 0.0f64;
            for &v in leaf {
                sum += v;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for c in 0..n_classes {
                    total_prob[c] += leaf[c] * inv;
                }
            }
        }
        let mut best_idx = 0;
        let mut best_val = total_prob[0];
        for (i, &v) in total_prob[..n_classes].iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        &self.classes[best_idx]
    }
}

/// All models. Blob keys are file names without .gz, e.g. "thdict.bin".
pub struct Models {
    pub thdict: Thdict,
    pub sylseg: Sylseg,
    pub sylvar: Sylvar,
    pub phs: Phs,
    pub pos: PosModel,
    pub wordtier: Option<WordTier>,
    pub edu: Option<EduModel>,
    pub w2v: Option<W2v>,
    pub ner: Option<NerModel>,
    pub textclass: Option<TextClassModel>,
    /// PRON patterns from sylrule.txt + thaisyl.txt (order preserved), compiled.
    pub pron: Vec<PronPattern>,
}

pub struct PronPattern {
    pub pattern: String,
    pub codes: Vec<String>,
    pub re: regex::Regex,
    pub n_groups: usize,
    /// `^(?:pat)$` variant for SylAna's full-string match (Python `re.match("^"+f+"$")`).
    /// Only built for patterns with capture groups — the only ones SylAna iterates.
    pub re_full: Option<regex::Regex>,
}

fn blob<'a>(blobs: &'a HashMap<String, Vec<u8>>, name: &str) -> Option<&'a [u8]> {
    blobs.get(name).map(|v| v.as_slice())
}

impl Models {
    pub fn try_from_blobs(blobs: &HashMap<String, Vec<u8>>) -> Result<Models, String> {
        let mut gz = HashMap::new();
        for (k, v) in blobs {
            let key = if k.ends_with(".bin.gz") {
                k.trim_end_matches(".gz").to_string()
            } else if k.ends_with(".bin") {
                k.clone()
            } else {
                continue;
            };
            gz.insert(key, try_gunzip(v).map_err(|e| format!("gunzip {} failed: {}", k, e))?);
        }
        let g = |k: &str| -> Option<&Vec<u8>> { gz.get(k) };
        let thdict = Thdict::decode(g("thdict.bin").ok_or_else(|| "thdict.bin missing".to_string())?)?;
        let sylseg = Sylseg::decode(g("sylseg.bin").ok_or_else(|| "sylseg.bin missing".to_string())?)?;
        let sylvar = Sylvar::decode(g("sylvar.bin").ok_or_else(|| "sylvar.bin missing".to_string())?)?;
        let phs = Phs::decode(g("phs.bin").ok_or_else(|| "phs.bin missing".to_string())?)?;
        let pos = PosModel::decode(g("pos.bin").ok_or_else(|| "pos.bin missing".to_string())?)?;
        let wordtier = g("wordtier.bin").map(|d| WordTier::decode(d)).transpose()?;
        let edu = g("edu.bin").map(|d| EduModel::decode(d)).transpose()?;
        let w2v = g("w2v.bin").map(|d| W2v::decode(d)).transpose()?;
        let ner = g("ner.bin").map(|d| NerModel::decode(d)).transpose()?;
        let textclass = g("textclass.bin").map(|d| TextClassModel::decode(d)).transpose()?;
        let sylrule = blob(blobs, "sylrule.txt").ok_or_else(|| "sylrule.txt missing".to_string())?;
        let thaisyl = blob(blobs, "thaisyl.txt").ok_or_else(|| "thaisyl.txt missing".to_string())?;
        let sylrule_str = std::str::from_utf8(sylrule).map_err(|e| format!("sylrule.txt invalid utf-8: {}", e))?;
        let thaisyl_str = std::str::from_utf8(thaisyl).map_err(|e| format!("thaisyl.txt invalid utf-8: {}", e))?;
        let pron_raw = build_pron(sylrule_str, thaisyl_str);
        let mut pron = Vec::with_capacity(pron_raw.len());
        for (pattern, codes) in pron_raw {
            let anchored = format!("^(?:{})", pattern);
            let re = regex::Regex::new(&anchored)
                .map_err(|e| format!("bad PRON pattern {:?}: {}", pattern, e))?;
            let n_groups = re.captures_len().saturating_sub(1);
            let re_full = if pattern.contains('(') {
                let full = format!("^(?:{})$", pattern);
                Some(
                    regex::Regex::new(&full)
                        .map_err(|e| format!("bad PRON pattern {:?}: {}", pattern, e))?,
                )
            } else {
                None
            };
            pron.push(PronPattern { pattern, codes, re, n_groups, re_full });
        }
        Ok(Models { thdict, sylseg, sylvar, phs, pos, wordtier, edu, w2v, ner, textclass, pron })
    }

    /// `blobs`: gunzipped model payloads keyed by file name ("thdict.bin", ...).
    pub fn from_blobs(blobs: &HashMap<String, Vec<u8>>) -> Models {
        Self::try_from_blobs(blobs).expect("failed to decode models from blobs")
    }
    pub fn load_dir(dir: &std::path::Path) -> std::io::Result<Models> {
        let mut blobs = HashMap::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("gz") {
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                let key = name.trim_end_matches(".gz").to_string();
                blobs.insert(key, std::fs::read(&path)?);
            } else if path.extension().and_then(|e| e.to_str()) == Some("txt") {
                let key = path.file_name().unwrap().to_string_lossy().to_string();
                blobs.insert(key, std::fs::read(&path)?);
            }
        }
        Ok(Models::from_blobs(&blobs))
    }
    /// Attach tier-2 model blobs (edu, w2v, and ner if given) without re-decoding tier-1.
    pub fn attach_blobs(&mut self, blobs: &HashMap<String, Vec<u8>>) -> Result<(), String> {
        let mut gz = HashMap::new();
        for (k, v) in blobs {
            let key = if k.ends_with(".bin.gz") {
                k.trim_end_matches(".gz").to_string()
            } else if k.ends_with(".bin") {
                k.clone()
            } else if k == "edu" || k == "w2v" || k == "ner" || k == "textclass" {
                format!("{}.bin", k)
            } else {
                continue;
            };
            let decompressed = if v.len() >= 2 && v[0] == 0x1f && v[1] == 0x8b {
                try_gunzip(v).map_err(|e| format!("gunzip {} failed: {}", k, e))?
            } else {
                v.clone()
            };
            gz.insert(key, decompressed);
        }
        if let Some(d) = gz.get("edu.bin") {
            self.edu = Some(EduModel::decode(d)?);
        }
        if let Some(d) = gz.get("w2v.bin") {
            self.w2v = Some(W2v::decode(d)?);
        }
        if let Some(d) = gz.get("ner.bin") {
            self.ner = Some(NerModel::decode(d)?);
        }
        if let Some(d) = gz.get("textclass.bin") {
            self.textclass = Some(TextClassModel::decode(d)?);
        }
        Ok(())
    }
}

/// Build the PRON pattern list exactly like read_sylpattern + read_syldict:
/// substitution order X, C, Y, R, K, A, Z, D, W, '.', T (T-flag from field 3),
/// then thaisyl.dict lines appended (tab-separated).
pub fn build_pron(sylrule: &str, thaisyl: &str) -> Vec<(String, Vec<String>)> {
    use std::collections::HashMap;
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    let mut push = |k: String, v: String| {
        if !map.contains_key(&k) {
            order.push(k.clone());
        }
        map.entry(k).or_default().push(v);
    };
    for line in sylrule.lines() {
        if line.starts_with('#') {
            continue;
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        let tmp: Vec<&str> = line.split(',').collect();
        if tmp.len() < 3 {
            continue;
        }
        let mut p = tmp[0].to_string();
        p = p.replace("X", "([ก-ฮ])");
        p = p.replace("C", "([กขคจดตทบปผพฟสศซ])");
        p = p.replace("Y", "([ก-ฬฮ])");
        p = p.replace("R", "([รลว])");
        p = p.replace("K", "([ก-ฮ])");
        p = p.replace(
            "A",
            "([กจฆดตบปอขฉฐถผฝศษสหคชพภทธมยรลวนณซญฑฏฌ])",
        );
        p = p.replace("Z", "([กงดนมบรลฎฏจตณถพศทสชคภปญ])");
        p = p.replace("D", "([กงดนมบวยต])");
        p = p.replace("W", "[ก-ฮ]");
        p = p.replace(".", "[\\.]");
        if tmp[2] == "T" {
            p = p.replace("T", "[่้๊๋]");
        } else {
            p = p.replace("T", "[่้๊๋]*");
        }
        push(p, tmp[1].to_string());
    }
    for line in thaisyl.lines() {
        if line.starts_with('#') {
            continue;
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        let tmp: Vec<&str> = line.split('\t').collect();
        if tmp.len() < 2 {
            continue;
        }
        push(tmp[0].to_string(), tmp[1].to_string());
    }
    order
        .into_iter()
        .map(|k| {
            let v = map.remove(&k).unwrap();
            (k, v)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_w2v_decode_bit_exactness() {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let p_w2v = std::path::Path::new(manifest_dir).join("data/w2v.bin.gz");
        let gz_bytes = std::fs::read(&p_w2v).expect("w2v.bin.gz");
        let mut decoder = GzDecoder::new(&gz_bytes[..]);
        let mut raw_data = Vec::new();
        decoder.read_to_end(&mut raw_data).expect("gunzip");

        // Method 1: old decode
        let mut r = Reader::new(&raw_data);
        check_magic(&mut r, b"WV01").unwrap();
        let n = r.uvarint();
        let dim = r.uvarint();
        for _ in 0..n {
            let _ = r.str_();
        }
        let cnt = n * dim;
        let mut vectors = Vec::with_capacity(cnt);
        let raw: &[u8] = r.bytes(cnt * 4);
        for chunk in raw.as_chunks::<4>().0 {
            vectors.push(f32::from_le_bytes(*chunk));
        }
        let mut norms1 = Vec::with_capacity(n);
        for i in 0..n {
            let row = &vectors[i * dim..(i + 1) * dim];
            let s: f32 = row.iter().map(|x| x * x).sum();
            norms1.push(s.sqrt());
        }
        let mut vectors_unit1 = Vec::with_capacity(cnt);
        for i in 0..n {
            let row = &vectors[i * dim..(i + 1) * dim];
            let norm = norms1[i];
            for &x in row {
                vectors_unit1.push(x / norm);
            }
        }

        // Method 2: new single pass decode
        let mut r = Reader::new(&raw_data);
        check_magic(&mut r, b"WV01").unwrap();
        let n2 = r.uvarint();
        let dim2 = r.uvarint();
        assert_eq!(n, n2);
        assert_eq!(dim, dim2);
        for _ in 0..n {
            let _ = r.str_();
        }
        let raw: &[u8] = r.bytes(cnt * 4);
        let mut norms2 = Vec::with_capacity(n);
        let mut vectors_unit2 = Vec::with_capacity(cnt);
        let row_bytes_len = dim * 4;
        for i in 0..n {
            let row_raw = &raw[i * row_bytes_len..(i + 1) * row_bytes_len];
            let s: f32 = row_raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| {
                    let x = f32::from_le_bytes(*c);
                    x * x
                })
                .sum();
            let norm = s.sqrt();
            norms2.push(norm);
            for c in row_raw.as_chunks::<4>().0 {
                let x = f32::from_le_bytes(*c);
                vectors_unit2.push(x / norm);
            }
        }

        assert_eq!(norms1.len(), norms2.len());
        for i in 0..norms1.len() {
            assert_eq!(
                norms1[i].to_bits(),
                norms2[i].to_bits(),
                "norm mismatch at row {}",
                i
            );
        }
        assert_eq!(vectors_unit1.len(), vectors_unit2.len());
        for i in 0..vectors_unit1.len() {
            assert_eq!(
                vectors_unit1[i].to_bits(),
                vectors_unit2[i].to_bits(),
                "vector_unit mismatch at index {}",
                i
            );
        }
        println!("All {} norms and {} vector_unit elements BIT-EXACT!", norms1.len(), vectors_unit1.len());
    }

    #[test]
    fn test_attach_blobs_w2v() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let p_w2v = std::path::Path::new(manifest_dir).join("data/w2v.bin.gz");
        let gz_bytes = std::fs::read(&p_w2v).expect("w2v.bin.gz");

        let mut models = crate::embedded::models();
        assert!(models.w2v.is_none());

        let mut blobs = HashMap::new();
        blobs.insert("w2v.bin.gz".to_string(), gz_bytes);
        models.attach_blobs(&blobs).expect("attach_blobs should succeed");

        assert!(models.w2v.is_some());
        let tltk = crate::Tltk::new(models);
        let mut sess = tltk.session();
        let res = sess.process("similarity", "หมา แมว", "medium");
        assert!(res.is_ok(), "similarity should succeed after attach: {:?}", res);
        let text = res.unwrap();
        assert!(text.contains("ความคล้ายระหว่าง 'หมา' กับ 'แมว' ="), "got: {}", text);
    }
}
