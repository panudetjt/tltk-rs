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

/// SG01 — syllable trigram counts + derived maps (from sylseg.3g)
pub struct Sylseg {
    pub syls: Vec<String>,
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
        if std::env::var("TLTK_DEBUG").is_ok() {
            eprintln!(
                "sylseg decoded: syls={} tri={} bicount={} bitype={} count={} type={} TW={} TL={}",
                syls.len(), tri.len(), bi_count.len(), bi_type.len(), count.len(), type_.len(), total_word, total_lex
            );
        }
        Ok(Sylseg { syls, tri, bi_count, bi_type, count, type_, total_word, total_lex })
    }
    pub fn syl_id(&self, s: &str) -> Option<u32> {
        self.syls.binary_search_by(|p| p.as_str().cmp(s)).ok().map(|i| i as u32)
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
    ($get:ident, $rows:ident, $vals:ident, $arity:expr) => {
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
            ($arity:expr) => {{
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
        Ok(PosModel { classes, tagdict, weights })
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
    /// row-major f32 [vocab][dim]
    pub vectors: Vec<f32>,
    /// row-major f32 [vocab][dim] normalized: each row = row / norm.
    /// Memory cost: one extra f32 matrix (n * dim * 4 bytes, doubling vector memory)
    /// to eliminate per-query unit vector allocations while keeping raw vectors for exact similarity parity.
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
        let mut vectors = Vec::with_capacity(cnt);
        let raw: &[u8] = r.bytes(cnt * 4);
        for chunk in raw.chunks_exact(4) {
            vectors.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let mut norms = Vec::with_capacity(n);
        for i in 0..n {
            let row = &vectors[i * dim..(i + 1) * dim];
            let s: f32 = row.iter().map(|x| x * x).sum();
            norms.push(s.sqrt());
        }
        let mut vectors_unit = Vec::with_capacity(cnt);
        for i in 0..n {
            let row = &vectors[i * dim..(i + 1) * dim];
            let norm = norms[i];
            for &x in row {
                vectors_unit.push(x / norm);
            }
        }
        Ok(W2v { words, dim, vectors, vectors_unit, norms, idx })
    }
    pub fn contains(&self, w: &str) -> bool {
        self.idx.contains_key(w)
    }
    pub fn index(&self, w: &str) -> Option<usize> {
        self.idx.get(w).copied()
    }
    pub fn vector(&self, i: usize) -> &[f32] {
        &self.vectors[i * self.dim..(i + 1) * self.dim]
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

/// All models. Blob keys are file names without .gz, e.g. "thdict.bin".
pub struct Models {
    pub thdict: Thdict,
    pub sylseg: Sylseg,
    pub sylvar: Sylvar,
    pub phs: Phs,
    pub pos: PosModel,
    pub edu: Option<EduModel>,
    pub w2v: Option<W2v>,
    pub ner: Option<NerModel>,
    /// PRON patterns from sylrule.txt + thaisyl.txt (order preserved), compiled.
    pub pron: Vec<PronPattern>,
}

pub struct PronPattern {
    pub pattern: String,
    pub codes: Vec<String>,
    pub re: regex::Regex,
    pub n_groups: usize,
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
        let edu = g("edu.bin").map(|d| EduModel::decode(d)).transpose()?;
        let w2v = g("w2v.bin").map(|d| W2v::decode(d)).transpose()?;
        let ner = g("ner.bin").map(|d| NerModel::decode(d)).transpose()?;
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
            pron.push(PronPattern { pattern, codes, re, n_groups });
        }
        Ok(Models { thdict, sylseg, sylvar, phs, pos, edu, w2v, ner, pron })
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
