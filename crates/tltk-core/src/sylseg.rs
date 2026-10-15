//! Syllable segmentation (port of nlp.sylseg / syl_segment / prob_wb / mergekaran).

use crate::prelude::*;
use crate::Session;

/// Insertion-ordered span map (Python dict[(i,j)] semantics: replace keeps position).
pub struct SpanMap<T> {
    pub items: Vec<(usize, T)>,
}

impl<T> SpanMap<T> {
    pub fn new() -> Self {
        SpanMap { items: Vec::new() }
    }
    pub fn get(&self, k: usize) -> Option<&T> {
        self.items.iter().find(|(kk, _)| *kk == k).map(|(_, v)| v)
    }
    /// Python dict assignment: replace value in place if key exists, else append.
    pub fn set(&mut self, k: usize, v: T) {
        if let Some(e) = self.items.iter_mut().find(|(kk, _)| *kk == k) {
            e.1 = v;
        } else {
            self.items.push((k, v));
        }
    }
    pub fn contains(&self, k: usize) -> bool {
        self.items.iter().any(|(kk, _)| *kk == k)
    }
}
impl<T> Default for SpanMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> SpanMap<T> {
    pub fn deep(&self) -> Self {
        SpanMap { items: self.items.clone() }
    }
}

/// Probabilities over syllable strings with once-ever mutation overlays.
impl<'a> Session<'a> {
    pub fn syl_id(&self, s: &str) -> Option<u32> {
        self.m.sylseg.syl_id(s)
    }

    /// BiCount[(w1,w2)] with compute_colloc's bump overlay applied.
    pub fn bi_count(&self, w1: &str, w2: &str) -> f64 {
        if let Some(v) = self.bicount_ov.get(&(w1.to_string(), w2.to_string())) {
            return *v as f64;
        }
        match (self.syl_id(w1), self.syl_id(w2)) {
            (Some(a), Some(b)) => self.m.sylseg.bi_count_get(a, b) as f64,
            _ => 0.0,
        }
    }
    pub fn count(&self, w: &str) -> f64 {
        if let Some(v) = self.count_ov.get(w) {
            return *v as f64;
        }
        match self.syl_id(w) {
            Some(a) => self.m.sylseg.count_get(a) as f64,
            None => 0.0,
        }
    }
    fn bi_type(&self, w1: &str, w2: &str) -> f64 {
        match (self.syl_id(w1), self.syl_id(w2)) {
            (Some(a), Some(b)) => self.m.sylseg.bi_type_get(a, b) as f64,
            _ => 0.0,
        }
    }
    fn type_(&self, w: &str) -> f64 {
        match self.syl_id(w) {
            Some(a) => self.m.sylseg.type_get(a) as f64,
            None => 0.0,
        }
    }

    /// Port of compute_colloc(stat="mi", w1, w2) — includes the stateful bump:
    /// if BiCount[(w1,w2)] < 1 or Count[w1] < 1 or Count[w2] < 1, all three get
    /// +1 (BiCount/Count) and TotalWord += 2 — exactly once per (w1,w2) per
    /// process. Note Python's defaultdict read side-effects are irrelevant here
    /// because the bump branch re-reads the (possibly 0) values.
    pub fn compute_colloc_mi(&mut self, w1: &str, w2: &str) -> f64 {
        let bi = self.bi_count(w1, w2);
        let c1 = self.count(w1);
        let c2 = self.count(w2);
        if bi < 1.0 || c1 < 1.0 || c2 < 1.0 {
            let key = (w1.to_string(), w2.to_string());
            if !self.colloc_bumped.contains(&key) {
                self.colloc_bumped.insert(key.clone());
                // += 1 on BiCount[(w1,w2)], Count[w1], Count[w2]
                let base_bi = self.bi_count(w1, w2) as i64;
                *self.bicount_ov.entry(key).or_insert(base_bi) += 1;
                let base_c1 = self.count(w1) as i64;
                *self.count_ov.entry(w1.to_string()).or_insert(base_c1) += 1;
                let base_c2 = self.count(w2) as i64;
                *self.count_ov.entry(w2.to_string()).or_insert(base_c2) += 1;
                self.total_word += 2;
            }
        }
        let bi = self.bi_count(w1, w2);
        let c1 = self.count(w1);
        let c2 = self.count(w2);
        let mi = (bi * self.total_word as f64) / (c1 * c2);
        // Python: value = abs(math.log(mi, 2)) — abs of the LOG, not of mi
        let value = mi.log2().abs();
        static DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *DEBUG.get_or_init(|| std::env::var("TLTK_DEBUG").is_ok()) {
            eprintln!("colloc mi {:?},{:?} bi={} c1={} c2={} TW={} -> {}", w1, w2, bi, c1, c2, self.total_word, value);
        }
        value
    }

    /// Port of prob_wb — Witten-Bell smoothed trigram probability (log space),
    /// tltk 1.10 coefficients (0.8 / 0.15 / 0.1 / 1/N).
    pub fn prob_wb(&self, w: &str, pw1: &str, pw2: &str) -> f64 {
        static DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dbg = *DEBUG.get_or_init(|| std::env::var("TLTK_DEBUG").is_ok());
        let mut p3 = 0.0;
        let mut p2 = 0.0;
        let mut p1 = 0.0;
        let tri = self.tri_count(pw2, pw1, w);
        if tri > 0.0 {
            p3 = tri / (self.bi_count(pw2, pw1) + self.bi_type(pw2, pw1));
        }
        let bi = self.bi_count(pw1, w);
        if bi > 0.0 {
            p2 = bi / (self.count(pw1) + self.type_(pw1));
        }
        let c = self.count(w);
        if c > 0.0 {
            p1 = c / (self.total_word as f64 + self.total_lex as f64);
        }
        let p = 0.8 * p3 + 0.15 * p2 + 0.1 * p1 + 1.0 / (self.total_word as f64 + self.total_lex as f64);
        if dbg {
            eprintln!(
                "prob_wb w={:?} pw1={:?} pw2={:?} tri={} p3={} bi={} p2={} c={} p1={} TW={} TL={} p={} ln={}",
                w, pw1, pw2,
                tri, p3,
                bi, p2,
                c, p1,
                self.total_word, self.total_lex, p, p.ln()
            );
        }
        p.ln()
    }

    fn tri_count(&self, x: &str, y: &str, z: &str) -> f64 {
        match (self.syl_id(x), self.syl_id(y), self.syl_id(z)) {
            (Some(a), Some(b), Some(c)) => self.m.sylseg.tri_get(a, b, c) as f64,
            _ => 0.0,
        }
    }

    /// Port of prob_trisyl — sum of prob_wb over the syllable list with SegSep BOS.
    pub fn prob_trisyl(&self, syls: &[String]) -> f64 {
        let mut probx = 1.0f64;
        let seg = SEG_SEP_STR.to_string();
        let mut pw1 = seg.clone();
        let mut pw2 = seg;
        for w in syls {
            probx += self.prob_wb(w, &pw1, &pw2);
            pw2 = pw1;
            pw1 = w.clone();
        }
        probx
    }

    /// Port of syl_segment — top-level: chunk split + non-Thai passthrough.
    pub fn syl_segment(&mut self, input: &str) -> String {
        let mut output = String::new();
        let input = preprocess(input);
        for s in input.split(SEG_SEP) {
            for inp in s.split(SSEG_SEP) {
                if inp.is_empty() {
                    continue;
                }
                let out = if starts_non_thai(inp) {
                    inp.to_string()
                } else {
                    self.sylseg(inp)
                };
                output.push_str(&out);
                output.push(SYL_SEP);
            }
            output.push_str("<s/>");
        }
        output
    }

    /// Port of sylseg — trigram chart parse over syllable-pattern edges.
    pub fn sylseg(&mut self, input: &str) -> String {
        let chars: Vec<char> = input.chars().collect();
        let n = chars.len();
        let mut schart: Vec<SpanMap<Vec<String>>> = (0..=n).map(|_| SpanMap::new()).collect();
        let mut prob_end: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();

        // edge generation: every pattern (insertion order) × every position
        let dbg = std::env::var("TLTK_DEBUG").is_ok();
        for pp in &self.m.pron {
            for i in 0..n {
                let rest: String = chars[i..].iter().collect();
                if let Some(m) = pp.re.captures(&rest) {
                    let whole = m.get(0).unwrap();
                    let k = i + whole.as_str().chars().count();
                    let form = whole.as_str().to_string();
                    if dbg {
                        eprintln!("edge ({},{}) {:?} pat={:?}", i, k, form, pp.pattern);
                    }
                    schart[i].set(k, vec![form.clone()]);
                    prob_end.insert((i, k), self.prob_trisyl(&[form]));
                }
            }
        }

        for j in 0..n {
            if let Some(s1) = schart[0].get(j).cloned() {
                let mut row0 = std::mem::take(&mut schart[0]);
                for &(k, ref s2) in &schart[j].items {
                    let mut merged_list: Vec<String> = Vec::with_capacity(s1.len() + s2.len());
                    merged_list.extend_from_slice(&s1);
                    merged_list.extend_from_slice(s2);
                    let tmp = mergekaran(&merged_list);
                    let dbg = std::env::var("TLTK_DEBUG").is_ok();
                    if !row0.contains(k) {
                        if dbg {
                            eprintln!("round j={} NEW (0,{}) tmp={:?} p={:.6}", j, k, tmp, self.prob_trisyl(&tmp));
                        }
                        row0.set(k, tmp.clone());
                        prob_end.insert((0, k), self.prob_trisyl(&tmp));
                    } else {
                        let p = self.prob_trisyl(&tmp);
                        let cur = prob_end.get(&(0, k)).copied().unwrap_or(0.0);
                        if dbg {
                            eprintln!("round j={} CMP (0,{}) tmp={:?} p={:.6} cur={:.6} take={}", j, k, tmp, p, cur, p > cur);
                        }
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

/// Port of mergekaran — reattach short karan-marked syllables to the previous syllable.
pub fn mergekaran(lst: &[String]) -> Vec<String> {
    static RE_KARAN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE_KARAN.get_or_init(|| regex::Regex::new(r"(.+)์").unwrap());
    let mut rs: Vec<String> = Vec::new();
    let mut found = false;
    let mut x = String::new();
    for s in lst.iter().rev() {
        // Python: if re.search(r"(.+)[ิุ]์",s) and len<4 -> stash; ELIF re.search(r"(.+)์",s) and len<4 -> stash.
        // A match of the first implies a match of the second, so the union is the second + len<4.
        if re.is_match(s) && s.chars().count() < 4 {
            found = true;
            x = s.clone();
            continue;
        }
        if found {
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
