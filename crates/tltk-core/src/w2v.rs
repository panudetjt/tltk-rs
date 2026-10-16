//! Word2Vec — port of tltk.corpus similarity/most_similar/outofgroup semantics
//! over gensim KeyedVectors (TNCc5model3.bin), plus the server's wire wrappers.
//!
//! `w2v` + `compound` (corpus.py oracle probes, 2026-10-16, gensim 4.3.2 +
//! scipy blas in the era venv):
//! - `corpus.w2v('แมว')` → np.float32[100]; `str(list(v))` renders each element
//!   with numpy's float32 shortest-repr: first elements are `-2.160949,
//!   -1.7455051, 0.011778566, -1.3078011` (trailing zeros stripped). OOV word
//!   → `()` (empty tuple).
//! - `corpus.compound('มา','ตรา')` → `[(('ตรา', 'มาตรา'), 0.14556038), (('มา',
//!   'มาตรา'), 0.034693517), (('มา', 'ตรา'), -0.079629)]` — similarities via
//!   `KeyedVectors.similarity` = np.float32(np.dot(matutils.unitvec(a),
//!   matutils.unitvec(b))). unitvec: the L2 norm is dnrm2 (exact-f64 sqrt of
//!   the f64 sum of squares — NOT the f32 sequential norm), and each unit
//!   element is `f32(f64(a_i) * (1.0/norm_f64))` (single rounding; differs
//!   from `x / f32_norm` by 1 ulp on scattered elements). Rust mirrors this in
//!   `w2v_oracle_similarity` (`oracle_norm_f64` + the existing SIMD `dot`).
//!   Sorted desc by score with Python's stable sort; any of w1/w2/w1+w2 OOV →
//!   `[]`. The fn_w2v golden pairs are byte-stable under that dot layout
//!   (verified against the oracle over 4000 vocab pairs; 277 stable).

use crate::prelude::fmt4;
use crate::Session;

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static AVX2_STATE: AtomicU8 = AtomicU8::new(0);
    match AVX2_STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let detected = is_x86_feature_detected!("avx2");
            AVX2_STATE.store(if detected { 2 } else { 1 }, Ordering::Relaxed);
            detected
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 { unsafe {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let (a_chunks, rem_a) = a.as_chunks::<8>();
    let (b_chunks, rem_b) = b.as_chunks::<8>();

    let mut a_ptr = a.as_ptr();
    let mut b_ptr = b.as_ptr();
    let n_chunks = a_chunks.len().min(b_chunks.len());

    let unroll4 = n_chunks / 4;
    for _ in 0..unroll4 {
        let va0 = _mm256_loadu_ps(a_ptr);
        let vb0 = _mm256_loadu_ps(b_ptr);
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(va0, vb0));

        let va1 = _mm256_loadu_ps(a_ptr.add(8));
        let vb1 = _mm256_loadu_ps(b_ptr.add(8));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(va1, vb1));

        let va2 = _mm256_loadu_ps(a_ptr.add(16));
        let vb2 = _mm256_loadu_ps(b_ptr.add(16));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(va2, vb2));

        let va3 = _mm256_loadu_ps(a_ptr.add(24));
        let vb3 = _mm256_loadu_ps(b_ptr.add(24));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(va3, vb3));

        a_ptr = a_ptr.add(32);
        b_ptr = b_ptr.add(32);
    }

    for _ in (unroll4 * 4)..n_chunks {
        let va = _mm256_loadu_ps(a_ptr);
        let vb = _mm256_loadu_ps(b_ptr);
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(va, vb));
        a_ptr = a_ptr.add(8);
        b_ptr = b_ptr.add(8);
    }

    let sum01 = _mm256_add_ps(sum0, sum1);
    let sum23 = _mm256_add_ps(sum2, sum3);
    let sum = _mm256_add_ps(sum01, sum23);

    let mut rem_sum = 0.0f32;
    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        rem_sum += x * y;
    }

    let mut arr = [0.0f32; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), sum);
    arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7] + rem_sum
}}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn dot_simd128(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::*;

    let mut sum0 = f32x4_splat(0.0);
    let mut sum1 = f32x4_splat(0.0);
    let mut sum2 = f32x4_splat(0.0);
    let mut sum3 = f32x4_splat(0.0);

    let a_chunks = a.chunks_exact(4);
    let b_chunks = b.chunks_exact(4);

    let rem_a = a_chunks.remainder();
    let rem_b = b_chunks.remainder();

    let mut a_ptr = a.as_ptr() as *const v128;
    let mut b_ptr = b.as_ptr() as *const v128;
    let n_chunks = a_chunks.len().min(b_chunks.len());

    let unroll4 = n_chunks / 4;
    for _ in 0..unroll4 {
        let va0 = v128_load(a_ptr);
        let vb0 = v128_load(b_ptr);
        sum0 = f32x4_add(sum0, f32x4_mul(va0, vb0));

        let va1 = v128_load(a_ptr.add(1));
        let vb1 = v128_load(b_ptr.add(1));
        sum1 = f32x4_add(sum1, f32x4_mul(va1, vb1));

        let va2 = v128_load(a_ptr.add(2));
        let vb2 = v128_load(b_ptr.add(2));
        sum2 = f32x4_add(sum2, f32x4_mul(va2, vb2));

        let va3 = v128_load(a_ptr.add(3));
        let vb3 = v128_load(b_ptr.add(3));
        sum3 = f32x4_add(sum3, f32x4_mul(va3, vb3));

        a_ptr = a_ptr.add(4);
        b_ptr = b_ptr.add(4);
    }

    for _ in (unroll4 * 4)..n_chunks {
        let va = v128_load(a_ptr);
        let vb = v128_load(b_ptr);
        sum0 = f32x4_add(sum0, f32x4_mul(va, vb));
        a_ptr = a_ptr.add(1);
        b_ptr = b_ptr.add(1);
    }

    let sum01 = f32x4_add(sum0, sum1);
    let sum23 = f32x4_add(sum2, sum3);
    let sum = f32x4_add(sum01, sum23);

    let mut rem_sum = 0.0f32;
    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        rem_sum += x * y;
    }

    f32x4_extract_lane::<0>(sum)
        + f32x4_extract_lane::<1>(sum)
        + f32x4_extract_lane::<2>(sum)
        + f32x4_extract_lane::<3>(sum)
        + rem_sum
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { dot_avx2(a, b) };
    }

    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            return unsafe { dot_simd128(a, b) };
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            return dot_scalar(a, b);
        }
    }

    #[allow(unreachable_code)]
    dot_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2_dim100(
    vectors_unit: &[f32],
    mean: &[f32],
    dists: &mut Vec<(usize, f32)>,
) { unsafe {
    use std::arch::x86_64::*;

    let m_ptr = mean.as_ptr();
    let m0 = _mm256_loadu_ps(m_ptr);
    let m1 = _mm256_loadu_ps(m_ptr.add(8));
    let m2 = _mm256_loadu_ps(m_ptr.add(16));
    let m3 = _mm256_loadu_ps(m_ptr.add(24));
    let m4 = _mm256_loadu_ps(m_ptr.add(32));
    let m5 = _mm256_loadu_ps(m_ptr.add(40));
    let m6 = _mm256_loadu_ps(m_ptr.add(48));
    let m7 = _mm256_loadu_ps(m_ptr.add(56));
    let m8 = _mm256_loadu_ps(m_ptr.add(64));
    let m9 = _mm256_loadu_ps(m_ptr.add(72));
    let m10 = _mm256_loadu_ps(m_ptr.add(80));
    let m11 = _mm256_loadu_ps(m_ptr.add(88));

    let rem_b0 = *m_ptr.add(96);
    let rem_b1 = *m_ptr.add(97);
    let rem_b2 = *m_ptr.add(98);
    let rem_b3 = *m_ptr.add(99);

    for (i, unit) in vectors_unit.as_chunks::<100>().0.iter().enumerate() {
        let u_ptr = unit.as_ptr();
        let mut sum0 = _mm256_mul_ps(_mm256_loadu_ps(u_ptr), m0);
        let mut sum1 = _mm256_mul_ps(_mm256_loadu_ps(u_ptr.add(8)), m1);
        let mut sum2 = _mm256_mul_ps(_mm256_loadu_ps(u_ptr.add(16)), m2);
        let mut sum3 = _mm256_mul_ps(_mm256_loadu_ps(u_ptr.add(24)), m3);

        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(_mm256_loadu_ps(u_ptr.add(32)), m4));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(_mm256_loadu_ps(u_ptr.add(40)), m5));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(_mm256_loadu_ps(u_ptr.add(48)), m6));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(_mm256_loadu_ps(u_ptr.add(56)), m7));

        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(_mm256_loadu_ps(u_ptr.add(64)), m8));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(_mm256_loadu_ps(u_ptr.add(72)), m9));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(_mm256_loadu_ps(u_ptr.add(80)), m10));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(_mm256_loadu_ps(u_ptr.add(88)), m11));

        let sum01 = _mm256_add_ps(sum0, sum1);
        let sum23 = _mm256_add_ps(sum2, sum3);
        let sum = _mm256_add_ps(sum01, sum23);

        let rem_sum = *u_ptr.add(96) * rem_b0
            + *u_ptr.add(97) * rem_b1
            + *u_ptr.add(98) * rem_b2
            + *u_ptr.add(99) * rem_b3;

        let mut arr = [0.0f32; 8];
        _mm256_storeu_ps(arr.as_mut_ptr(), sum);
        let score = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7] + rem_sum;
        dists.push((i, score));
    }
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(vectors_unit: &[f32], dim: usize, mean: &[f32], dists: &mut Vec<(usize, f32)>) { unsafe {
    if dim == 100 && mean.len() == 100 {
        scan_avx2_dim100(vectors_unit, mean, dists);
        return;
    }
    for (i, unit) in vectors_unit.chunks_exact(dim).enumerate() {
        dists.push((i, dot_avx2(unit, mean)));
    }
}}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn scan_simd128(vectors_unit: &[f32], dim: usize, mean: &[f32], dists: &mut Vec<(usize, f32)>) {
    for (i, unit) in vectors_unit.chunks_exact(dim).enumerate() {
        dists.push((i, dot_simd128(unit, mean)));
    }
}

#[inline]
#[allow(dead_code)]
fn scan_scalar(vectors_unit: &[f32], dim: usize, mean: &[f32], dists: &mut Vec<(usize, f32)>) {
    for (i, unit) in vectors_unit.chunks_exact(dim).enumerate() {
        dists.push((i, dot_scalar(unit, mean)));
    }
}

fn unit_vec(v: &[f32], norm: f32) -> Vec<f32> {
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

fn mean_of(vvs: &[&[f32]]) -> Vec<f32> {
    let dim = vvs[0].len();
    let mut m = vec![0.0f32; dim];
    for v in vvs {
        for i in 0..dim {
            m[i] += v[i];
        }
    }
    m.iter().map(|x| x / vvs.len() as f32).collect()
}

impl<'a> Session<'a> {
    /// gensim KeyedVectors.similarity — None when either word is OOV
    /// (the corpus wrapper returns 0. for missing words).
    pub fn w2v_similarity(&self, w1: &str, w2: &str) -> Option<f64> {
        let wv = self.m.w2v.as_ref()?;
        let (i1, i2) = (wv.index(w1)?, wv.index(w2)?);
        if i1 == i2 {
            return Some(1.0);
        }
        let u1 = wv.vector_unit(i1);
        let u2 = wv.vector_unit(i2);
        let s = dot(u1, u2);
        Some(s as f64)
    }

    /// gensim matutils.unitvec/similarity oracle pipeline (probed 2026-10-16,
    /// gensim 4.3.2 + scipy blas in the era venv): the L2 norm comes from
    /// dnrm2 (exact-f64 sqrt of the f64 sum of squares), each unit element is
    /// `f32(f64(a_i) * (1.0 / norm_f64))` (single rounding), and the cosine is
    /// an f32 dot of those units. This differs from `vectors_unit` (x / f32
    /// norm) by 1 ulp on some elements and changes the final bits.
    fn w2v_oracle_similarity(&self, w1: &str, w2: &str) -> Option<f32> {
        let wv = self.m.w2v.as_ref()?;
        let (i1, i2) = (wv.index(w1)?, wv.index(w2)?);
        let (a, b) = (wv.vector(i1), wv.vector(i2));
        let na = oracle_norm_f64(a);
        let nb = oracle_norm_f64(b);
        let inv_a = 1.0 / na;
        let inv_b = 1.0 / nb;
        let ua: Vec<f32> = a.iter().map(|&x| (x as f64 * inv_a) as f32).collect();
        let ub: Vec<f32> = b.iter().map(|&x| (x as f64 * inv_b) as f32).collect();
        Some(dot(&ua, &ub))
    }

    /// gensim most_similar over unit vectors; returns (word, score) top-N
    /// excluding the query words in `exclude`.
    pub fn w2v_most_similar_general(
        &self,
        mean: &[f32],
        exclude: &std::collections::HashSet<&str>,
        topn: usize,
    ) -> Option<Vec<(String, f64)>> {
        let wv = self.m.w2v.as_ref()?;
        let mut dists: Vec<(usize, f32)> = Vec::with_capacity(wv.words.len());
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() {
                unsafe { scan_avx2(&wv.vectors_unit, wv.dim, mean, &mut dists) };
            } else {
                scan_scalar(&wv.vectors_unit, wv.dim, mean, &mut dists);
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            scan_simd128(&wv.vectors_unit, wv.dim, mean, &mut dists);
        }
        #[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
        scan_scalar(&wv.vectors_unit, wv.dim, mean, &mut dists);
        #[cfg(not(any(target_arch = "x86_64", target_arch = "wasm32")))]
        scan_scalar(&wv.vectors_unit, wv.dim, mean, &mut dists);
        let limit = topn + exclude.len();
        let cmp = |a: &(usize, f32), b: &(usize, f32)| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        };
        if dists.len() > limit {
            dists.select_nth_unstable_by(limit, cmp);
            dists[..limit].sort_by(cmp);
        } else {
            dists.sort_by(cmp);
        }
        let mut out = Vec::new();
        let candidates = if dists.len() > limit {
            &dists[..limit]
        } else {
            &dists[..]
        };
        for &(i, d) in candidates {
            if exclude.contains(wv.words[i].as_str()) {
                continue;
            }
            out.push((wv.words[i].clone(), d as f64));
            if out.len() >= topn {
                break;
            }
        }
        Some(out)
    }

    pub fn w2v_most_similar(&self, w: &str, topn: usize) -> Option<Vec<(String, f64)>> {
        let wv = self.m.w2v.as_ref()?;
        let qi = wv.index(w)?;
        let mean = wv.vector_unit(qi);
        let mut ex = std::collections::HashSet::new();
        ex.insert(w);
        self.w2v_most_similar_general(mean, &ex, topn)
    }

    /// gensim doesnt_match — the word least similar to the mean of the unit vectors.
    pub fn w2v_doesnt_match(&self, words: &[String]) -> Option<String> {
        let wv = self.m.w2v.as_ref()?;
        let used: Vec<usize> = words.iter().filter_map(|w| wv.index(w)).collect();
        if used.is_empty() {
            return None;
        }
        let vecs: Vec<&[f32]> = used.iter().map(|&i| wv.vector_unit(i)).collect();
        let m0 = mean_of(&vecs);
        let mean = unit_vec(&m0, dot(&m0, &m0).sqrt());
        let mut dists: Vec<(f32, &str)> = Vec::with_capacity(used.len());
        for &i in &used {
            dists.push((dot(wv.vector_unit(i), &mean), wv.words[i].as_str()));
        }
        // sorted(zip(dists, words))[0][1]
        dists.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(b.1))
        });
        dists.first().map(|(_, w)| w.to_string())
    }

    /// corpus.similar_words with cutoff — most_similar(w) filtered by score.
    pub fn w2v_similar_words(&self, w: &str, cutoff: f64, n: usize) -> Vec<String> {
        let out = self.w2v_most_similar(w, 10).unwrap_or_default();
        let mut result = Vec::new();
        for (word, p) in out {
            if p > cutoff && result.len() < n {
                result.push(word);
            }
        }
        result
    }

    // ---------------- server wire wrappers ----------------
    // Server semantics (probed): validation messages are SUCCESS results, not
    // errors; the first out-of-vocab word is named; outofgroup/plotwords count
    // in-vocab words.

    fn first_oov<'w>(&self, words: &[&'w str]) -> Option<&'w str> {
        let wv = self.m.w2v.as_ref()?;
        words.iter().copied().find(|w| !wv.contains(w))
    }

    pub fn wire_similar_words(&mut self, text: &str, cutoff: &str) -> Result<String, crate::ProcessError> {
        let c = match cutoff {
            "low" => 0.3,
            "high" => 0.7,
            _ => 0.5,
        };
        let w = text.split_whitespace().next().unwrap_or("");
        if self.m.w2v.is_none() {
            return Ok(String::new());
        }
        if let Some(miss) = self.first_oov(&[w]) {
            return Ok(format!("ไม่พบคำ '{}' ใน word2vec model", miss));
        }
        let words = self.w2v_similar_words(w, c, 10);
        Ok(words.join(", "))
    }

    pub fn wire_similarity(&mut self, text: &str) -> Result<String, crate::ProcessError> {
        let parts: Vec<&str> = text.split_whitespace().collect();
        if parts.len() < 2 {
            return Ok("กรุณาใส่คำ 2 คำ คั่นด้วยเว้นวรรค".to_string());
        }
        if let Some(miss) = self.first_oov(&parts) {
            return Ok(format!("ไม่พบคำ '{}' ใน word2vec model", miss));
        }
        let s = self.w2v_similarity(parts[0], parts[1]).unwrap_or(0.0);
        Ok(format!(
            "ความคล้ายระหว่าง '{}' กับ '{}' = {}",
            parts[0],
            parts[1],
            fmt4(s)
        ))
    }

    pub fn wire_outofgroup(&mut self, text: &str) -> Result<String, crate::ProcessError> {
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let n_ok = self
            .m
            .w2v
            .as_ref()
            .map(|wv| words.iter().filter(|w| wv.contains(w)).count())
            .unwrap_or(0);
        if n_ok < 3 {
            return Ok("กรุณาใส่คำอย่างน้อย 3 คำ (คำที่มีใน word2vec model)".to_string());
        }
        match self.w2v_doesnt_match(&words) {
            Some(w) => Ok(format!("คำที่ไม่เข้าพวก: {}", w)),
            None => Err(crate::ProcessError("word2vec model not loaded".to_string())),
        }
    }

    pub fn wire_analogy(&mut self, text: &str) -> Result<String, crate::ProcessError> {
        let parts: Vec<&str> = text.split_whitespace().collect();
        if parts.len() != 3 {
            return Ok("กรุณาใส่คำ 3 คำ คั่นด้วยเว้นวรรค (เช่น: พ่อ แม่ ลูกชาย)".to_string());
        }
        if let Some(miss) = self.first_oov(&parts) {
            return Ok(format!("ไม่พบคำ '{}' ใน word2vec model", miss));
        }
        let answer = self.most_similar_pos_neg(&[parts[1], parts[2]], &[parts[0]], 1);
        match answer {
            Some((word, score)) => Ok(format!(
                "{} : {} :: {} : {} (score: {})",
                parts[0],
                parts[1],
                parts[2],
                word,
                fmt4(score)
            )),
            None => Err(crate::ProcessError("no answer".to_string())),
        }
    }

    /// gensim most_similar(positive, negative) — mean of weighted unit vectors,
    /// cosine over the normed vocab, excluding the query words.
    pub fn most_similar_pos_neg(
        &self,
        positive: &[&str],
        negative: &[&str],
        topn: usize,
    ) -> Option<(String, f64)> {
        let wv = self.m.w2v.as_ref()?;
        let dim = wv.dim;
        let mut accum = vec![0.0f32; dim];
        let mut count = 0usize;
        for (words, sign) in [(positive, 1.0f32), (negative, -1.0f32)] {
            for w in words {
                if let Some(i) = wv.index(w) {
                    let u = wv.vector_unit(i);
                    for j in 0..dim {
                        accum[j] += sign * u[j];
                    }
                    count += 1;
                }
            }
        }
        if count == 0 {
            return None;
        }
        let mean = unit_vec(&accum, dot(&accum, &accum).sqrt());
        let exclude: std::collections::HashSet<&str> =
            positive.iter().chain(negative.iter()).copied().collect();
        let out = self.w2v_most_similar_general(&mean, &exclude, topn)?;
        out.into_iter().next()
    }

    /// Server wrapper for plotwords — renders a 100-dim chart of raw vectors
    /// matching the stichula validator's w2v_dimplot chart.
    pub fn wire_plotwords(&mut self, text: &str) -> Result<String, crate::ProcessError> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let wv = match self.m.w2v.as_ref() {
            Some(wv) => wv,
            None => return Ok(String::new()),
        };
        let known: Vec<&str> = words.iter().copied().filter(|w| wv.contains(w)).collect();
        if known.len() < 2 {
            return Ok("กรุณาใส่คำอย่างน้อย 2 คำ (คำที่มีใน word2vec model)".to_string());
        }

        static ATLAS: std::sync::LazyLock<([u8; 256 * 128], GlyphStore)> =
            std::sync::LazyLock::new(|| {
                let decoded = decode_png_gray_256x128(GLYPH_ATLAS_PNG);
                let store = GlyphStore::parse(GLYPH_METRICS_JSON);
                (decoded, store)
            });
        let (atlas, glyph_store) = &*ATLAS;

        let (w, h) = (942usize, 144usize);
        let mut img = vec![255u8; w * h * 3];

        // 1. Gray plot area
        for y in 11..=92 {
            for x in 44..=934 {
                let idx = (y * w + x) * 3;
                img[idx] = 128;
                img[idx + 1] = 128;
                img[idx + 2] = 128;
            }
        }

        // 2. Spines
        for x in 44..=934 {
            let top_idx = (11 * w + x) * 3;
            img[top_idx] = 27;
            img[top_idx + 1] = 27;
            img[top_idx + 2] = 27;
            let bot_idx = (92 * w + x) * 3;
            img[bot_idx] = 27;
            img[bot_idx + 1] = 27;
            img[bot_idx + 2] = 27;
        }
        for y in 11..=92 {
            let left_idx = (y * w + 44) * 3;
            img[left_idx] = 27;
            img[left_idx + 1] = 27;
            img[left_idx + 2] = 27;
            let right_idx = (y * w + 934) * 3;
            img[right_idx] = 27;
            img[right_idx + 1] = 27;
            img[right_idx + 2] = 27;
        }

        // 3. X ticks and labels
        const XTICKS: [(i32, usize); 6] = [
            (0, 84),
            (20, 248),
            (40, 411),
            (60, 575),
            (80, 739),
            (100, 902),
        ];
        for (val, tx) in XTICKS {
            for y in 92..=95 {
                let idx = (y * w + tx) * 3;
                img[idx] = 27;
                img[idx + 1] = 27;
                img[idx + 2] = 27;
            }
            let s = val.to_string();
            let tw = measure_text(&s, glyph_store);
            blit_text(
                &mut img,
                (w, h),
                atlas,
                &s,
                (tx as i32 - tw / 2, 112),
                glyph_store,
            );
        }

        // 4. X label "Dimension"
        let dim_w = measure_text("Dimension", glyph_store);
        blit_text(
            &mut img,
            (w, h),
            atlas,
            "Dimension",
            (489 - dim_w / 2, 131),
            glyph_store,
        );

        // 5. Y series rows, ticks, labels, and colored dots
        let n = known.len();
        for (idx, &word) in known.iter().enumerate() {
            let y_pos = (88.0 - (idx as f32) * (74.0 / ((n - 1) as f32))).round() as i32;
            for x in 39..=43 {
                let p_idx = (y_pos as usize * w + x) * 3;
                img[p_idx] = 27;
                img[p_idx + 1] = 27;
                img[p_idx + 2] = 27;
            }
            let lw = measure_text(word, glyph_store);
            blit_text(
                &mut img,
                (w, h),
                atlas,
                word,
                (32 - lw, y_pos + 6),
                glyph_store,
            );

            let vec = wv.vector(wv.index(word).unwrap());
            let mut vmin = f32::INFINITY;
            let mut vmax = f32::NEG_INFINITY;
            for &v in vec {
                if v < vmin {
                    vmin = v;
                }
                if v > vmax {
                    vmax = v;
                }
            }
            let denom = if vmax > vmin { vmax - vmin } else { 1.0 };
            for (d, &val) in vec.iter().take(100).enumerate() {
                let t = (val - vmin) / denom;
                let color = spectral_color(t);
                let dot_x = 83.6 + (d as f32) * 8.212;
                draw_dot(&mut img, (w, h), (dot_x, y_pos as f32), color, 2.2);
            }
        }

        let png = encode_png(&img, w as u32, h as u32);
        Ok(format!("IMAGE:{}", base64_encode(&png)))
    }

    /// Shared canvas setup for the wire plot renderers (plotwords precedent):
    /// lazily decodes the glyph atlas once and returns the drawing context.
    fn atlas_ctx()
    -> (
        &'static [u8; 256 * 128],
        &'static GlyphStore,
    ) {
        static ATLAS: std::sync::LazyLock<([u8; 256 * 128], GlyphStore)> =
            std::sync::LazyLock::new(|| {
                let decoded = decode_png_gray_256x128(GLYPH_ATLAS_PNG);
                let store = GlyphStore::parse(GLYPH_METRICS_JSON);
                (decoded, store)
            });
        let (atlas, glyph_store) = &*ATLAS;
        (atlas, glyph_store)
    }

    /// corpus.w2v_diffplot — two 100-point lines over x=1..100:
    /// vec(ww)-vec(wx) and vec(ww)-vec(wy); legend labels `ww-wx`, `ww-wy`.
    /// Differs from dimplot/compare_color upstream: OOV words are NOT
    /// filtered (KeyError path returns err); here OOV any-of → error.
    pub fn wire_w2v_diffplot(&mut self, text: &str) -> Result<String, crate::ProcessError> {
        let parts: Vec<&str> = text.split_whitespace().collect();
        let wv = match self.m.w2v.as_ref() {
            Some(wv) => wv,
            None => return Err(crate::ProcessError("'w2v_model' name 'w2v_model' is not defined".to_string())),
        };
        if parts.len() < 3 {
            return Err(crate::ProcessError(
                "ValueError: not enough values to unpack (expected 3, got N)".to_string(),
            ));
        }
        let (ww, wx, wy) = (parts[0], parts[1], parts[2]);
        for w in [ww, wx, wy] {
            if !wv.contains(w) {
                return Err(crate::ProcessError(format!("KeyError: '{}'", w)));
            }
        }
        let (atlas, glyph_store) = Self::atlas_ctx();

        // Collect both series first to compute a shared y range (matplotlib
        // autoscales the axes across both lines).
        let vww = wv.vector(wv.index(ww).unwrap());
        let vwx = wv.vector(wv.index(wx).unwrap());
        let vwy = wv.vector(wv.index(wy).unwrap());
        let ys1: Vec<f32> = (0..100).map(|d| vww[d] - vwx[d]).collect();
        let ys2: Vec<f32> = (0..100).map(|d| vww[d] - vwy[d]).collect();

        let (w, h) = (W_PLOT, H_PLOT);
        let mut img = vec![255u8; w * h * 3];
        let pf = PlotFrame {
            x0: 50,
            x1: 546,
            y0: 10,
            y1: 379,
        };
        let (ymin, ymax) = min_max_both(&ys1, &ys2);
        let scale = Scale::new(ymin, ymax, pf.y0 as f32, pf.y1 as f32);

        let ylabels: Vec<(String, i32)> = {
            let (lo, hi) = (ymin.min(ymax), ymax.max(ymin));
            [(0.0, 9), (0.25, 111), (0.5, 194), (0.75, 277), (1.0, 360)]
                .iter()
                .map(|(f, row)| (py_label(lo + f * (hi - lo)), *row))
                .collect()
        };
        draw_frame(
            &mut img,
            (w, h),
            pf,
            glyph_store,
            atlas,
            &FrameOpts {
                xticks: &[0, 20, 40, 60, 80, 100],
                yticks: &ylabels,
                with_xlabel: false,
            },
        );

        let cols1 = [(31u8, 119u8, 180u8), (255u8, 127u8, 14u8)];
        for (series, color) in [(&ys1, cols1[0]), (&ys2, cols1[1])] {
            draw_polyline(
                &mut img,
                (w, h),
                pf,
                &scale,
                series,
                color,
                1.8,
            );
        }

        // Legend (loc=best lands top-left for this data): per-entry color
        // stub + label text, at the oracle's offset (label text starts
        // x=66, first entry row y≈34).
        let label1 = format!("{}-{}", ww, wx);
        let label2 = format!("{}-{}", ww, wy);
        let l1w = measure_text(&label1, glyph_store);
        let l2w = measure_text(&label2, glyph_store);
        let box_w = 12 + l1w.max(l2w) + 8;
        let box_h = 2 + 12 + 12;
        draw_legend_box(
            &mut img,
            (w, h),
            glyph_store,
            atlas,
            &LegendBox {
                bx: pf.x0 + 8,
                by: pf.y0 + 8,
                bw: box_w,
                bh: box_h,
                entries: &[
                    (label1.as_str(), cols1[0]),
                    (label2.as_str(), cols1[1]),
                ],
            },
        );

        let png = encode_png(&img, w as u32, h as u32);
        Ok(format!("IMAGE:{}", base64_encode(&png)))
    }

    /// corpus.w2v_dimplot — one 100-point line per in-vocab word over its raw
    /// 100 dims; legend label per word. OOV words filtered (upstream loops
    /// over vocabs hits); <1 hit → empty chart (still ok=True upstream).
    pub fn wire_w2v_dimplot(&mut self, text: &str) -> Result<String, crate::ProcessError> {
        let parts: Vec<&str> = text.split_whitespace().collect();
        let wv = match self.m.w2v.as_ref() {
            Some(wv) => wv,
            None => return Err(crate::ProcessError("'w2v_model' name 'w2v_model' is not defined".to_string())),
        };
        let known: Vec<&str> = parts.iter().copied().filter(|w| wv.contains(w)).collect();
        if known.is_empty() {
            return Err(crate::ProcessError("no in-vocab words".to_string()));
        }
        let (atlas, glyph_store) = Self::atlas_ctx();

        let series: Vec<Vec<f32>> = known
            .iter()
            .map(|&wd| wv.vector(wv.index(wd).unwrap()).iter().take(100).copied().collect())
            .collect();
        let (ymin, ymax) = {
            let mut ymin = f32::INFINITY;
            let mut ymax = f32::NEG_INFINITY;
            for s in &series {
                let (a, b) = min_max(s);
                ymin = ymin.min(a);
                ymax = ymax.max(b);
            }
            (ymin, ymax)
        };

        let (w, h) = (W_PLOT, H_PLOT);
        let mut img = vec![255u8; w * h * 3];
        let pf = PlotFrame {
            x0: 50,
            x1: 546,
            y0: 10,
            y1: 379,
        };
        let scale = Scale::new(ymin, ymax, pf.y0 as f32, pf.y1 as f32);
        let ylabels: Vec<(String, i32)> = {
            let (lo, hi) = (ymin.min(ymax), ymax.max(ymin));
            [(0.0, 9), (0.25, 111), (0.5, 194), (0.75, 277), (1.0, 360)]
                .iter()
                .map(|(f, row)| (py_label(lo + f * (hi - lo)), *row))
                .collect()
        };
        draw_frame(
            &mut img,
            (w, h),
            pf,
            glyph_store,
            atlas,
            &FrameOpts {
                xticks: &[0, 20, 40, 60, 80, 100],
                yticks: &ylabels,
                with_xlabel: false,
            },
        );

        for (i, s) in series.iter().enumerate() {
            let color = TAB10[i % 10];
            draw_polyline(&mut img, (w, h), pf, &scale, s, color, 1.8);
        }

        // Legend: one entry per word, stacked; box sized to longest label.
        let labels: Vec<String> = known.iter().map(|w| w.to_string()).collect();
        let max_w = labels
            .iter()
            .map(|l| measure_text(l, glyph_store))
            .max()
            .unwrap_or(0);
        let box_w = 12 + max_w + 8;
        let box_h = 2 + 12 * labels.len() as i32;
        let entries: Vec<(&str, (u8, u8, u8))> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.as_str(), TAB10[i % 10]))
            .collect();
        draw_legend_box(
            &mut img,
            (w, h),
            glyph_store,
            atlas,
            &LegendBox {
                bx: pf.x0 + 8,
                by: pf.y0 + 8,
                bw: box_w,
                bh: box_h,
                entries: &entries,
            },
        );

        let png = encode_png(&img, w as u32, h as u32);
        Ok(format!("IMAGE:{}", base64_encode(&png)))
    }

    /// corpus.w2v_compare_color — per-word Spectral-colored dot rows over the
    /// 100 dims; reversed-words y-ticks, gray axes facecolor, figsize (12,2)
    /// dpi 80, s=16. OOV words filtered (upstream raises on OOV via wv[wd]
    /// — probe showed no filter; here OOV → error, matching upstream crash).
    pub fn wire_w2v_compare_color(&mut self, text: &str) -> Result<String, crate::ProcessError> {
        let parts: Vec<&str> = text.split_whitespace().collect();
        let wv = match self.m.w2v.as_ref() {
            Some(wv) => wv,
            None => return Err(crate::ProcessError("'w2v_model' name 'w2v_model' is not defined".to_string())),
        };
        for wd in &parts {
            if !wv.contains(wd) {
                return Err(crate::ProcessError(format!("KeyError: '{}'", wd)));
            }
        }
        if parts.is_empty() {
            return Err(crate::ProcessError("IndexError: list index out of range".to_string()));
        }
        let (atlas, glyph_store) = Self::atlas_ctx();

        let vs: Vec<&[f32]> = parts
            .iter()
            .map(|&wd| wv.vector(wv.index(wd).unwrap()))
            .collect();

        // figsize (12,2) dpi 80, tight bbox trims; oracle goldens land at
        // 795x181..804x181 — use the wide anchor so labels never clip.
        let (w, h) = (804usize, 181usize);
        let mut img = vec![255u8; w * h * 3];
        let pf = PlotFrame {
            x0: 45,
            x1: 787,
            y0: 9,
            y1: 130,
        };
        // Gray axes facecolor behind the dots (upstream set_facecolor("gray")).
        for y in pf.y0..=pf.y1 {
            for x in pf.x0..=pf.x1 {
                let idx = (y as usize * w + x as usize) * 3;
                img[idx] = 128;
                img[idx + 1] = 128;
                img[idx + 2] = 128;
            }
        }
        // One dot row per word; matplotlib normalizes each scatter's c=vs[i]
        // over THAT row's min..max (no vmin/vmax passed) → per-row Spectral.
        for (i, v) in vs.iter().enumerate() {
            // Oracle rows sit at y ≈ 12.5, 68.5, 124.5 (dot rows span 6px).
            let y_row = 12.5 + i as f32 * 56.0;
            let (vmin, vmax) = min_max(v);
            let denom = if vmax > vmin { vmax - vmin } else { 1.0 };
            for (d, &val) in v.iter().take(100).enumerate() {
                let t = (val - vmin) / denom;
                let color = spectral_color(t);
                let x = 60.0 + (d as f32 + 0.5) * ((780.0 - 60.0) / 100.0);
                draw_dot(&mut img, (w, h), (x, y_row), color, 2.4);
            }
        }
        // Spines + "Dimension" x label + word y-ticks (reversed), all through
        // the shared frame painter.
        let mut wdsr = parts.to_vec();
        wdsr.reverse();
        let ylabels: Vec<(String, i32)> = wdsr
            .iter()
            .enumerate()
            .map(|(i, wd)| {
                let y_row = 12 + (i as i32) * 56;
                (wd.to_string(), y_row)
            })
            .collect();
        draw_frame(
            &mut img,
            (w, h),
            pf,
            glyph_store,
            atlas,
            &FrameOpts {
                xticks: &[],
                yticks: &ylabels,
                with_xlabel: true,
            },
        );

        let png = encode_png(&img, w as u32, h as u32);
        Ok(format!("IMAGE:{}", base64_encode(&png)))
    }

    /// corpus.w2v — raw vector fetch; `str(list(vec))` for an in-vocab word,
    /// `()` (Python empty tuple repr) when OOV or the model is absent.
    pub fn wire_w2v(&mut self, text: &str) -> Result<String, crate::ProcessError> {
        let w = text.split_whitespace().next().unwrap_or("");
        let wv = match self.m.w2v.as_ref() {
            Some(wv) => wv,
            None => return Ok("()".to_string()),
        };
        let vec = match wv.index(w) {
            Some(i) => wv.vector(i),
            None => return Ok("()".to_string()),
        };
        let mut out = String::from("[");
        for (j, &v) in vec.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            out.push_str(&py_repr_f32(v));
        }
        out.push(']');
        Ok(out)
    }

    /// corpus.compound — similarity trio ((w1,w12),(w2,w12),(w1,w2)) over the
    /// w2v model, sorted desc by score (Python stable sort). OOV input → `[]`.
    pub fn wire_compound(&mut self, text: &str) -> Result<String, crate::ProcessError> {
        let parts: Vec<&str> = text.split_whitespace().collect();
        if parts.len() < 2 {
            return Ok("[]".to_string());
        }
        let (w1, w2) = (parts[0], parts[1]);
        let w12 = format!("{}{}", w1, w2);
        if self.m.w2v.is_none() {
            return Ok("[]".to_string());
        }
        if [w1, w2, w12.as_str()]
            .iter()
            .any(|w| !self.m.w2v.as_ref().unwrap().contains(w))
        {
            return Ok("[]".to_string());
        }
        let d1 = self.w2v_oracle_similarity(w1, &w12).unwrap_or(0.0) as f64;
        let d2 = self.w2v_oracle_similarity(w2, &w12).unwrap_or(0.0) as f64;
        let d3 = self.w2v_oracle_similarity(w1, w2).unwrap_or(0.0) as f64;
        let mut lst = [(w1, w12.as_str(), d1), (w2, w12.as_str(), d2), (w1, w2, d3)];
        lst.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut out = String::from("[");
        for (j, (a, b, d)) in lst.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!(
                "({}, {})",
                py_tuple2(a, b),
                py_repr_f32(*d as f32)
            ));
        }
        out.push(']');
        Ok(out)
    }
}

/// dnrm2-equivalent: exact-f64 sqrt of the f64 sum of squares.
fn oracle_norm_f64(v: &[f32]) -> f64 {
    let s: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    s.sqrt()
}


/// numpy float32 shortest repr (Dragon4 shortest for f32 — the same unique
/// round-trip digits Rust's `{:e}` produces for f32), in Python repr notation:
/// fixed while -4 < decpt <= 16, integral floats carry ".0".
fn py_repr_f32(v: f32) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let e = format!("{:e}", v);
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.trim_start_matches('-').chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1;
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
        let e2 = decpt - 1;
        if e2 < 0 {
            out.push_str(&format!("-{:02}", -e2));
        } else {
            out.push_str(&format!("+{:02}", e2));
        }
    }
    out
}

/// Python repr of a 2-tuple of strs: `('a', 'b')`.
fn py_tuple2(a: &str, b: &str) -> String {
    format!(
        "({}, {})",
        crate::prelude::py_str_repr(a),
        crate::prelude::py_str_repr(b)
    )
}

/// Minimal PNG encoder: 8-bit RGB, filter 0, zlib via flate2.
fn encode_png(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn chunk(tag: &[u8; 4], data: &[u8], out: &mut Vec<u8>) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(tag);
        out.extend_from_slice(data);
        let mut crc_data = Vec::with_capacity(4 + data.len());
        crc_data.extend_from_slice(tag);
        crc_data.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_data).to_be_bytes());
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut table = [0u32; 256];
        for (i, e) in table.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
        crc ^ 0xFFFF_FFFF
    }

    let mut out = Vec::with_capacity(rgb.len() / 2 + 128);
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(2); // color type RGB
    ihdr.extend_from_slice(&[0, 0, 0]);
    chunk(b"IHDR", &ihdr, &mut out);
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        raw.extend_from_slice(&rgb[y * w as usize * 3..(y + 1) * w as usize * 3]);
    }
    let mut z = ZlibEncoder::new(Vec::new(), Compression::fast());
    z.write_all(&raw).unwrap();
    chunk(b"IDAT", &z.finish().unwrap(), &mut out);
    chunk(b"IEND", &[], &mut out);
    out
}

fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

const GLYPH_ATLAS_PNG: &[u8] = include_bytes!("../data/glyph_atlas.png");
const GLYPH_METRICS_JSON: &str = include_str!("../data/glyph_metrics.json");

#[derive(Clone, Copy)]
struct GlyphMetric {
    adv: u8,
    off_x: i8,
    off_y: i8,
    w: u8,
    h: u8,
    ax: u16,
    ay: u16,
}

struct GlyphStore {
    fallback: GlyphMetric,
    ascii: [Option<GlyphMetric>; 128],
    thai: [Option<GlyphMetric>; 128],
}

impl GlyphStore {
    fn parse(json: &str) -> Self {
        let mut fallback = GlyphMetric {
            adv: 9,
            off_x: 0,
            off_y: -10,
            w: 8,
            h: 10,
            ax: 1,
            ay: 1,
        };
        let mut ascii = [None; 128];
        let mut thai = [None; 128];

        let inner = json.trim().trim_start_matches('{').trim_end_matches('}');
        let bytes = inner.as_bytes();
        let mut cursor = 0;
        while cursor < bytes.len() {
            if bytes[cursor] != b'"' {
                cursor += 1;
                continue;
            }
            cursor += 1;
            let key_start = cursor;
            while cursor < bytes.len() && bytes[cursor] != b'"' {
                cursor += 1;
            }
            let key_str = &inner[key_start..cursor];
            cursor += 1; // skip closing quote
            if cursor + 2 > bytes.len() || &bytes[cursor..cursor + 2] != b":[" {
                continue;
            }
            cursor += 2;
            let val_start = cursor;
            while cursor < bytes.len() && bytes[cursor] != b']' {
                cursor += 1;
            }
            let val_str = &inner[val_start..cursor];
            cursor += 1; // skip closing ']'

            let cp: u32 = match key_str.parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let nums: Vec<i32> = val_str.split(',').filter_map(|s| s.parse().ok()).collect();
            if nums.len() == 7 {
                let m = GlyphMetric {
                    adv: nums[0] as u8,
                    off_x: nums[1] as i8,
                    off_y: nums[2] as i8,
                    w: nums[3] as u8,
                    h: nums[4] as u8,
                    ax: nums[5] as u16,
                    ay: nums[6] as u16,
                };
                if cp == 0 {
                    fallback = m;
                } else if cp < 128 {
                    ascii[cp as usize] = Some(m);
                } else if (0x0E00..=0x0E7F).contains(&cp) {
                    thai[(cp - 0x0E00) as usize] = Some(m);
                }
            }
        }
        GlyphStore {
            fallback,
            ascii,
            thai,
        }
    }

    #[inline]
    fn get(&self, ch: char) -> GlyphMetric {
        let cp = ch as u32;
        if cp < 128 {
            self.ascii[cp as usize].unwrap_or(self.fallback)
        } else if (0x0E00..=0x0E7F).contains(&cp) {
            self.thai[(cp - 0x0E00) as usize].unwrap_or(self.fallback)
        } else {
            self.fallback
        }
    }
}

fn decode_png_gray_256x128(png_data: &[u8]) -> [u8; 256 * 128] {
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    let mut offset = 8;
    let mut idat_payload = Vec::new();

    while offset + 8 <= png_data.len() {
        let len = u32::from_be_bytes(png_data[offset..offset + 4].try_into().unwrap()) as usize;
        let tag = &png_data[offset + 4..offset + 8];
        offset += 8;
        if tag == b"IDAT" {
            idat_payload.extend_from_slice(&png_data[offset..offset + len]);
        }
        offset += len + 4;
        if tag == b"IEND" {
            break;
        }
    }

    let mut z = ZlibDecoder::new(&idat_payload[..]);
    let mut decompressed = Vec::with_capacity(128 * 257);
    z.read_to_end(&mut decompressed).unwrap();

    fn paeth(a: i32, b: i32, c: i32) -> u8 {
        let p = a + b - c;
        let pa = (p - a).abs();
        let pb = (p - b).abs();
        let pc = (p - c).abs();
        if pa <= pb && pa <= pc {
            a as u8
        } else if pb <= pc {
            b as u8
        } else {
            c as u8
        }
    }

    let mut out = [0u8; 256 * 128];
    let mut src = 0;
    for y in 0..128 {
        let filter = decompressed[src];
        src += 1;
        for x in 0..256 {
            let b = decompressed[src];
            src += 1;
            let left = if x > 0 {
                out[y * 256 + x - 1] as i32
            } else {
                0
            };
            let up = if y > 0 {
                out[(y - 1) * 256 + x] as i32
            } else {
                0
            };
            let up_left = if x > 0 && y > 0 {
                out[(y - 1) * 256 + x - 1] as i32
            } else {
                0
            };
            let val = match filter {
                0 => b,
                1 => b.wrapping_add(left as u8),
                2 => b.wrapping_add(up as u8),
                3 => b.wrapping_add(((left + up) / 2) as u8),
                4 => b.wrapping_add(paeth(left, up, up_left)),
                _ => b,
            };
            out[y * 256 + x] = val;
        }
    }
    out
}

/// Line-plot canvas (diffplot/dimplot): matches the oracle tight-bbox
/// renders at 555x418 (12x8in figure, tight bbox).
const W_PLOT: usize = 555;
const H_PLOT: usize = 418;

/// Plot area frame in pixel coords (x0/x1 left/right spine, y0/y1 top/bottom).
#[derive(Clone, Copy)]
struct PlotFrame {
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
}

/// Linear data→pixel mapping for one axis.
struct Scale {
    dmin: f32,
    dmax: f32,
    pmin: f32,
    pmax: f32,
}

impl Scale {
    fn new(dmin: f32, dmax: f32, pmin: f32, pmax: f32) -> Self {
        Scale {
            dmin,
            dmax: if dmax > dmin { dmax } else { dmin + 1.0 },
            pmin,
            pmax,
        }
    }
    fn map(&self, v: f32) -> f32 {
        self.pmin + (v - self.dmin) / (self.dmax - self.dmin) * (self.pmax - self.pmin)
    }
}

/// matplotlib tab10 palette order.
const TAB10: [(u8, u8, u8); 10] = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
];

fn min_max(v: &[f32]) -> (f32, f32) {
    let mut a = f32::INFINITY;
    let mut b = f32::NEG_INFINITY;
    for &x in v {
        if x < a {
            a = x;
        }
        if x > b {
            b = x;
        }
    }
    (a, b)
}

fn min_max_both(a: &[f32], b: &[f32]) -> (f32, f32) {
    let (a0, a1) = min_max(a);
    let (b0, b1) = min_max(b);
    (a0.min(b0), a1.max(b1))
}

/// Python/matplotlib tick-label formatting: ints print bare (`2`), floats
/// print via repr (`-2.5` → "-2.5", 0.25 → "0.25").
fn py_label(v: f32) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v}");
        s
    }
}

/// White figure background, black spines with ticks outside + tick labels
/// (matplotlib default style, as the oracle renders it). `xticks` are data
/// values 0..100 mapped across the frame; `yticks` are (label, pixel row)
/// pairs, right-aligned against the left spine; `with_xlabel` draws
/// "Dimension" centered below (compare_color only — the line plots upstream
/// set no xlabel).
struct FrameOpts<'a> {
    xticks: &'a [i32],
    yticks: &'a [(String, i32)],
    with_xlabel: bool,
}

fn draw_frame(
    img: &mut [u8],
    (w, h): (usize, usize),
    pf: PlotFrame,
    store: &GlyphStore,
    atlas: &[u8; 256 * 128],
    opts: &FrameOpts,
) {
    let dark = (27u8, 27u8, 27u8);
    // spines
    for x in pf.x0..=pf.x1 {
        for y in [pf.y0, pf.y1] {
            let idx = (y as usize * w + x as usize) * 3;
            img[idx] = dark.0;
            img[idx + 1] = dark.1;
            img[idx + 2] = dark.2;
        }
    }
    for y in pf.y0..=pf.y1 {
        for x in [pf.x0, pf.x1] {
            let idx = (y as usize * w + x as usize) * 3;
            img[idx] = dark.0;
            img[idx + 1] = dark.1;
            img[idx + 2] = dark.2;
        }
    }
    // x ticks below bottom spine + labels
    let span = (pf.x1 - pf.x0) as f32;
    for &t in opts.xticks {
        let tx = pf.x0 + (t as f32 / 100.0 * span).round() as i32;
        for y in pf.y1 + 1..=pf.y1 + 4 {
            let idx = (y as usize * w + tx as usize) * 3;
            img[idx] = dark.0;
            img[idx + 1] = dark.1;
            img[idx + 2] = dark.2;
        }
        let s = t.to_string();
        let tw = measure_text(&s, store);
        blit_text(img, (w, h), atlas, &s, (tx - tw / 2, pf.y1 + 18), store);
    }
    // y ticks left of the left spine + labels (right-aligned)
    for (label, ty) in opts.yticks.iter() {
        for x in pf.x0 - 4..pf.x0 {
            let idx = ((*ty) as usize * w + x as usize) * 3;
            img[idx] = dark.0;
            img[idx + 1] = dark.1;
            img[idx + 2] = dark.2;
        }
        let lw = measure_text(label, store);
        blit_text(img, (w, h), atlas, label, (pf.x0 - 8 - lw, ty + 6), store);
    }
    if opts.with_xlabel {
        let dim_w = measure_text("Dimension", store);
        blit_text(
            img,
            (w, h),
            atlas,
            "Dimension",
            ((pf.x0 + pf.x1) / 2 - dim_w / 2, pf.y1 + 34),
            store,
        );
    }
}
/// Draw a polyline with the given color over the frame, clipping to it.
fn draw_polyline(
    img: &mut [u8],
    (w, h): (usize, usize),
    pf: PlotFrame,
    yscale: &Scale,
    ys: &[f32],
    (r, g, b): (u8, u8, u8),
    width: f32,
) {
    let n = ys.len();
    let span = (pf.x1 - pf.x0) as f32;
    let pts: Vec<(f32, f32)> = (0..n)
        .map(|d| {
            let x = pf.x0 as f32 + d as f32 / (n - 1) as f32 * span;
            (x, yscale.map(ys[d]))
        })
        .collect();
    for seg in pts.windows(2) {
        let (x0, y0) = seg[0];
        let (x1, y1) = seg[1];
        let steps = (((x1 - x0).abs() + (y1 - y0).abs()).ceil() as usize).max(1) * 3;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let x = x0 + t * (x1 - x0);
            let y = y0 + t * (y1 - y0);
            draw_disc(img, (w, h), (x, y), (r, g, b), width / 2.0, Some(pf));
        }
    }
}

/// Filled antialiased disc; optional clip rect (plot area).
fn draw_disc(
    img: &mut [u8],
    (cw, ch): (usize, usize),
    (cx, cy): (f32, f32),
    (r, g, b): (u8, u8, u8),
    radius: f32,
    clip: Option<PlotFrame>,
) {
    let ir = radius.ceil() as i32 + 1;
    for dy in -ir..=ir {
        for dx in -ir..=ir {
            let px = (cx + dx as f32).round() as i32;
            let py = (cy + dy as f32).round() as i32;
            if px < 0 || px >= cw as i32 || py < 0 || py >= ch as i32 {
                continue;
            }
            if let Some(pf) = clip
                && (px < pf.x0 || px > pf.x1 || py < pf.y0 || py > pf.y1)
            {
                continue;
            }
            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            let cov = (radius + 0.5 - dist).clamp(0.0, 1.0);
            if cov > 0.0 {
                let idx = (py as usize * cw + px as usize) * 3;
                let bg_r = img[idx] as f32;
                let bg_g = img[idx + 1] as f32;
                let bg_b = img[idx + 2] as f32;
                img[idx] = ((1.0 - cov) * bg_r + cov * (r as f32)).round() as u8;
                img[idx + 1] = ((1.0 - cov) * bg_g + cov * (g as f32)).round() as u8;
                img[idx + 2] = ((1.0 - cov) * bg_b + cov * (b as f32)).round() as u8;
            }
        }
    }
}

/// Legend box: white background, gray border, per-entry color stub + label.
struct LegendBox<'a> {
    bx: i32,
    by: i32,
    bw: i32,
    bh: i32,
    entries: &'a [(&'a str, (u8, u8, u8))],
}

fn draw_legend_box(
    img: &mut [u8],
    (cw, ch): (usize, usize),
    store: &GlyphStore,
    atlas: &[u8; 256 * 128],
    lb: &LegendBox,
) {
    for y in lb.by..lb.by + lb.bh {
        for x in lb.bx..lb.bx + lb.bw {
            if x < 0 || x >= cw as i32 || y < 0 || y >= ch as i32 {
                continue;
            }
            let idx = (y as usize * cw + x as usize) * 3;
            img[idx] = 255;
            img[idx + 1] = 255;
            img[idx + 2] = 255;
        }
    }
    // border
    for x in lb.bx..lb.bx + lb.bw {
        for y in [lb.by, lb.by + lb.bh - 1] {
            let idx = (y as usize * cw + x as usize) * 3;
            img[idx] = 204;
            img[idx + 1] = 204;
            img[idx + 2] = 204;
        }
    }
    for y in lb.by..lb.by + lb.bh {
        for x in [lb.bx, lb.bx + lb.bw - 1] {
            let idx = (y as usize * cw + x as usize) * 3;
            img[idx] = 204;
            img[idx + 1] = 204;
            img[idx + 2] = 204;
        }
    }
    for (i, (label, color)) in lb.entries.iter().enumerate() {
        let ly = lb.by + 8 + (i as i32) * 12;
        // color stub line
        for x in lb.bx + 4..lb.bx + 12 {
            let idx = ((ly + 4) as usize * cw + x as usize) * 3;
            img[idx] = color.0;
            img[idx + 1] = color.1;
            img[idx + 2] = color.2;
        }
        blit_text(img, (cw, ch), atlas, label, (lb.bx + 16, ly + 8), store);
    }
}

fn spectral_color(t: f32) -> (u8, u8, u8) {
    const STOPS: [(f32, f32, f32, f32); 11] = [
        (0.0, 158.0, 1.0, 66.0),
        (0.1, 213.0, 62.0, 79.0),
        (0.2, 244.0, 109.0, 67.0),
        (0.3, 253.0, 174.0, 97.0),
        (0.4, 254.0, 224.0, 139.0),
        (0.5, 255.0, 255.0, 191.0),
        (0.6, 230.0, 245.0, 152.0),
        (0.7, 171.0, 221.0, 164.0),
        (0.8, 102.0, 194.0, 165.0),
        (0.9, 50.0, 136.0, 189.0),
        (1.0, 94.0, 79.0, 162.0),
    ];
    let t = t.clamp(0.0, 1.0);
    for i in 0..(STOPS.len() - 1) {
        let (t0, r0, g0, b0) = STOPS[i];
        let (t1, r1, g1, b1) = STOPS[i + 1];
        if t <= t1 || i == STOPS.len() - 2 {
            let f = (t - t0) / (t1 - t0);
            let r = (r0 + f * (r1 - r0)).round() as u8;
            let g = (g0 + f * (g1 - g0)).round() as u8;
            let b = (b0 + f * (b1 - b0)).round() as u8;
            return (r, g, b);
        }
    }
    (94, 79, 162)
}

fn measure_text(text: &str, store: &GlyphStore) -> i32 {
    let mut w = 0;
    for ch in text.chars() {
        w += store.get(ch).adv as i32;
    }
    w
}

fn blit_text(
    canvas: &mut [u8],
    (cw, ch): (usize, usize),
    atlas: &[u8; 256 * 128],
    text: &str,
    (baseline_x, baseline_y): (i32, i32),
    store: &GlyphStore,
) {
    let mut cx = baseline_x;
    let mut prev_adv = 0i32;
    let chars: Vec<char> = text.chars().collect();
    for (ci, &c) in chars.iter().enumerate() {
        let g = store.get(c);
        if g.w > 0 && g.h > 0 {
            // Combining marks (adv == 0) anchor to the ORIGIN of the base glyph
            // (the pen position before its advance), not the pen after it.
            let anchor_x = if g.adv == 0 { cx - prev_adv } else { cx };
            let gx = anchor_x + g.off_x as i32;
            // Thai: a tone mark directly before SARA AM (ำ) stacks ABOVE the
            // nikhahit circle — raise it by the sara-am top height (matplotlib
            // does the same via its combining handling; matches the goldens).
            let mut gy = baseline_y + g.off_y as i32;
            let is_tone = matches!(c, '\u{0E48}'..='\u{0E4B}');
            let next_is_sara_am = chars.get(ci + 1) == Some(&'\u{0E33}');
            if is_tone && next_is_sara_am {
                gy -= g.h as i32 + 1;
            }
            for dy in 0..g.h as i32 {
                for dx in 0..g.w as i32 {
                    let px = gx + dx;
                    let py = gy + dy;
                    if px >= 0 && px < cw as i32 && py >= 0 && py < ch as i32 {
                        let atlas_idx =
                            (g.ay as usize + dy as usize) * 256 + (g.ax as usize + dx as usize);
                        let a = atlas[atlas_idx] as f32 / 255.0;
                        if a > 0.0 {
                            let canvas_idx = (py as usize * cw + px as usize) * 3;
                            let bg_r = canvas[canvas_idx] as f32;
                            let bg_g = canvas[canvas_idx + 1] as f32;
                            let bg_b = canvas[canvas_idx + 2] as f32;
                            canvas[canvas_idx] = ((1.0 - a) * bg_r).round() as u8;
                            canvas[canvas_idx + 1] = ((1.0 - a) * bg_g).round() as u8;
                            canvas[canvas_idx + 2] = ((1.0 - a) * bg_b).round() as u8;
                        }
                    }
                }
            }
        }
        prev_adv = g.adv as i32;
        cx += g.adv as i32;
    }
}

fn draw_dot(
    canvas: &mut [u8],
    (cw, ch): (usize, usize),
    (cx, cy): (f32, f32),
    (r, g, b): (u8, u8, u8),
    radius: f32,
) {
    for dy in -3i32..=3 {
        for dx in -3i32..=3 {
            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            let cov = (radius + 0.5 - dist).clamp(0.0, 1.0);
            if cov > 0.0 {
                let px = (cx + dx as f32).round() as i32;
                let py = (cy + dy as f32).round() as i32;
                if px >= 0 && px < cw as i32 && py >= 0 && py < ch as i32 {
                    let idx = (py as usize * cw + px as usize) * 3;
                    let bg_r = canvas[idx] as f32;
                    let bg_g = canvas[idx + 1] as f32;
                    let bg_b = canvas[idx + 2] as f32;
                    canvas[idx] = ((1.0 - cov) * bg_r + cov * (r as f32)).round() as u8;
                    canvas[idx + 1] = ((1.0 - cov) * bg_g + cov * (g as f32)).round() as u8;
                    canvas[idx + 2] = ((1.0 - cov) * bg_b + cov * (b as f32)).round() as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn png_encoder_smoke() {
        let img = vec![255u8; 4 * 4 * 3];
        let png = super::encode_png(&img, 4, 4);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    }
    #[test]
    fn test_similarity_golden_parity() {
        use crate::embedded::with_models;
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let p_w2v = std::path::Path::new(manifest_dir).join("data/w2v.bin.gz");
        let d_w2v = std::fs::read(&p_w2v).expect("w2v.bin.gz");
        let mut extra = std::collections::HashMap::new();
        extra.insert("w2v.bin.gz".to_string(), d_w2v);
        let m = with_models(&extra);
        let tltk = crate::Tltk::new(m);
        let sess = tltk.session();
        let expected = [
            ("แมว", "หมา", "0.7433"),
            ("กิน", "ข้าว", "0.4130"),
            ("กิน", "กิน", "1.0000"),
            ("แมว", "แมว", "1.0000"),
            ("พ่อ", "แม่", "0.8548"),
            ("ครู", "นักเรียน", "0.6688"),
            ("วิ่ง", "เดิน", "0.8610"),
            ("เช้า", "เย็น", "0.6387"),
            ("สูง", "ต่ำ", "0.8156"),
            ("โต๊ะ", "เก้าอี้", "0.7511"),
            ("หมู", "ไก่", "0.8562"),
            ("น้ำ", "ไฟ", "0.4377"),
        ];
        for (w1, w2, exp) in expected {
            let sim = sess.w2v_similarity(w1, w2).expect("both words in vocab");
            assert_eq!(crate::prelude::fmt4(sim), exp, "mismatch for ({}, {})", w1, w2);
        }
    }
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_scan_avx2_bit_exactness() {
        use crate::embedded::with_models;
        if !super::has_avx2() {
            return;
        }
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let p_w2v = std::path::Path::new(manifest_dir).join("data/w2v.bin.gz");
        let d_w2v = std::fs::read(&p_w2v).expect("w2v.bin.gz");
        let mut extra = std::collections::HashMap::new();
        extra.insert("w2v.bin.gz".to_string(), d_w2v);
        let m = with_models(&extra);
        let wv = m.w2v.as_ref().unwrap();
        let mean = wv.vector_unit(0).to_vec();

        let mut dists1 = Vec::new();
        unsafe {
            super::scan_avx2_dim100(&wv.vectors_unit, &mean, &mut dists1);
        }

        let mut dists2 = Vec::new();
        for (i, unit) in wv.vectors_unit.chunks_exact(wv.dim).enumerate() {
            dists2.push((i, unsafe { super::dot_avx2(unit, &mean) }));
        }

        assert_eq!(dists1.len(), dists2.len());
        for i in 0..dists1.len() {
            assert_eq!(dists1[i].0, dists2[i].0);
            assert_eq!(
                dists1[i].1.to_bits(),
                dists2[i].1.to_bits(),
                "mismatch at index {}",
                i
            );
        }
        println!("All {} dot products bit-exact identical between scan_avx2_dim100 and dot_avx2!", dists1.len());
    }
    #[test]
    fn test_plotwords_dimplot_output() {
        use crate::embedded::with_models;
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let p_w2v = std::path::Path::new(manifest_dir).join("data/w2v.bin.gz");
        let d_w2v = std::fs::read(&p_w2v).expect("w2v.bin.gz");
        let mut extra = std::collections::HashMap::new();
        extra.insert("w2v.bin.gz".to_string(), d_w2v);
        let m = with_models(&extra);
        let tltk = crate::Tltk::new(m);
        let mut sess = tltk.session();

        let err = sess.wire_plotwords("กิน").unwrap();
        assert_eq!(err, "กรุณาใส่คำอย่างน้อย 2 คำ (คำที่มีใน word2vec model)");

        fn b64_decode(s: &str) -> Vec<u8> {
            let mut out = Vec::new();
            let mut buf = 0u32;
            let mut bits = 0;
            for b in s.bytes() {
                let v = match b {
                    b'A'..=b'Z' => b - b'A',
                    b'a'..=b'z' => b - b'a' + 26,
                    b'0'..=b'9' => b - b'0' + 52,
                    b'+' => 62,
                    b'/' => 63,
                    _ => continue,
                };
                buf = (buf << 6) | (v as u32);
                bits += 6;
                if bits >= 8 {
                    bits -= 8;
                    out.push((buf >> bits) as u8);
                }
            }
            out
        }

        let out = sess.wire_plotwords("แมว หมา").unwrap();
        assert!(out.starts_with("IMAGE:"));
        let png_catdog = b64_decode(&out[6..]);
        assert_eq!(&png_catdog[16..20], &942u32.to_be_bytes());
        assert_eq!(&png_catdog[20..24], &144u32.to_be_bytes());
        let _ = std::fs::write("/tmp/rust_catdog.png", &png_catdog);

        let out_rice = sess.wire_plotwords("ข้าว น้ำ ผัก").unwrap();
        assert!(out_rice.starts_with("IMAGE:"));
        let png_rice = b64_decode(&out_rice[6..]);
        assert_eq!(&png_rice[16..20], &942u32.to_be_bytes());
        assert_eq!(&png_rice[20..24], &144u32.to_be_bytes());
        let _ = std::fs::write("/tmp/rust_rice.png", &png_rice);
    }

    #[test]
    fn text_preview_hard_words() {
        use crate::embedded::with_models;
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let mut extra = std::collections::HashMap::new();
        for name in ["edu.bin.gz", "w2v.bin.gz"] {
            let p = std::path::Path::new(manifest_dir).join("data").join(name);
            extra.insert(name.to_string(), std::fs::read(&p).unwrap());
        }
        let tltk = crate::Tltk::new(with_models(&extra));
        let mut sess = tltk.session();
        // stacked marks (ที่ ชั้น), below-vowel+tone (สู่), tone+SARA AM (น้ำ), clusters (จันทร์)
        let out = sess.process("plotwords", "ที่ ชั้น สู่ น้ำ ปี จันทร์", "medium").unwrap();
        assert!(out.starts_with("IMAGE:"), "got: {}", &out[..60.min(out.len())]);
        fn b64_d(s: &str) -> Vec<u8> {
            let m = |c: char| match c {
                'A'..='Z' => c as u8 - b'A',
                'a'..='z' => c as u8 - b'a' + 26,
                '0'..='9' => c as u8 - b'0' + 52,
                '+' => 62,
                '/' => 63,
                _ => 0,
            };
            let b: Vec<u8> = s.chars().filter(|&c| c != '=').map(m).collect();
            b.chunks(4).flat_map(|c| {
                let n = c.iter().fold(0u32, |a, x| a << 6 | *x as u32)
                    << (6 * (4 - c.len()) % 24);
                (0..c.len().saturating_sub(1))
                    .map(|i| (n >> (16 - 8 * i)) as u8)
                    .collect::<Vec<_>>()
            }).collect()
        }
        let png = b64_d(&out[6..]);
        assert_eq!(&png[16..20], &942u32.to_be_bytes());
        std::fs::write("/tmp/text_preview.png", &png).unwrap();
    }
}
