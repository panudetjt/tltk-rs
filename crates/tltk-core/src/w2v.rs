//! Word2Vec — port of tltk.corpus similarity/most_similar/outofgroup semantics
//! over gensim KeyedVectors (TNCc5model3.bin), plus the server's wire wrappers.

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
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    let rem_a = a_chunks.remainder();
    let rem_b = b_chunks.remainder();

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
}
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
) {
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

    for (i, unit) in vectors_unit.chunks_exact(100).enumerate() {
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
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(vectors_unit: &[f32], dim: usize, mean: &[f32], dists: &mut Vec<(usize, f32)>) {
    if dim == 100 && mean.len() == 100 {
        scan_avx2_dim100(vectors_unit, mean, dists);
        return;
    }
    for (i, unit) in vectors_unit.chunks_exact(dim).enumerate() {
        dists.push((i, dot_avx2(unit, mean)));
    }
}

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
        let u1 = wv.vector(i1);
        let u2 = wv.vector(i2);
        let s = dot(u1, u2);
        Some(s as f64)
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

    /// Server wrapper for plotwords — renders a PNG scatter of the first two
    /// principal components, same `IMAGE:<base64 png>` wire shape as the
    /// matplotlib backend (pixel-level parity is out of scope).
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
        let pts: Vec<Vec<f32>> = known
            .iter()
            .map(|w| wv.vector(wv.index(w).unwrap()).to_vec())
            .collect();
        let (p1, p2) = pca2(&pts);
        let proj: Vec<(f32, f32)> = pts.iter().map(|p| (dot(p, &p1), dot(p, &p2))).collect();
        let minx = proj.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
        let maxx = proj.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
        let miny = proj.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let maxy = proj.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        let (w, h) = (640usize, 400usize);
        let mut img = vec![255u8; w * h * 3];
        let span_x = (maxx - minx).abs().max(1e-6);
        let span_y = (maxy - miny).abs().max(1e-6);
        for p in &proj {
            let x = (((p.0 - minx) / span_x) * (w as f32 - 80.0) + 40.0) as usize;
            let y = (((p.1 - miny) / span_y) * (h as f32 - 80.0) + 40.0) as usize;
            let y = h - 1 - y.min(h - 1); // y up
            let x = x.min(w - 1);
            // 6px dot in #764ba2
            for dy in -3i32..=3 {
                for dx in -3i32..=3 {
                    if dx * dx + dy * dy <= 9 {
                        let xx = x as i32 + dx;
                        let yy = y as i32 + dy;
                        if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                            let o = (yy as usize * w + xx as usize) * 3;
                            img[o] = 0x76;
                            img[o + 1] = 0x4b;
                            img[o + 2] = 0xa2;
                        }
                    }
                }
            }
        }
        let png = encode_png(&img, w as u32, h as u32);
        Ok(format!("IMAGE:{}", base64_encode(&png)))
    }
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

fn pca2(pts: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let dim = pts[0].len();
    let mut mean = vec![0.0f32; dim];
    for p in pts {
        for i in 0..dim {
            mean[i] += p[i];
        }
    }
    for m in mean.iter_mut() {
        *m /= pts.len() as f32;
    }
    let mut cov = vec![vec![0.0f32; dim]; dim];
    for p in pts {
        for i in 0..dim {
            for j in 0..dim {
                cov[i][j] += (p[i] - mean[i]) * (p[j] - mean[j]);
            }
        }
    }
    let n = pts.len() as f32;
    for row in cov.iter_mut() {
        for v in row.iter_mut() {
            *v /= n;
        }
    }
    let power = |m: &Vec<Vec<f32>>| -> Vec<f32> {
        let mut v = vec![1.0f32; dim];
        for _ in 0..64 {
            let mut nv = vec![0.0f32; dim];
            for i in 0..dim {
                nv[i] = (0..dim).map(|j| m[i][j] * v[j]).sum();
            }
            let norm = (nv.iter().map(|x| x * x).sum::<f32>()).sqrt();
            for x in nv.iter_mut() {
                *x /= norm;
            }
            v = nv;
        }
        v
    };
    let p1 = power(&cov);
    // symmetric deflation: cov2 = cov − (p1ᵀ cov p1) p1 p1ᵀ
    let covp1: Vec<f32> = (0..dim).map(|i| (0..dim).map(|j| cov[i][j] * p1[j]).sum()).collect();
    let lam: f32 = (0..dim).map(|k| p1[k] * covp1[k]).sum();
    let mut cov2 = cov.clone();
    for i in 0..dim {
        for j in 0..dim {
            cov2[i][j] -= lam * p1[i] * p1[j];
        }
    }
    let p2 = power(&cov2);
    (p1, p2)
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
}
