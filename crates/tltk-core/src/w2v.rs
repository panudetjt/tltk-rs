//! Word2Vec — port of tltk.corpus similarity/most_similar/outofgroup semantics
//! over gensim KeyedVectors (TNCc5model3.bin), plus the server's wire wrappers.

use crate::prelude::fmt4;
use crate::Session;

/// f32 dot product (naive accumulation order).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn unit_vec(v: &[f32], norm: f32) -> Vec<f32> {
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

fn mean_of(vvs: &[Vec<f32>]) -> Vec<f32> {
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
        let v1 = wv.vector(i1);
        let v2 = wv.vector(i2);
        let n1 = dot(v1, v1) as f64;
        let n2 = dot(v2, v2) as f64;
        // gensim: dot(v1,v2) / (norm(v1)*norm(v2)) computed in float32 then float()
        let s = dot(v1, v2);
        Some((s / ((n1.sqrt() as f32) * (n2.sqrt() as f32))) as f64)
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
        let dim = wv.dim;
        let mut dists: Vec<(usize, f32)> = Vec::with_capacity(wv.words.len());
        for i in 0..wv.words.len() {
            let row = &wv.vectors[i * dim..(i + 1) * dim];
            let unit: Vec<f32> = row.iter().map(|x| x / wv.norms[i]).collect();
            dists.push((i, dot(&unit, mean)));
        }
        dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = Vec::new();
        let limit = topn + exclude.len();
        for (i, d) in dists {
            if exclude.contains(wv.words[i].as_str()) {
                continue;
            }
            out.push((wv.words[i].clone(), d as f64));
            if out.len() >= topn || out.len() >= limit {
                break;
            }
        }
        Some(out)
    }

    pub fn w2v_most_similar(&self, w: &str, topn: usize) -> Option<Vec<(String, f64)>> {
        let wv = self.m.w2v.as_ref()?;
        let qi = wv.index(w)?;
        let mean = unit_vec(wv.vector(qi), wv.norms[qi]);
        let mut ex = std::collections::HashSet::new();
        ex.insert(w);
        self.w2v_most_similar_general(&mean, &ex, topn)
    }

    /// gensim doesnt_match — the word least similar to the mean of the unit vectors.
    pub fn w2v_doesnt_match(&self, words: &[String]) -> Option<String> {
        let wv = self.m.w2v.as_ref()?;
        let used: Vec<usize> = words.iter().filter_map(|w| wv.index(w)).collect();
        if used.is_empty() {
            return None;
        }
        let vecs: Vec<Vec<f32>> = used
            .iter()
            .map(|&i| unit_vec(wv.vector(i), wv.norms[i]))
            .collect();
        let m0 = mean_of(&vecs);
        let mean = unit_vec(&m0, dot(&m0, &m0).sqrt());
        let mut dists: Vec<(f32, &str)> = Vec::with_capacity(used.len());
        for &i in &used {
            let row = wv.vector(i);
            let unit: Vec<f32> = row.iter().map(|x| x / wv.norms[i]).collect();
            dists.push((dot(&unit, &mean), wv.words[i].as_str()));
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
                    let u = unit_vec(wv.vector(i), wv.norms[i]);
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
}
