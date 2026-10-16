//! EDU segmentation — port of nlp.segment + features()/distance() and sklearn
//! DictVectorizer/RandomForestClassifier inference.

use crate::Session;

/// nlp.features(sentence, index) — ordered dict; string values become
/// "k=v" columns, bool/numeric become "k" columns (sklearn DictVectorizer).
pub fn edu_features(sentence: &[&str], index: usize) -> Vec<(String, f64)> {
    let word = sentence[index];
    let len = sentence.len();
    let mut f: Vec<(String, f64)> = Vec::new();
    let push_str = |f: &mut Vec<(String, f64)>, k: &str, v: &str| f.push((format!("{}={}", k, v), 1.0));
    let push_bool = |f: &mut Vec<(String, f64)>, k: &str, v: bool| f.push((k.to_string(), if v { 1.0 } else { 0.0 }));
    let push_num = |f: &mut Vec<(String, f64)>, k: &str, v: f64| f.push((k.to_string(), v));

    push_str(&mut f, "word", word);
    if index == 0 {
        push_str(&mut f, "prev_word", "");
    } else {
        push_str(&mut f, "prev_word", sentence[index - 1]);
    }
    if index <= 1 {
        push_str(&mut f, "prev_biword", "");
    } else {
        push_str(&mut f, "prev_biword", &format!("{}{}", sentence[index - 2], sentence[index - 1]));
    }
    if index == len - 1 {
        push_str(&mut f, "next_word", "");
    } else {
        push_str(&mut f, "next_word", sentence[index + 1]);
    }
    if index >= len - 2 {
        push_str(&mut f, "next_biword", "");
    } else {
        push_str(&mut f, "next_biword", &format!("{}{}", sentence[index + 1], sentence[index + 2]));
    }
    push_bool(&mut f, "has_hyphen", word.contains('-'));
    push_bool(&mut f, "is_numeric", word == "DIGIT");
    push_bool(&mut f, "is_space", word.contains("<s/>"));
    push_bool(&mut f, "after_space", index != 0 && sentence[index - 1] == "<s/>");
    push_bool(&mut f, "before_space", index != len - 1 && sentence[index + 1] == "<s/>");
    let dist = if !sentence[..index].contains(&"<s/>") {
        0.0
    } else {
        distance(sentence, index) as f64
    };
    push_num(&mut f, "dist_space", dist);
    push_bool(&mut f, "Mai_yamok", word.contains('ๆ'));
    push_bool(&mut f, "after_FOR", index != 0 && sentence[index - 1] == "FOREIGN");
    push_bool(&mut f, "before_FOR", index != len - 1 && sentence[index + 1] == "FOREIGN");
    push_bool(&mut f, "after_DIG", index != 0 && sentence[index - 1] == "DIGIT");
    push_bool(&mut f, "before_DIG", index != len - 1 && sentence[index + 1] == "DIGIT");
    f
}

/// nlp.distance
fn distance(sentence: &[&str], index: usize) -> usize {
    let mut i = index - 1;
    while sentence[i] != "<s/>" && i > 0 {
        i -= 1;
    }
    index - i
}

/// sklearn DictVectorizer.transform — column index per feature, value.
pub(crate) fn vectorize(sess: &Session, features: &[(String, f64)]) -> Vec<(u32, f64)> {
    let mut out = Vec::new();
    for (k, v) in features {
        if let Some(&idx) = sess.m.edu.as_ref().unwrap().feat_index.get(k) {
            out.push((idx, *v));
        }
    }
    out.sort_by_key(|(i, _)| *i);
    out
}

/// RandomForestClassifier.predict_proba — average tree probabilities, then
/// sklearn's predict = argmax (first max wins: index 0 = "").
pub(crate) fn rf_predict(sess: &Session, row: &[(u32, f64)]) -> (usize, [f64; 2]) {
    let edu = sess.m.edu.as_ref().unwrap();
    let n_classes = edu.classes.len().max(2);
    let mut proba = vec![0.0f64; n_classes];
    for tree in &edu.trees {
        let mut node = 0usize;
        loop {
            let [l, r, f] = tree.nodes[node];
            if l == -1 && r == -1 {
                break;
            }
            let fv = row
                .binary_search_by(|(i, _)| (*i as i32).cmp(&f))
                .ok()
                .map(|i| row[i].1)
                .unwrap_or(0.0);
            node = if fv <= tree.thresholds[node] { l as usize } else { r as usize };
        }
        let vals = &tree.values[node];
        let sum: f64 = vals.iter().sum();
        for (ci, v) in vals.iter().enumerate() {
            if sum > 0.0 {
                proba[ci] += v / sum;
            }
        }
    }
    let nt = edu.trees.len() as f64;
    for p in proba.iter_mut() {
        *p /= nt;
    }
    let mut best = 0usize;
    for i in 1..proba.len() {
        if proba[i] > proba[best] {
            best = i;
        }
    }
    (best, [proba[0], proba.get(1).copied().unwrap_or(0.0)])
}

impl<'a> Session<'a> {
    /// Port of nlp.segment (EDU + word segmentation).
    pub fn segment(&mut self, txt: &str) -> String {
        if self.m.edu.is_none() {
            return String::new();
        }
        let txt = format!("{} ", txt);
        let sylseg_out = self.syl_segment(&txt);
        let sylseg_out = sylseg_out.replace(' ', "<s/>");
        static RE1: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE2: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re1 = RE1.get_or_init(|| regex::Regex::new(r"([^~])<s/>").unwrap());
        let re2 = RE2.get_or_init(|| regex::Regex::new(r"<s/>([^~])").unwrap());
        let sylseg_out = re1.replace_all(&sylseg_out, "${1}~<s/>").into_owned();
        let sylseg_out = re2.replace_all(&sylseg_out, "<s/>~${1}").into_owned();

        static RE_DIG: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE_FOR: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re_dig = RE_DIG.get_or_init(|| regex::Regex::new(r"~[0-9\.\,]+~").unwrap());
        let re_for = RE_FOR.get_or_init(|| {
            regex::Regex::new(r#"~[a-zA-Z0-9\\\/\?'"\(\)\.]+~"#).unwrap()
        });
        let sylcopy = re_dig.replace_all(&sylseg_out, "~DIGIT~").into_owned();
        let sylcopy = re_for.replace_all(&sylcopy, "~FOREIGN~").into_owned();

        let parcopy: Vec<&str> = sylcopy.split('~').collect();
        let par: Vec<&str> = sylseg_out.split('~').collect();

        let mut tags: Vec<&str> = Vec::with_capacity(par.len());
        for index in 0..par.len() {
            let feats = edu_features(&parcopy, index);
            let row = vectorize(self, &feats);
            let (cls, _) = rf_predict(self, &row);
            tags.push(if cls == 0 { "" } else { "<u/>" });
        }

        let mut output = String::new();
        let mut syl_seq = String::new();
        for (w, t) in par.iter().zip(tags.iter()) {
            if *t == "<u/>" {
                let _ = self.word_segment(&syl_seq);
                let out = self.wordseg_colloc(&syl_seq);
                output.push_str(&out);
                output.push_str("<u/>");
                syl_seq = String::new();
            } else {
                syl_seq.push_str(w);
                syl_seq.push('~');
            }
        }
        static RE3: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        static RE4: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re3 = RE3.get_or_init(|| regex::Regex::new(r"([0-9])<s/>").unwrap());
        let re4 = RE4.get_or_init(|| regex::Regex::new(r"<s/>([0-9])").unwrap());
        let output = re3.replace_all(&output, "${1}|<s/>").into_owned();
        
        re4.replace_all(&output, "<s/>|${1}").into_owned()
    }
}

#[cfg(test)]
mod tests {
    use crate::embedded;
    use crate::Tltk;
    use std::collections::HashMap;

    fn test_session() -> Tltk {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = std::path::Path::new(manifest_dir).join("data/edu.bin.gz");
        let data = std::fs::read(&path).expect("edu.bin.gz should exist");
        let mut extra = HashMap::new();
        extra.insert("edu.bin.gz".to_string(), data);
        Tltk::new(embedded::with_models(&extra))
    }

    #[test]
    fn test_segment_oracle_parity() {
        let tltk = test_session();
        let mut sess = tltk.session();

        // All 5 corpus segment texts:
        assert_eq!(
            sess.segment("แมวกินปลาทะเล"),
            "แมว|กิน|ปลาทะเล|<s/><u/>"
        );
        assert_eq!(
            sess.segment("ผมชอบกินข้าวผัดกะเพราหมูสับ"),
            "ผม|ชอบ|กิน|ข้าวผัดกะเพราหมู|สับ|<s/><u/>"
        );
        assert_eq!(
            sess.segment("กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย"),
            "กรุงเทพมหานคร|เป็น|เมืองหลวง|ของ|ประเทศ|ไทย|<s/><u/>"
        );
        assert_eq!(
            sess.segment("วันนี้อากาศดีมาก ๆ"),
            "วันนี้|อากาศ|ดี|มาก|ๆ|<s/><s/><u/>"
        );
        assert_eq!(
            sess.segment("เขาไปตลาดซื้อผัก ผลไม้ และของชำ"),
            "เขา|ไป|ตลาด|ซื้อ|ผัก|<s/>|ผลไม้|<s/>|และ|ของชำ|<s/><u/>"
        );
        // Crafted longer text:
        assert_eq!(
            sess.segment("แมวกินปลาทะเลและผมชอบกินข้าวผัดกะเพราหมูสับที่กรุงเทพมหานคร"),
            "แมว|กิน|ปลาทะเล|และ|ผม|ชอบ|กิน|ข้าวผัดกะเพราหมู|สับ|ที่|กรุงเทพมหานคร|<s/><u/>"
        );
        // New everyday Thai sentences:
        assert_eq!(
            sess.segment("แม่ไปตลาดซื้อกับข้าว"),
            "แม่|ไป|ตลาด|ซื้อ|กับข้าว|<s/><u/>"
        );
        assert_eq!(
            sess.segment("เขาชอบอ่านหนังสือพิมพ์"),
            "เขา|ชอบ|อ่าน|หนังสือพิมพ์|<s/><u/>"
        );
        assert_eq!(
            sess.segment("การเดินทางโดยรถไฟสะดวกและปลอดภัย"),
            "การ|เดินทาง|โดย|รถไฟ|สะดวก|และ|ปลอดภัย|<s/><u/>"
        );
        assert_eq!(
            sess.segment("นายสมชาย ใจดี เดินทางไปทำงานที่กรุงเทพฯ"),
            "นาย|สม|ชาย|<s/>|ใจดี<u/>เดินทาง|ไป|ทำงาน|ที่|กรุงเทพ|ฯ<s/><u/>"
        );
        assert_eq!(
            sess.segment("เด็กๆ กำลังวิ่งเล่นในสนามหญ้าอย่างสนุกสนาน ขณะที่คุณพ่อคุณแม่นั่งดูอยู่ใต้ต้นไม้ใหญ่"),
            "เด็กๆ|<s/><s/>|กำลัง|วิ่ง|เล่น|ใน|สนามหญ้า|อย่าง|สนุกสนาน<u/>ขณะที่|คุณ|พ่อคุณ|แม่|นั่ง|ดู|อยู่|ใต้|ต้นไม้|ใหญ่|<s/><u/>"
        );
        // Long text (~150-250 chars: 209 chars):
        assert_eq!(
            sess.segment("กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทยและมีประชากรอาศัยอยู่เป็นจำนวนมากเขาไปตลาดเพื่อซื้อผักผลไม้และอาหารสดสำหรับทำกับข้าวในตอนเย็นนักเรียนกำลังอ่านหนังสือและค้นคว้าข้อมูลทางวิชาการอยู่ในห้องสมุดของมหาวิทยาลัย"),
            "กรุงเทพมหานคร|เป็น|เมืองหลวง|ของ|ประเทศ|ไทย|และ|มี|ประชากร|อาศัยอยู่|เป็น|จำนวน|มาก|เขา|ไป|ตลาด|เพื่อ|ซื้อ|ผัก|ผลไม้|และ|อาหารสด|สำหรับ|ทำกับข้าว|ใน|ตอนเย็น|นักเรียน|กำลัง|อ่านหนังสือ|และ|ค้นคว้า|ข้อมูล|ทาง|วิชาการ|อยู่|ใน|ห้องสมุด|ของ|มหาวิทยาลัย|<s/><u/>"
        );
    }
}
