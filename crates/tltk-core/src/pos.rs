//! POS tagging — port of nltk.tag.perceptron (AveragedPerceptron + PerceptronTagger)
//! and nlp.pos_tag / change_tag.

use crate::Session;

const START: [&str; 2] = ["-START-", "-START2-"];
const END: [&str; 2] = ["-END-", "-END2-"];

/// Python str.isdigit() — true for Nd digits including Thai ๐-๙.
fn py_isdigit_char(c: char) -> bool {
    c.is_ascii_digit() || ('\u{0E50}'..='\u{0E59}').contains(&c)
}

/// nltk PerceptronTagger.normalize
fn normalize(word: &str) -> String {
    if word.contains('-') && !word.starts_with('-') {
        return "!HYPHEN".to_string();
    }
    let all_digit = !word.is_empty() && word.chars().all(py_isdigit_char);
    if all_digit && word.chars().count() == 4 {
        return "!YEAR".to_string();
    }
    if word.chars().next().map(py_isdigit_char).unwrap_or(false) {
        return "!DIGITS".to_string();
    }
    word.to_lowercase()
}

fn suffix3(s: &str) -> &str {
    let chars: Vec<char> = s.chars().collect();
    let start = chars.len().saturating_sub(3);
    &s[char_idx(s, start)..]
}

fn char_idx(s: &str, char_i: usize) -> usize {
    s.char_indices().nth(char_i).map(|(i, _)| i).unwrap_or(s.len())
}

/// nltk AveragedPerceptron.predict — sum weights, best = max((score, label)).
fn predict(sess: &Session, features: &[(&str, f64)]) -> String {
    use std::collections::HashMap;
    let mut scores: HashMap<&str, f64> = HashMap::new();
    for (feat, value) in features {
        if *value == 0.0 {
            continue;
        }
        // feature block start precomputed in perfect hash; verify hit then scan block
        if let Some(start_idx) = sess.m.pos.feat_hash.get(feat) {
            let mut i = start_idx as usize;
            if i < sess.m.pos.weights.len() && sess.m.pos.weights[i].0 == *feat {
                while i < sess.m.pos.weights.len() && sess.m.pos.weights[i].0 == *feat {
                    let (_f, t, w) = &sess.m.pos.weights[i];
                    *scores.entry(t.as_str()).or_insert(0.0) += value * w;
                    i += 1;
                }
            }
        }
    }
    let mut best: Option<(&str, (f64, &str))> = None;
    for label in &sess.m.pos.classes {
        let s = scores.get(label.as_str()).copied().unwrap_or(0.0);
        let key = (s, label.as_str());
        match best {
            None => best = Some((label.as_str(), key)),
            Some((_, bk)) => {
                // max by (score, label)
                if key.0.partial_cmp(&bk.0).unwrap_or(std::cmp::Ordering::Equal) == std::cmp::Ordering::Greater
                    || (key.0 == bk.0 && key.1 > bk.1)
                {
                    best = Some((label.as_str(), key));
                }
            }
        }
    }
    best.map(|(l, _)| l.to_string()).unwrap_or_default()
}

/// nltk PerceptronTagger._get_features — 13 features, all value 1.0.
fn get_features(i: usize, word: &str, context: &[String], prev: &str, prev2: &str) -> Vec<(String, f64)> {
    let i = i + START.len();
    let f: Vec<(String, f64)> = vec![
        ("bias".into(), 1.0),
        (format!("i suffix {}", suffix3(word)), 1.0),
        (format!("i pref1 {}", word.chars().next().map(|c| c.to_string()).unwrap_or_default()), 1.0),
        (format!("i-1 tag {}", prev), 1.0),
        (format!("i-2 tag {}", prev2), 1.0),
        (format!("i tag+i-2 tag {} {}", prev, prev2), 1.0),
        (format!("i word {}", context[i]), 1.0),
        (format!("i-1 tag+i word {} {}", prev, context[i]), 1.0),
        (format!("i-1 word {}", context[i - 1]), 1.0),
        (format!("i-1 suffix {}", suffix3(&context[i - 1])), 1.0),
        (format!("i-2 word {}", context[i - 2]), 1.0),
        (format!("i+1 word {}", context[i + 1]), 1.0),
        (format!("i+1 suffix {}", suffix3(&context[i + 1])), 1.0),
        (format!("i+2 word {}", context[i + 2]), 1.0),
    ];
    f
}

/// nltk PerceptronTagger.tag
pub fn perceptron_tag(sess: &Session, tokens: &[String]) -> Vec<(String, String)> {
    let mut prev = START[0].to_string();
    let mut prev2 = START[1].to_string();
    let mut context: Vec<String> = vec![START[0].to_string(), START[1].to_string()];
    context.extend(tokens.iter().map(|w| normalize(w)));
    context.push(END[0].to_string());
    context.push(END[1].to_string());

    let mut output = Vec::with_capacity(tokens.len());
    for (i, word) in tokens.iter().enumerate() {
        let tag = match sess.m.pos.tagdict.get(word) {
            Some(t) => t.clone(),
            None => {
                let feats = get_features(i, word, &context, &prev, &prev2);
                let refs: Vec<(&str, f64)> = feats.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                predict(sess, &refs)
            }
        };
        output.push((word.clone(), tag.clone()));
        prev2 = prev;
        prev = tag;
    }
    output
}

/// nlp.change_tag — the four patterns are single-char classes / simple sets;
/// hand-rolled to avoid regex escape dialects.
pub fn change_tag(w: &str, pos: &str) -> String {
    let chars: Vec<char> = w.chars().collect();
    const SYM: &[char] = &['#', '+', '-', '_', '=', '*', '&', '^', '%', '$', '@', '}', '{', ']', '[', '<', '>', '/'];
    // Python PUNCT class: r"[\,\;\:\.\(\)\'\"\!\?]"
    const PUNCT2: &[char] = &[
        ',', ';', ':', '.', '(', ')', '\'', '"', '!', '?',
    ];
    const NUM_TAIL: &[char] = &['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ',', '+', '-', '*', '/'];
    if chars.len() == 1 && SYM.contains(&chars[0]) {
        "SYM".to_string()
    } else if chars.len() == 1 && PUNCT2.contains(&chars[0]) {
        "PUNCT".to_string()
    } else if !chars.is_empty()
        && chars[0].is_ascii_digit()
        && chars[1..].iter().all(|c| NUM_TAIL.contains(c))
    {
        "NUM".to_string()
    } else if w == "<s/>" {
        "PUNCT".to_string()
    } else if !chars.is_empty()
        && chars
            .iter()
            .all(|c| *c == '<' || *c == '>' || c.is_ascii_alphabetic() || *c == '/')
    {
        "SYM".to_string()
    } else if !chars.is_empty() && chars.iter().all(|c| c.is_ascii_alphanumeric() || *c == '_') {
        "X".to_string()
    } else {
        pos.to_string()
    }
}

/// nlp.pos_tag — returns per-sentence tagged word lists.
pub fn pos_tag(sess: &mut Session, input: &str) -> Vec<Vec<(String, String)>> {
    let mut results = Vec::new();
    let out = sess.word_segment(input);
    for x in out.split("<s/>") {
        if x.is_empty() {
            continue;
        }
        let mut sent: Vec<String> = x.split('|').map(|s| s.to_string()).collect();
        sent.retain(|w| !w.is_empty());
        let mut tag_result = Vec::with_capacity(sent.len() + 1);
        for (w, pos) in perceptron_tag(sess, &sent) {
            let w = if w == " " { "<s/>".to_string() } else { w };
            let pos = change_tag(&w, &pos);
            tag_result.push((w, pos));
        }
        tag_result.push(("<s/>".to_string(), "PUNCT".to_string()));
        results.push(tag_result);
    }
    results
}

impl<'a> Session<'a> {
    /// Server wire format for pos_tag: sentences joined with NEWLINE, each
    /// sentence "w/TAG w/TAG ... <s/>/PUNCT".
    pub fn pos_tag_wire(&mut self, input: &str) -> String {
        let results = pos_tag(self, input);
        let mut sentences: Vec<String> = Vec::new();
        for tag_result in results {
            let parts: Vec<String> = tag_result.iter().map(|(w, p)| format!("{}/{}", w, p)).collect();
            sentences.push(parts.join(" "));
        }
        sentences.join("\n")
    }
}
