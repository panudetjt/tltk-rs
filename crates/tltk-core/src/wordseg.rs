//! Word segmentation (port of nlp.word_segment / wordseg_colloc / gen_unknown_w /
//! eng_abbr) and the EDU-side entry used by segment().

use crate::g2p::is_eng_abbr;
use crate::prelude::*;
use crate::sylseg::SpanMap;
use crate::Session;
use std::collections::HashMap;

impl<'a> Session<'a> {
    /// Port of word_segment(method="colloc").
    pub fn word_segment(&mut self, input: &str) -> String {
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
                    self.wordseg_colloc(inp)
                };
                output.push_str(&out);
                output.push(WORD_SEP);
            }
            output.push_str("<s/>");
        }
        output
    }

    /// Port of wordseg_colloc — processes only the FIRST SegSep part (Python
    /// returns inside the loop).
    pub fn wordseg_colloc(&mut self, input: &str) -> String {
        let part: Vec<&str> = input.split(SEG_SEP).collect();
        if let Some(&inx) = part.first() {
            let syl_out = self.syl_segment(inx);
            let mut syl_lst: Vec<String> = syl_out.split(SYL_SEP).map(|s| s.to_string()).collect();
            if syl_lst.last().map(|s| s.as_str()) == Some("<s/>") {
                syl_lst.pop();
            }
            let n = syl_lst.len();
            let mut chart: Vec<SpanMap<Vec<String>>> = (0..=n).map(|_| SpanMap::new()).collect();
            let mut colloc_st: HashMap<(usize, usize), f64> = HashMap::new();

            // gen_unknown_w: each syllable a word; unknown syllables chain forward
            for i in 0..n {
                chart[i].set(i + 1, vec![syl_lst[i].clone()]);
                if !self.m.thdict.contains(&syl_lst[i]) {
                    for j in (i + 1)..n {
                        if !self.m.thdict.contains(&syl_lst[j]) {
                            let w: String = syl_lst[i..=j].concat();
                            chart[i].set(j + 1, vec![w]);
                        } else {
                            break;
                        }
                    }
                }
            }
            // eng_abbr
            let mut i = 0usize;
            while i + 1 < n {
                if is_eng_abbr(&syl_lst[i]) {
                    let mut j = i + 1;
                    while j < n && is_eng_abbr(&syl_lst[j]) {
                        j += 1;
                    }
                    if j > i + 1 {
                        let w: String = syl_lst[i..j].concat();
                        chart[i].set(j, vec![w]);
                        i = j + 1;
                    } else {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
            // dictionary edges with collocation strength
            for i in 0..n {
                for j in i..=n {
                    let wrd: String = syl_lst[i..j].concat();
                    if self.m.thdict.contains(&wrd) {
                        chart[i].set(j, vec![wrd]);
                        if j > i + 1 {
                            let mut st = 0.0f64;
                            for ii in i..(j - 1) {
                                st += self.compute_colloc_mi(&syl_lst[ii], &syl_lst[ii + 1]);
                            }
                            colloc_st.insert((i, j), st);
                        } else {
                            colloc_st.insert((i, j), 0.0);
                        }
                    }
                }
            }
            if self.chart_parse(&mut chart, &mut colloc_st, n) {
                let words = chart[0].get(n).unwrap();
                words.join("|")
            } else {
                format!("<Fail>{}</Fail>", input)
            }
        } else {
            String::new()
        }
    }
}
