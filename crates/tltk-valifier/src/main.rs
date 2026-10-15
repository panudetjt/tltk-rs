//! Valifier — parity harness for tltk-core.
//!
//! Modes:
//!   valifier run <corpus.jsonl> <models-dir>     run core, print results JSONL
//!   valifier check <corpus.jsonl> <golden.jsonl> <models-dir>
//!       compare core output vs golden responses (from stichula capture or the
//!       local python oracle); exit 1 on any mismatch, printing diffs
//!   valifier selftest <models-dir>               a few known sanity checks
//!
//! corpus.jsonl rows: {"function": "...", "text": "...", "cutoff": "medium"(opt)}
//! golden.jsonl rows: {...same..., "expected": "...", "ok": true|false}

use std::io::BufRead;
use std::path::PathBuf;

#[derive(serde::Deserialize, Clone)]
struct Case {
    function: String,
    text: String,
    #[serde(default)]
    cutoff: Option<String>,
    #[serde(default)]
    expected: Option<String>,
    #[serde(default)]
    ok: Option<bool>,
}

fn load_cases(path: &str) -> Vec<Case> {
    let f = std::fs::File::open(path).expect("cannot open corpus");
    let r = std::io::BufReader::new(f);
    r.lines()
        .map(|l| {
            let l = l.unwrap();
            if l.trim().is_empty() {
                return None;
            }
            serde_json::from_str::<Case>(&l).ok()
        })
        .flatten()
        .collect()
}

fn load_models(dir: &str) -> tltk_core::Tltk {
    let models = tltk_core::model::Models::load_dir(&PathBuf::from(dir))
        .expect("cannot load models dir");
    tltk_core::Tltk::new(models)
}

fn run_case(sess: &mut tltk_core::Session, c: &Case) -> Result<String, tltk_core::ProcessError> {
    let cutoff = c.cutoff.as_deref().unwrap_or("medium");
    sess.process(&c.function, &c.text, cutoff)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("run") => {
            let cases = load_cases(&args[2]);
            let tltk = load_models(&args[3]);
            let mut sess = tltk.session();
            for c in &cases {
                let res = run_case(&mut sess, c);
                let row = serde_json::json!({
                    "function": c.function,
                    "text": c.text,
                    "cutoff": c.cutoff.clone().unwrap_or_else(|| "medium".into()),
                    "result": match &res { Ok(s) => serde_json::Value::String(s.clone()), Err(e) => serde_json::Value::String(format!("!ERROR! {}", e)) },
                });
                println!("{}", row);
            }
        }
        Some("check") => {
            let cases = load_cases(&args[2]);
            let golden: std::collections::HashMap<(String, String, String), (String, bool)> =
                load_cases(&args[3])
                    .into_iter()
                    .map(|g| {
                        (
                            (g.function.clone(), g.text.clone(), g.cutoff.clone().unwrap_or_else(|| "medium".into())),
                            (g.expected.clone().unwrap_or_default(), g.ok.unwrap_or(true)),
                        )
                    })
                    .collect();
            let tltk = load_models(&args[4]);
            let mut sess = tltk.session();
            let mut pass = 0usize;
            let mut fail = 0usize;
            let mut missing = 0usize;
            for c in &cases {
                let key = (
                    c.function.clone(),
                    c.text.clone(),
                    c.cutoff.clone().unwrap_or_else(|| "medium".into()),
                );
                let (expected, ok) = match golden.get(&key) {
                    Some(v) => v,
                    None => {
                        missing += 1;
                        continue;
                    }
                };
                if !ok {
                    // golden says the server errored — expect our error too
                    let is_err = run_case(&mut sess, c).is_err();
                    if is_err {
                        pass += 1;
                    } else {
                        fail += 1;
                        println!("MISMATCH(function={} text={:?}): expected error, got result", c.function, c.text);
                    }
                    continue;
                }
                match run_case(&mut sess, c) {
                    Ok(got) if &got == expected => pass += 1,
                    Ok(got) => {
                        fail += 1;
                        println!("MISMATCH(function={} text={:?}):", c.function, c.text);
                        println!("  expected: {:?}", expected);
                        println!("  got     : {:?}", got);
                    }
                    Err(e) => {
                        fail += 1;
                        println!("MISMATCH(function={} text={:?}): expected {:?}, got ERROR {}", c.function, c.text, expected, e);
                    }
                }
            }
            println!(
                "valifier: pass={} fail={} missing_golden={}",
                pass, fail, missing
            );
            if fail > 0 {
                std::process::exit(1);
            }
        }
        Some("selftest") => {
            let tltk = load_models(&args[2]);
            let mut sess = tltk.session();
            let checks: Vec<(&str, &str, &str)> = vec![
                ("word_segment", "แมวกินปลา", "แมว|กิน|ปลา|<s/>"),
                ("syl_segment", "แมวกินปลา", "แมว~กิน~ปลา~<s/>"),
                ("th2roman", "แมวกินปลา", "maeokinpla <s/>"),
                ("th2ipa", "แมวกินปลา", "mɛːw1.kin1.plaː1 <s/>"),
                (
                    "g2p",
                    "แมวกินปลา",
                    "แมว~กิน~ปลา<tr/>mxxw0~kin0~plaa0|<s/>",
                ),
                ("th2read", "แมวกินปลา", "แมว-กิน-ปฺลา-"),
                (
                    "spell_variants",
                    "แมวกินปลา",
                    "['แมวกินปลา', 'แมวกิลปลา', 'แมวกิรปลา']",
                ),
                ("spoonerism", "แมวกินปลา", "['มา-กิน-แปฺลว']"),
                ("pos_tag", "แมวกินปลา", "แมว/NOUN กิน/VERB ปลา/NOUN <s/>/PUNCT"),
            ];
            let mut bad = 0;
            for (f, t, want) in checks {
                match sess.process(f, t, "medium") {
                    Ok(got) if got == want => println!("ok   {} {:?}", f, t),
                    Ok(got) => {
                        bad += 1;
                        println!("FAIL {} {:?}\n  want: {:?}\n  got : {:?}", f, t, want, got);
                    }
                    Err(e) => {
                        bad += 1;
                        println!("FAIL {} {:?} -> error {}", f, t, e);
                    }
                }
            }
            if bad > 0 {
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("usage: valifier run|check|selftest ...");
            std::process::exit(2);
        }
    }
}
