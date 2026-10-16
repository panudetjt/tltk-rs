//! Valifier — parity harness and benchmark tool for tltk-core.
//!
//! Modes:
//!   verifier run <corpus.jsonl> <models-dir>     run core, print results JSONL
//!   verifier check <corpus.jsonl> <golden.jsonl> <models-dir>
//!       compare core output vs golden responses (from stichula capture or the
//!       local python oracle); exit 1 on any mismatch, printing diffs
//!   verifier selftest <models-dir>               a few known sanity checks
//!   verifier bench <corpus.jsonl> <models-dir> [--warmup W] [--repeat N] [--json] [--long] [--mem]
//!       measure per-function and optional long-input pass latency and memory
//!   verifier mem <corpus.jsonl> <models-dir>
//!       measure memory usage (after load, after run, peak)
//!
//! corpus.jsonl rows: {"function": "...", "text": "...", "cutoff": "medium"(opt)}
//! golden.jsonl rows: {...same..., "expected": "...", "ok": true|false}

use std::collections::HashMap;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
struct ProcMemory {
    vm_rss_kb: u64,
    vm_hwm_kb: u64,
}

fn parse_proc_status<R: BufRead>(reader: R) -> Option<ProcMemory> {
    let mut rss = None;
    let mut hwm = None;
    for line in reader.lines() {
        let line = line.ok()?;
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(num_str) = rest.split_whitespace().next() {
                rss = num_str.parse::<u64>().ok();
            }
        } else if let Some(rest) = line.strip_prefix("VmHWM:")
            && let Some(num_str) = rest.split_whitespace().next() {
                hwm = num_str.parse::<u64>().ok();
            }
    }
    match (rss, hwm) {
        (Some(vm_rss_kb), Some(vm_hwm_kb)) => Some(ProcMemory {
            vm_rss_kb,
            vm_hwm_kb,
        }),
        _ => None,
    }
}

fn read_proc_status_path(path: &str) -> Option<ProcMemory> {
    let file = std::fs::File::open(path).ok()?;
    let reader = std::io::BufReader::new(file);
    parse_proc_status(reader)
}

fn read_proc_status() -> Option<ProcMemory> {
    let path = std::env::var("TLTK_PROC_STATUS")
        .or_else(|_| std::env::var("PROC_STATUS_PATH"))
        .unwrap_or_else(|_| "/proc/self/status".to_string());
    read_proc_status_path(&path)
}

fn load_cases(path: &str) -> Vec<Case> {
    let f = std::fs::File::open(path).expect("cannot open corpus");
    let r = std::io::BufReader::new(f);
    r.lines()
        .filter_map(|l| {
            let l = l.unwrap();
            if l.trim().is_empty() {
                return None;
            }
            serde_json::from_str::<Case>(&l).ok()
        })
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

fn compute_median(mut values: Vec<u64>) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2
    }
}

fn build_synthetic_input(distinct: &[&str], target_chars: usize) -> String {
    if distinct.is_empty() {
        return String::new();
    }
    let mut pieces = Vec::new();
    let mut cur_len = 0;
    let mut idx = 0;
    while cur_len < target_chars {
        let item = distinct[idx % distinct.len()];
        pieces.push(item);
        if cur_len > 0 {
            cur_len += 1;
        }
        cur_len += item.chars().count();
        idx += 1;
    }
    pieces.join(" ")
}

fn print_help() {
    println!("Valifier — parity harness and benchmark tool for tltk-core.\n");
    println!("Usage:");
    println!("  verifier run <corpus.jsonl> <models-dir>");
    println!("  verifier check <corpus.jsonl> <golden.jsonl> <models-dir>");
    println!("  verifier selftest <models-dir>");
    println!("  verifier bench <corpus.jsonl> <models-dir> [--warmup W] [--repeat N] [--json] [--long] [--mem]");
    println!("  verifier mem <corpus.jsonl> <models-dir>\n");
    println!("Bench options:");
    println!("  --warmup W    Number of warmup passes before timing (default: 3)");
    println!("  --repeat N    Number of timed passes (default: 10)");
    println!("  --json        Output benchmark results as JSON to stdout; human table to stderr");
    println!("  --long        Include synthetic long input benchmarks (~200, ~500, ~800 chars)");
    println!("  --mem         Measure memory usage via /proc/self/status at each phase\n");
    println!("Note on fresh Session per pass:");
    println!("  TLTK sessions accumulate colloc-bump state; fresh-per-pass keeps runs comparable.");
}

fn cmd_run(args: &[String]) {
    if args.len() < 4 {
        eprintln!("usage: verifier run <corpus.jsonl> <models-dir>");
        std::process::exit(2);
    }
    let cases = load_cases(&args[2]);
    let tltk = load_models(&args[3]);
    let mut sess = tltk.session();
    for c in &cases {
        let res = run_case(&mut sess, c);
        let row = serde_json::json!({
            "function": c.function,
            "text": c.text,
            "cutoff": c.cutoff.clone().unwrap_or_else(|| "medium".into()),
            "result": match &res {
                Ok(s) => serde_json::Value::String(s.clone()),
                Err(e) => serde_json::Value::String(format!("!ERROR! {}", e)),
            },
        });
        println!("{}", row);
    }
}

fn cmd_check(args: &[String]) {
    if args.len() < 5 {
        eprintln!("usage: verifier check <corpus.jsonl> <golden.jsonl> <models-dir>");
        std::process::exit(2);
    }
    let cases = load_cases(&args[2]);
    let golden: HashMap<(String, String, String), (String, bool)> =
        load_cases(&args[3])
            .into_iter()
            .map(|g| {
                (
                    (
                        g.function.clone(),
                        g.text.clone(),
                        g.cutoff.unwrap_or_else(|| "medium".into()),
                    ),
                    (g.expected.unwrap_or_default(), g.ok.unwrap_or(true)),
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
            let is_err = run_case(&mut sess, c).is_err();
            if is_err {
                pass += 1;
            } else {
                fail += 1;
                println!(
                    "MISMATCH(function={} text={:?}): expected error, got result",
                    c.function, c.text
                );
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
                println!(
                    "MISMATCH(function={} text={:?}): expected {:?}, got ERROR {}",
                    c.function, c.text, expected, e
                );
            }
        }
    }
    println!(
        "verifier: pass={} fail={} missing_golden={}",
        pass, fail, missing
    );
    if fail > 0 {
        std::process::exit(1);
    }
}

fn cmd_selftest(args: &[String]) {
    if args.len() < 3 {
        eprintln!("usage: verifier selftest <models-dir>");
        std::process::exit(2);
    }
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

fn cmd_bench(args: &[String]) {
    let rest = &args[2..];
    let mut positional = Vec::new();
    let mut warmup = 3usize;
    let mut repeat = 10usize;
    let mut json = false;
    let mut long = false;
    let mut mem = false;

    let mut i = 0;
    while i < rest.len() {
        let arg = &rest[i];
        if arg == "--help" || arg == "-h" {
            print_help();
            return;
        } else if arg == "--json" {
            json = true;
        } else if arg == "--long" {
            long = true;
        } else if arg == "--mem" {
            mem = true;
        } else if arg == "--warmup" {
            i += 1;
            if i < rest.len() {
                warmup = rest[i].parse().unwrap_or(3);
            }
        } else if let Some(val) = arg.strip_prefix("--warmup=") {
            warmup = val.parse().unwrap_or(3);
        } else if arg == "--repeat" {
            i += 1;
            if i < rest.len() {
                repeat = rest[i].parse().unwrap_or(10);
            }
        } else if let Some(val) = arg.strip_prefix("--repeat=") {
            repeat = val.parse().unwrap_or(10);
        } else if arg.starts_with('-') {
            eprintln!("unknown option: {}", arg);
            std::process::exit(2);
        } else {
            positional.push(arg.clone());
        }
        i += 1;
    }

    if positional.len() < 2 {
        eprintln!(
            "usage: verifier bench <corpus.jsonl> <models-dir> [--warmup W] [--repeat N] [--json] [--long] [--mem]"
        );
        std::process::exit(2);
    }
    let corpus_path = &positional[0];
    let models_dir = &positional[1];

    let cases = load_cases(corpus_path);
    let tltk = load_models(models_dir);
    let mem_after_load = if mem { read_proc_status() } else { None };
    let mut fn_order: Vec<String> = Vec::new();
    let mut fn_groups: HashMap<String, Vec<&Case>> = HashMap::new();
    for c in &cases {
        if !fn_groups.contains_key(&c.function) {
            fn_order.push(c.function.clone());
        }
        fn_groups.entry(c.function.clone()).or_default().push(c);
    }

    let mut fn_results: Vec<(String, usize, u64)> = Vec::new();

    if warmup > 0 {
        for func_name in &fn_order {
            let cases_for_fn = &fn_groups[func_name];
            for _ in 0..warmup {
                let mut sess = tltk.session();
                for c in cases_for_fn {
                    let res = run_case(&mut sess, c);
                    let _ = std::hint::black_box(res);
                }
            }
        }
    }
    let mem_after_warmup = if mem { read_proc_status() } else { None };

    for func_name in &fn_order {
        let cases_for_fn = &fn_groups[func_name];

        let mut pass_times = Vec::with_capacity(repeat);
        for _ in 0..repeat {
            let mut sess = tltk.session();
            let start = Instant::now();
            for c in cases_for_fn {
                let res = run_case(&mut sess, c);
                let _ = std::hint::black_box(res);
            }
            let elapsed = start.elapsed();
            pass_times.push(elapsed.as_micros() as u64);
        }

        let median_us = compute_median(pass_times);
        fn_results.push((func_name.clone(), cases_for_fn.len(), median_us));
    }
    let mem_after_pass = if mem { read_proc_status() } else { None };

    let total_cases: usize = fn_results.iter().map(|(_, c, _)| *c).sum();
    let total_median: u64 = fn_results.iter().map(|(_, _, m)| *m).sum();

    let long_targets = [
        "syl_segment",
        "g2p",
        "th2ipa",
        "th2roman",
        "word_segment",
        "segment",
        "pos_tag",
        "ner_tag",
    ];
    let len_buckets = [(200, "200"), (500, "500"), (800, "800")];

    let mut long_results: Vec<(String, u64)> = Vec::new();

    if long {
        for target_fn in &long_targets {
            let mut distinct = Vec::new();
            if let Some(cases_for_fn) = fn_groups.get(*target_fn) {
                for c in cases_for_fn {
                    if !distinct.contains(&c.text.as_str()) {
                        distinct.push(c.text.as_str());
                    }
                }
            }
            for &(target_chars, bucket_label) in &len_buckets {
                let label = format!("{}/{}", target_fn, bucket_label);
                let synth_text = build_synthetic_input(&distinct, target_chars);
                let synth_case = Case {
                    function: target_fn.to_string(),
                    text: synth_text,
                    cutoff: None,
                    expected: None,
                    ok: None,
                };

                for _ in 0..warmup {
                    let mut sess = tltk.session();
                    let res = run_case(&mut sess, &synth_case);
                    let _ = std::hint::black_box(res);
                }

                let mut pass_times = Vec::with_capacity(repeat);
                for _ in 0..repeat {
                    let mut sess = tltk.session();
                    let start = Instant::now();
                    let res = run_case(&mut sess, &synth_case);
                    let _ = std::hint::black_box(res);
                    let elapsed = start.elapsed();
                    pass_times.push(elapsed.as_micros() as u64);
                }

                let median_us = compute_median(pass_times);
                long_results.push((label, median_us));
            }
        }
    }
    let mem_after_long = if mem && long { read_proc_status() } else { None };

    let emit_human = |line: &str| {
        if json {
            eprintln!("{}", line);
        } else {
            println!("{}", line);
        }
    };

    for (name, count, median_us) in &fn_results {
        emit_human(&format!(
            "{:<19}cases={:<2}  median_us_per_pass={}",
            name, count, median_us
        ));
    }
    emit_human(&format!(
        "{:<19}cases={:<2}  median_us_per_pass={}",
        "total", total_cases, total_median
    ));

    if long {
        emit_human("");
        emit_human("long-inputs:");
        for (label, median_us) in &long_results {
            emit_human(&format!(
                "{:<19}cases={:<2}  median_us_per_pass={}",
                label, 1, median_us
            ));
        }
    }
    if mem {
        emit_human("");
        if let (Some(after_load), Some(after_warmup), Some(after_pass)) =
            (mem_after_load, mem_after_warmup, mem_after_pass)
        {
            emit_human("memory:");
            emit_human(&format!(
                "  {:<16}rss={:<8} KB  vmhwm={:<8} KB",
                "after_load:", after_load.vm_rss_kb, after_load.vm_hwm_kb
            ));
            emit_human(&format!(
                "  {:<16}rss={:<8} KB  vmhwm={:<8} KB",
                "after_warmup:", after_warmup.vm_rss_kb, after_warmup.vm_hwm_kb
            ));
            emit_human(&format!(
                "  {:<16}rss={:<8} KB  vmhwm={:<8} KB",
                "after_pass:", after_pass.vm_rss_kb, after_pass.vm_hwm_kb
            ));
            if long
                && let Some(after_long) = mem_after_long {
                    emit_human(&format!(
                        "  {:<16}rss={:<8} KB  vmhwm={:<8} KB",
                        "after_long:", after_long.vm_rss_kb, after_long.vm_hwm_kb
                    ));
                }
        } else {
            emit_human("memory: unavailable (no /proc)");
        }
    }

    if json {
        let mut per_func_json = serde_json::Map::new();
        for (name, count, median_us) in &fn_results {
            per_func_json.insert(
                name.clone(),
                serde_json::json!({
                    "cases": count,
                    "median_us_per_pass": median_us,
                }),
            );
        }

        let mut long_inputs_json = serde_json::Map::new();
        if long {
            for (label, median_us) in &long_results {
                long_inputs_json.insert(
                    label.clone(),
                    serde_json::json!({
                        "cases": 1,
                        "median_us_per_pass": median_us,
                    }),
                );
            }
        }

        let mut report = serde_json::json!({
            "per_function": per_func_json,
            "long_inputs": long_inputs_json,
            "total_median_us_per_pass": total_median,
            "repeat": repeat,
            "warmup": warmup,
        });

        if mem {
            let peak_hwm = mem_after_long
                .or(mem_after_pass)
                .or(mem_after_load)
                .map(|m| m.vm_hwm_kb);
            let mem_val = match (mem_after_load, mem_after_pass) {
                (Some(load), Some(pass)) => {
                    serde_json::json!({
                        "after_load_kb": load.vm_rss_kb,
                        "after_pass_kb": pass.vm_rss_kb,
                        "vmhwm_kb": peak_hwm.unwrap_or(pass.vm_hwm_kb),
                    })
                }
                _ => {
                    serde_json::json!({
                        "after_load_kb": serde_json::Value::Null,
                        "after_pass_kb": serde_json::Value::Null,
                        "vmhwm_kb": serde_json::Value::Null,
                    })
                }
            };
            report.as_object_mut().unwrap().insert("memory".to_string(), mem_val);
        }

        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}

fn cmd_mem(args: &[String]) {
    if args.len() >= 3 && (args[2] == "--help" || args[2] == "-h") {
        println!("usage: verifier mem <corpus.jsonl> <models-dir>");
        return;
    }
    if args.len() < 4 {
        eprintln!("usage: verifier mem <corpus.jsonl> <models-dir>");
        std::process::exit(2);
    }
    let cases = load_cases(&args[2]);
    let tltk = load_models(&args[3]);
    let mem_after_load = read_proc_status();

    let mut sess = tltk.session();
    for c in &cases {
        let res = run_case(&mut sess, c);
        let _ = std::hint::black_box(res);
    }
    let mem_after_run = read_proc_status();

    match (mem_after_load, mem_after_run) {
        (Some(load), Some(run)) => {
            println!(
                "mem: after_load_kb={} after_run_kb={} vmhwm_kb={}",
                load.vm_rss_kb,
                run.vm_rss_kb,
                run.vm_hwm_kb.max(load.vm_hwm_kb),
            );
        }
        _ => {
            println!("memory: unavailable (no /proc)");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("run") => cmd_run(&args),
        Some("check") => cmd_check(&args),
        Some("selftest") => cmd_selftest(&args),
        Some("bench") => cmd_bench(&args),
        Some("mem") => cmd_mem(&args),
        Some("--help") | Some("-h") => print_help(),
        _ => {
            eprintln!("usage: verifier run|check|selftest|bench|mem ...");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proc_status_fallback_nonexistent() {
        assert_eq!(read_proc_status_path("/nonexistent/proc/path"), None);
    }

    #[test]
    fn test_proc_status_parsing() {
        let fake_proc = "Name:\tverifier\nVmHWM:\t  270140 kB\nVmRSS:\t  258304 kB\nThreads:\t1\n";
        let parsed = parse_proc_status(fake_proc.as_bytes()).unwrap();
        assert_eq!(parsed.vm_rss_kb, 258304);
        assert_eq!(parsed.vm_hwm_kb, 270140);
    }

    #[test]
    fn test_proc_status_missing_fields() {
        let fake_proc = "Name:\tverifier\nVmRSS:\t  258304 kB\n";
        assert_eq!(parse_proc_status(fake_proc.as_bytes()), None);
    }
}
