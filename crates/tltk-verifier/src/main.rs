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
//!   verifier mem <corpus.jsonl> <models-dir> [--json]
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

/// RFC3339 UTC from unix seconds (no chrono dep; Howard Hinnant's civil algo).
fn rfc3339_utc(secs: u64) -> String {
    let days = secs / 86400;
    let rem = secs % 86400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mth = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mth <= 2 { y + 1 } else { y };
    format!("{y:04}-{mth:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn rfc3339_utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    rfc3339_utc(secs)
}

fn first_line(path: &str) -> Option<String> {
    let s = std::fs::read_to_string(path).ok()?;
    s.lines().next().map(|l| l.trim().to_string())
}

/// Short HEAD sha + dirty-file count, so every report names the exact source
/// state it measured (None outside a git repo / when git is unavailable).
fn git_state() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if sha.is_empty() {
        return None;
    }
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().count())
        .unwrap_or(0);
    Some(if dirty > 0 {
        format!("{sha}+{dirty} dirty files")
    } else {
        sha
    })
}

/// System runtime context recorded alongside every bench/mem report, so the
/// numbers carry their own measurement conditions (best-effort: fields degrade
/// to null off-Linux rather than failing the run).
fn environment_block() -> serde_json::Value {
    let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').next_back())
                .map(|m| m.trim().to_string())
        });
    let mem_total_kb = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("MemTotal:")).and_then(|l| {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<u64>().ok())
            })
        });
    let load_avg = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| {
            let vals: Vec<f64> = s
                .split_whitespace()
                .take(3)
                .filter_map(|v| v.parse().ok())
                .collect();
            (vals.len() == 3).then_some(vals)
        });
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);

    serde_json::json!({
        "timestamp_utc": rfc3339_utc_now(),
        "host": first_line("/proc/sys/kernel/hostname"),
        "os": std::env::consts::OS,
        "kernel": first_line("/proc/sys/kernel/osrelease"),
        "cpu_model": cpu_model,
        "cpu_cores_available": cores,
        "mem_total_kb": mem_total_kb,
        "load_avg_1m_5m_15m": load_avg,
        "git": git_state(),
        "rustc": env!("VERIFIER_RUSTC_VERSION"),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
    })
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
    println!("  verifier mem <corpus.jsonl> <models-dir> [--json]\n");
    println!("Bench options:");
    println!("  --warmup W    Number of warmup passes before timing (default: 3)");
    println!("  --repeat N    Number of timed passes (default: 10)");
    println!("  --json        Output benchmark results as JSON to stdout; human table to stderr");
    println!("  --long        Include synthetic long input benchmarks (~200, ~500, ~800 chars)");
    println!("  --mem         Measure memory usage via /proc/self/status at each phase");
    println!("  mem --json    Same, JSON (includes the recorded environment block)\n");
    println!("Every bench/mem run prints an environment block (timestamp, host, kernel,");
    println!("CPU, cores, RAM, load average, rustc, profile) so recorded numbers carry");
    println!("their measurement conditions.");
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
        ("textana", "แมวกินปลา\nหมูกินข้าว", "{\"CrfCNL\":0.0,\"CrfCTL\":1.0,\"CrfCVL\":1,\"CrfCWL\":1.0,\"DesChC\":20,\"DesClusterC\":1,\"DesCorrFinalC\":4,\"DesCorrToneC\":5,\"DesDoubVowelC\":0,\"DesEduC\":0,\"DesEduL\":4.5,\"DesEduLd\":0.7071067811865476,\"DesInCorrFinalC\":0,\"DesInCorrToneC\":1,\"DesKaranC\":0,\"DesLeadC\":0,\"DesNElC\":0,\"DesNEoC\":0,\"DesNEpC\":0,\"DesPC\":0,\"DesPL\":9,\"DesSpC\":0,\"DesSymbC\":1,\"DesTNCt2C\":2,\"DesTNCt3C\":1,\"DesTNCt4C\":1,\"DesTTC1\":6,\"DesTotT\":5,\"DesTotW\":6,\"DesWrdL\":1,\"DesWrdLd\":0.0,\"LdvTTR\":83.33333333333333,\"SynDepth\":0.0,\"WrdADJ\":0.0,\"WrdADP\":0.0,\"WrdADV\":0.0,\"WrdAUX\":0.0,\"WrdArf\":1.2,\"WrdBasicM\":1000.0,\"WrdBasicT\":833.3333333333334,\"WrdC\":1000.0,\"WrdCCONJ\":0.0,\"WrdCF\":0.0,\"WrdCluster\":166.66666666666666,\"WrdCorrFinal\":666.6666666666666,\"WrdCorrTone\":833.3333333333334,\"WrdDET\":0.0,\"WrdDoubVowel\":0.0,\"WrdF\":0.0,\"WrdFrmComp\":0.0,\"WrdFrmSing\":1000.0,\"WrdFrmTran\":0.0,\"WrdINTJ\":0.0,\"WrdInCorrFinal\":0.0,\"WrdInCorrTone\":166.66666666666666,\"WrdKaran\":0.0,\"WrdLead\":0.0,\"WrdNEl\":0.0,\"WrdNEo\":0.0,\"WrdNEp\":0.0,\"WrdNOUN\":666.6666666666666,\"WrdNUM\":0.0,\"WrdNeg\":0.0,\"WrdPART\":0.0,\"WrdPRON\":0.0,\"WrdPROPN\":0.0,\"WrdPUNCT\":166.66666666666666,\"WrdSCONJ\":0.0,\"WrdSYM\":0.0,\"WrdSemOpaq\":0.0,\"WrdSemSemi\":0.0,\"WrdSemSimp\":1000.0,\"WrdSemTran\":0.0,\"WrdTNCt1\":0.0,\"WrdTNCt2\":333.3333333333333,\"WrdTNCt3\":166.66666666666666,\"WrdTNCt4\":166.66666666666666,\"WrdTTC1\":1000.0,\"WrdTTC2\":0.0,\"WrdTTC3\":0.0,\"WrdTTC4\":0.0,\"WrdTfidf\":15396.34560397768,\"WrdTncDisp\":0.17838474217094413,\"WrdTtcDisp\":0.5830028328611897,\"WrdVERB\":333.3333333333333,\"deprel\":{},\"wrd\":{\"กิน\":2,\"ข้าว\":1,\"ปลา\":1,\"หมู\":1,\"แมว\":1},\"wrd_arf\":{\"กิน\":2.0,\"ข้าว\":1.0,\"ปลา\":1.0,\"หมู\":1.0,\"แมว\":1.0}}"),
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

    let environment = environment_block();
    {
        let env_line = format!(
            "# env: {}",
            serde_json::to_string(&environment).unwrap_or_default()
        );
        if json {
            eprintln!("{env_line}");
        } else {
            println!("{env_line}");
        }
    }

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
            "environment": environment,
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
    let mut json = false;
    let mut positional = Vec::new();
    for a in &args[2..] {
        match a.as_str() {
            "--help" | "-h" => {
                println!("usage: verifier mem <corpus.jsonl> <models-dir> [--json]");
                return;
            }
            "--json" => json = true,
            other => positional.push(other.to_string()),
        }
    }
    if positional.len() < 2 {
        eprintln!("usage: verifier mem <corpus.jsonl> <models-dir> [--json]");
        std::process::exit(2);
    }
    let environment = environment_block();
    let env_line = format!(
        "# env: {}",
        serde_json::to_string(&environment).unwrap_or_default()
    );
    let cases = load_cases(&positional[0]);
    let tltk = load_models(&positional[1]);
    let mem_after_load = read_proc_status();

    let mut sess = tltk.session();
    for c in &cases {
        let res = run_case(&mut sess, c);
        let _ = std::hint::black_box(res);
    }
    let mem_after_run = read_proc_status();

    let phase = (mem_after_load, mem_after_run);
    if json {
        let (after_load_kb, after_run_kb, vmhwm_kb) = match phase {
            (Some(load), Some(run)) => (
                serde_json::json!(load.vm_rss_kb),
                serde_json::json!(run.vm_rss_kb),
                serde_json::json!(run.vm_hwm_kb.max(load.vm_hwm_kb)),
            ),
            _ => (
                serde_json::Value::Null,
                serde_json::Value::Null,
                serde_json::Value::Null,
            ),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "environment": environment,
                "after_load_kb": after_load_kb,
                "after_run_kb": after_run_kb,
                "vmhwm_kb": vmhwm_kb,
            }))
            .unwrap()
        );
        return;
    }

    println!("{env_line}");
    match phase {
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
    fn test_rfc3339_utc() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(1_700_000_000), "2023-11-14T22:13:20Z");
        // 2026-10-16T12:00:00Z (bench-cut era)
        assert_eq!(rfc3339_utc(1_792_152_000), "2026-10-16T12:00:00Z");
    }

    #[test]
    fn test_proc_status_missing_fields() {
        let fake_proc = "Name:\tverifier\nVmRSS:\t  258304 kB\n";
        assert_eq!(parse_proc_status(fake_proc.as_bytes()), None);
    }
}
