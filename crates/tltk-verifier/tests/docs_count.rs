//! Docs-lint: hardcoded function counts in prose must match the wire.
//!
//! "41 wire functions" drifted to 38 within a day during the 2026-10-16 port
//! wave — the count lives in the worker's functions array and the docs keep
//! restating it. This test makes the corpus decide: every "<N> wire functions"
//! / "<N> functions:" mention in the top-level docs must equal the array
//! length. A doc that drops counts entirely passes (no mention, no drift).

fn wire_function_count() -> usize {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tltk-worker/src/lib.rs");
    let src = std::fs::read_to_string(path).expect("read tltk-worker/src/lib.rs");
    let arr = src
        .split("\"functions\": [")
        .nth(1)
        .expect("functions array not found in tltk-worker/src/lib.rs");
    let arr = &arr[..arr.find(']').expect("functions array not terminated")];
    let count = arr.matches('"').count() / 2;
    assert!(count > 0, "functions array parsed as empty");
    count
}

/// Numbers immediately preceding `phrase` in `text` (e.g. "41 wire functions").
fn counts_before(text: &str, phrase: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, _) in text.match_indices(phrase) {
        let bytes = text.as_bytes();
        let mut end = i;
        while end > 0 && bytes[end - 1].is_ascii_digit() {
            end -= 1;
        }
        if end < i {
            out.push(text[end..i].parse().expect("digits parse"));
        }
    }
    out
}

#[test]
fn docs_function_counts_match_the_wire() {
    let wire = wire_function_count();
    let checks = [
        ("README.md", &[" wire functions", " functions:"][..]),
        ("AGENTS.md", &[" wire functions", " functions:"][..]),
    ];
    for (doc, phrases) in checks {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../");
        let text = std::fs::read_to_string(format!("{path}{doc}")).expect("read doc");
        for phrase in phrases {
            for n in counts_before(&text, phrase) {
                assert_eq!(
                    n, wire,
                    "{doc}: hardcoded count {n} ({phrase}) drifted from the {wire} functions in crates/tltk-worker/src/lib.rs — update the doc or the array"
                );
            }
        }
    }
    // the check itself must stay live: at least one count is expected in README
    let readme =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md")).unwrap();
    let found = [" wire functions", " functions:"]
        .iter()
        .any(|p| !counts_before(&readme, p).is_empty());
    assert!(found, "no function count found in README — either restore one or retire this test");
}
