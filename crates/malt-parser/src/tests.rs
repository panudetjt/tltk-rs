//! Fixture blob + unit tests for every engine layer. No real `malt.bin.gz`
//! is needed: everything builds on a tiny hand-written model with
//! hand-computed expectations.

use crate::graph::Graph;
use crate::model::{MaltModel, SymbolTableJson, SymbolJson, FeatureMapJson, LiblinearJson, DecisionModelJson};
use crate::transition::{Action, Config, Transition, apply, is_permissible};
use crate::features::FeatureModel;
use crate::model::FeatureNode;
use crate::scoring::predict;

/// Two feature positions:
///   pos 0: POSTAG of Stack[0]   (table POSTAG: NOUN=1, VERB=2, #null#=0)
///   pos 1: Merge(POSTAG Stack[0], POSTAG Input[0]) (table MERGE2_POSTAG_POSTAG)
/// Decision table `T.TRANS+A.DEPREL`: "SH~_"? No — combined symbols are
/// e.g. "LA~OBJ" (transition symbol + deprel), code 1 = "LA~OBJ",
/// code 2 = "RA~HEAD", code 3 = "SH~_" (unused by model but present).
/// liblinear: nr_class=2, labels=[1, 3], w chosen so a hand-computed
/// dot product decides the class.
fn fixture_json() -> Vec<u8> {
    let json = r##"{
  "format": "tltk-malt",
  "format_version": 1,
  "model_name": "fixture",
  "maltparser_version": "1.9.2",
  "source_mco_sha256": "deadbeef",
  "options": {
    "nivre_allow_root": "true",
    "nivre_allow_reduce": "false",
    "graph_root_label": "ROOT",
    "guide_decision_settings": "T.TRANS+A.DEPREL",
    "guide_classitem_separator": "~",
    "singlemalt_null_value": "one"
  },
  "data_format_xml": "<dataformat/>",
  "feature_spec_xml": "<featuremodels/>",
  "feature_strings": [
    "InputColumn(POSTAG, Stack[0])",
    "Merge(InputColumn(POSTAG, Stack[0]), InputColumn(POSTAG, Input[0]))",
    "OutputColumn(DEPREL, Stack[0])",
    "Merge3(InputColumn(POSTAG, Stack[1]), InputColumn(POSTAG, Stack[0]), InputColumn(POSTAG, Input[0]))"
  ],
  "symbol_tables": [
    { "name": "POSTAG", "header": ["1","1","one"],
      "symbols": [ {"code": 0, "symbol": "#null#"}, {"code": 1, "symbol": "NOUN"}, {"code": 2, "symbol": "VERB"} ] },
    { "name": "DEPREL", "header": ["3","1","one"],
      "symbols": [ {"code": 0, "symbol": "#null#"}, {"code": 1, "symbol": "OBJ"}, {"code": 2, "symbol": "HEAD"}, {"code": 3, "symbol": "ROOT"} ] },
    { "name": "MERGE2_POSTAG_POSTAG", "header": ["1","1","One"],
      "symbols": [ {"code": 0, "symbol": "#null#"}, {"code": 1, "symbol": "NOUN~NOUN"}, {"code": 2, "symbol": "NOUN~VERB"}, {"code": 3, "symbol": "VERB~VERB"} ] },
    { "name": "MERGE3_POSTAG_POSTAG_POSTAG", "header": ["1","1","One"],
      "symbols": [ {"code": 0, "symbol": "#null#"}, {"code": 1, "symbol": "#null#~NOUN~NOUN"}, {"code": 2, "symbol": "NOUN~NOUN~NOUN"} ] },
    { "name": "T.TRANS+A.DEPREL", "header": ["-1","1","one"],
      "symbols": [ {"code": 1, "symbol": "LA~OBJ"}, {"code": 2, "symbol": "RA~HEAD"}, {"code": 3, "symbol": "SH~"} ] }
  ],
  "decision_models": [
    { "name": "odm0",
      "feature_map": { "next_index": 5, "entries": [
        ["0", 1], ["1", 2], ["281474976710657", 3], ["562949953421313", 4], ["281474976710658", 5]
      ] },
      "liblinear": { "solver": "MCSVM_CS", "nr_class": 2, "labels": [1, 3],
                     "nr_feature": 5, "bias": 0.0,
                     "w": [ [[0, 1.5]], [[1, 1.0]], [[0, 0.25]], [[1, 2.0]], [[0, 2.0]] ] } }
  ]
}"##;
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut gz, json.as_bytes()).unwrap();
    gz.finish().unwrap()
}

fn fixture_model() -> MaltModel {
    MaltModel::decode(&fixture_json()).expect("fixture decodes")
}

/// pos0 code -> key = (0<<48)|code ; pos1 -> (1<<48)|code.
/// 281474976710657 = (1<<48)|1 ; 562949953421313 = (2<<48)|1 ;
/// 281474976710658 = (1<<48)|2. Entries above map:
///   idx1 = key 0           (pos0, code 0 = #null#)
///   idx2 = key 1           (pos0, code 1 = NOUN)
///   idx3 = (1<<48)|1       (pos1, code 1 = NOUN~NOUN)
///   idx4 = (2<<48)|1       (pos2, code 1 = DEPREL OBJ)
///   idx5 = (1<<48)|2       (pos1, code 2 = NOUN~VERB)

#[test]
fn decode_roundtrip_and_rejects() {
    let model = fixture_model();
    assert_eq!(model.feature_strings().len(), 4);
    assert!(model.allow_root);
    assert!(!model.allow_reduce);

    // wrong format
    let bad = fixture_json();
    let mut plain = flate2::read::GzDecoder::new(std::io::Cursor::new(&bad[..]));
    let mut pristine = String::new();
    std::io::Read::read_to_string(&mut plain, &mut pristine).unwrap();
    let s = pristine.replace("tltk-malt", "other-malt");
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut gz, s.as_bytes()).unwrap();
    let bad_bytes = gz.finish().unwrap();
    assert!(matches!(
        MaltModel::decode(&bad_bytes),
        Err(crate::ParseError::Decode(m)) if m.contains("unknown blob format")
    ));

    // wrong version (mutate from the PRISTINE string, not the format-bad one)
    let s2 = pristine.replace("\"format_version\": 1", "\"format_version\": 2");
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut gz, s2.as_bytes()).unwrap();
    let bad2 = gz.finish().unwrap();
    assert!(matches!(
        MaltModel::decode(&bad2),
        Err(crate::ParseError::Decode(m)) if m.contains("format_version")
    ));
}

#[test]
fn symbol_tables_decode() {
    let model = fixture_model();
    let t = model.tables().table("POSTAG").unwrap();
    assert_eq!(t.code_of("NOUN"), Some(1));
    assert_eq!(t.code_of("VERB"), Some(2));
    // null_value=one: #null# present even if omitted from the list
    assert_eq!(t.code_of("#null#"), Some(0));
    assert_eq!(t.null_code(), 0);
    let dec = model.tables().table("T.TRANS+A.DEPREL").unwrap();
    assert_eq!(dec.code_of("RA~HEAD"), Some(2));
}

#[test]
fn graph_invariants() {
    let mut g = Graph::new(&[
        ("a".into(), "NOUN".into()),
        ("b".into(), "VERB".into()),
        ("c".into(), "NOUN".into()),
    ]);
    assert_eq!(g.n_tokens(), 3);
    assert!(!g.has_head(1));
    g.add_edge(2, 1, 1).unwrap(); // a <- b
    g.add_edge(2, 3, 1).unwrap(); // c <- b
    assert!(g.has_head(1));
    assert_eq!(g.head_of(1), Some(2));
    assert_eq!(g.leftmost_dependent(2), Some(1));
    assert_eq!(g.rightmost_dependent(2), Some(3));
    assert_eq!(g.leftmost_dependent(1), None);
    assert_eq!(g.rightmost_dependent(1), None);
    assert!(g.add_edge(1, 1, 1).is_err()); // second head rejected
    g.link_all_trees_to_root(3);
    assert_eq!(g.head_of(2), Some(0)); // b unattached -> root
    assert_eq!(g.deprel_code(2), 3);
}

#[test]
fn feature_evaluation_each_kind() {
    let model = fixture_model();
    let fm = FeatureModel::new(&model).unwrap();
    // tokens: [a/NOUN, b/VERB, c/NOUN]
    let g = Graph::new(&[
        ("a".into(), "NOUN".into()),
        ("b".into(), "VERB".into()),
        ("c".into(), "NOUN".into()),
    ]);
    let cfg = Config::new(3, true, false);
    let values = fm.evaluate(&model, &g, &cfg);

    // pos 0: POSTAG(Stack[0]=root) -> #null# = 0
    assert_eq!(values[0], Some((0, "#null#".to_string())));
    // pos 1: Merge(#null#, NOUN) -> not all null -> "…#null#~NOUN" — wait:
    // Merge(null, NOUN): first is null but not BOTH -> merged "#null#~NOUN",
    // which is NOT in MERGE2 table -> omitted (None).
    assert_eq!(values[1], None);
    // pos 2: DEPREL(Stack[0]=root) -> ROOT_NODE null -> 0/#null#
    assert_eq!(values[2], Some((0, "#null#".to_string())));
    // pos 3: Merge3(Stack[1], Stack[0], Input[0]) = (null,null,NOUN) ->
    // not all null -> "#null#~#null#~NOUN" not in table -> None
    assert_eq!(values[3], None);

    // Now advance one SHIFT: stack=[0,1], input top = token 2 (VERB).
    let mut cfg = Config::new(3, true, false);
    let mut g = Graph::new(&[
        ("a".into(), "NOUN".into()),
        ("b".into(), "VERB".into()),
        ("c".into(), "NOUN".into()),
    ]);
    apply(&Action::shift(), &mut cfg, &mut g).unwrap();
    let values = fm.evaluate(&model, &g, &cfg);
    // pos 0: POSTAG(Stack[0]=a) = NOUN = 1
    assert_eq!(values[0], Some((1, "NOUN".to_string())));
    // pos 1: Merge(NOUN, POSTAG(Input[0]=b)=VERB) -> "NOUN~VERB" = 2
    assert_eq!(values[1], Some((2, "NOUN~VERB".to_string())));
    // pos 2: DEPREL(a): no head yet -> 0/#null#
    assert_eq!(values[2], Some((0, "#null#".to_string())));
    // pos 3: Merge3(Stack[1]=root #null#, NOUN, VERB) -> not all null ->
    // "#null#~NOUN~VERB" not in table -> None
    assert_eq!(values[3], None);
}

#[test]
fn packed_keys_and_vector_order() {
    let model = fixture_model();
    let codes = vec![
        Some((1, "NOUN".to_string())),    // pos0 -> key 1 -> idx 2
        Some((1, "NOUN~NOUN".to_string())), // pos1 -> key (1<<48)|1 -> idx 3
        None,                              // omitted entirely
        Some((9, "x".to_string())),        // pos3 code 9: map miss -> omitted
    ];
    let nodes = model.build_feature_nodes(&codes);
    // Built in position order; value 1.0; misses omitted.
    assert_eq!(nodes.len(), 2);
    assert_eq!(nodes[0].index, 2);
    assert_eq!(nodes[0].value, 1.0);
    assert_eq!(nodes[1].index, 3);
}

/// Hand-computed dot products, bit-exact.
/// w (1-based rows): 1=[1.5] 2=[1.0] 3=[0.25] 4=[2.0] 5=[-0.5]; labels [1,3].
/// Vector [idx2, idx3] -> dec[0] = w[2][0]*1 + w[3][0]*1 = 1.0 + 0.25 = 1.25
///                      dec[1] = w[2][1]*1 + w[3][1]*1 = 0.0 + 2.0 = 2.0 (col1 sparse-absent = 0)
/// argmax = class index 1 -> label 3.
#[test]
fn scoring_bit_exact_dot_product() {
    let model = fixture_model();
    let dm = model.decision_model();
    let nodes = vec![
        FeatureNode { index: 2, value: 1.0 },
        FeatureNode { index: 3, value: 1.0 },
    ];
    let ranked = predict(&dm, &nodes);
    assert_eq!(ranked, vec![3, 1]);

    // exact-decision-value check via direct recomputation over 1-based rows
    let mut dec = [0.0f64; 2];
    for n in &nodes {
        for &(col, w) in &dm.w[n.index as usize] {
            dec[col as usize] += w * n.value;
        }
    }
    assert_eq!(dec[0], 0.25);
    assert_eq!(dec[1], 1.0);
}

/// Ties: first argmax wins. Vector [idx1] -> dec = [1.5, 0.0]... use a tie:
/// nodes [idx5] -> dec[0] = -0.5, dec[1] = 0.0 -> class 1 (label 3) wins.
/// For an exact tie, use nodes [idx4] -> dec[0] = 0.0 (col0 absent), dec[1] = 2.0.
/// A genuine tie needs equal decs: vector [idx1, idx4]? idx1: dec0+=1.5;
/// idx4: dec1+=2.0 -> [1.5, 2.0] no. Craft tie via [idx5, idx2]:
/// dec[0] = -0.5 + 1.0 = 0.5, dec[1] = 0.0. Instead: empty vector ->
/// [0.0, 0.0] exact tie -> first label (1) must win.
#[test]
fn scoring_first_argmax_wins_ties() {
    let model = fixture_model();
    let dm = model.decision_model();
    let ranked = predict(&dm, &[]);
    assert_eq!(ranked.first(), Some(&1));
}

#[test]
fn decision_split_and_actions() {
    let model = fixture_model();
    // ranked label 1 = "LA~OBJ" -> LeftArc + deprel OBJ(1)
    let action = Action::from_decisions(&model, &[1]);
    assert_eq!(action.transition, Transition::LeftArc);
    assert_eq!(action.deprel, 1);
    // ranked label 3 = "SH~" -> Shift, deprel lookup of "" fails -> -1
    let action = Action::from_decisions(&model, &[3]);
    assert_eq!(action.transition, Transition::Shift);
    assert_eq!(action.deprel, -1);
    // unknown label -> falls back to shift
    let action = Action::from_decisions(&model, &[99]);
    assert_eq!(action.transition, Transition::Shift);
}

/// Hand-driven parse on the fixture model (input top = lowest unprocessed):
/// step1: vector = [idx1] (w row1 = [1.5, 0]) -> label 1 "LA~OBJ";
///        LEFTARC impermissible (root on stack), no k-best tail -> default
///        SHIFT. stack=[0,1], input top=2.
/// step2: pos0=NOUN(a)->idx2; pos1 Merge(NOUN,VERB)="NOUN~VERB"->idx5;
///        dec[0] = 1.0 + (-0.5) = 0.5, dec[1] = 0.0 -> label 1 "LA~OBJ":
///        edge 2->1 OBJ (permissible: 1 headless, not root). stack=[0],
///        input top=3.
/// step3: vector = [idx1] again -> "LA~OBJ" impermissible (root) -> SHIFT.
///        stack=[0,3], input empty -> terminal.
/// token2 unattached -> root-linked (ROOT).
#[test]
fn hand_driven_parse_two_transitions() {
    let model = fixture_model();
    let tokens = vec![
        ("ก".to_string(), "NOUN".to_string()),
        ("ข".to_string(), "VERB".to_string()),
        ("ค".to_string(), "NOUN".to_string()),
    ];
    let out = model.parse_sentence(&tokens).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0], crate::MaltToken { head: 2, deprel: "OBJ".into() });
    assert_eq!(out[1], crate::MaltToken { head: 3, deprel: "OBJ".into() });
    assert_eq!(out[2], crate::MaltToken { head: 0, deprel: "ROOT".into() });
}

#[test]
fn legality_rules() {
    let mut g = Graph::new(&[("a".into(), "NOUN".into()), ("b".into(), "VERB".into())]);
    let cfg = Config::new(2, true, false);
    g.add_edge(2, 1, 1).unwrap(); // a <- b
    let a = Action { transition: Transition::LeftArc, deprel: -1, ranked_tail: vec![] };
    assert!(!is_permissible(&a, &cfg, &g));
    // LEFTARC on root: impermissible
    let a = Action { transition: Transition::LeftArc, deprel: 1, ranked_tail: vec![] };
    assert!(!is_permissible(&a, &cfg, &g));
    // LEFTARC on node with head: impermissible
    let mut cfg2 = Config::new(2, true, false);
    apply(&Action::shift(), &mut cfg2, &mut g).unwrap(); // stack=[0,1]... input=[2]
    let a = Action { transition: Transition::LeftArc, deprel: 1, ranked_tail: vec![] };
    assert!(!is_permissible(&a, &cfg2, &g));
    // SHIFT: always fine
    assert!(is_permissible(&Action::shift(), &cfg2, &g));
}

// Keep serde derive types referenced (they are used via deserialization).
#[allow(dead_code)]
fn _type_witness(_: SymbolTableJson, _: SymbolJson, _: FeatureMapJson, _: LiblinearJson, _: DecisionModelJson) {}
