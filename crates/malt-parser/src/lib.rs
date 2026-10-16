//! `malt-parser` — inference-only port of MaltParser 1.9.2's parsing engine
//! (the engine behind tltk's `MaltParser()` / `thamalt.mco`), so runtime no
//! longer needs a JVM.
//!
//! # What was ported, from which Java classes
//!
//! - Blob model (`model.rs`): the pinned schema-v1 JSON contract
//!   (`malt.bin.gz` = gzip(JSON)) produced by the converter task. Decode is
//!   strict on `format`/`format_version`, tolerant of unknown extra keys.
//! - Symbol tables (`symbol.rs`): `core/symbol/parse/ParseSymbolTable(Handler)`
//!   with `null_value=one` semantics
//!   (`core/symbol/nullvalue/{Input,Output}NullValues.java`): code 0 is
//!   `#null#`, shared by NO_NODE / ROOT_NODE / NO_VALUE.
//! - Feature model (`features.rs`): `core/syntaxgraph/feature/
//!   {InputColumnFeature,OutputColumnFeature,DGraphAddressFunction}.java`,
//!   `parser/algorithm/nivre/NivreAddressFunction.java` and
//!   `core/feature/map/{MergeFeature,Merge3Feature}.java`. The 23-feature
//!   spec is parsed from `feature_strings` (verbatim wire names such as
//!   `InputColumn(POSTAG, Stack[0])`); `feature_spec_xml` is carried in the
//!   blob but not parsed — the strings are the same model flattened and the
//!   blob contract makes the strings authoritative. `Merge*` resolve through
//!   their own symbol tables (`MERGE2_*`/`MERGE3_*`), joining the component
//!   *symbols* with `~` (STRING branch; all our columns are STRING) and
//!   keeping the first component's null code when ALL components are null.
//! - Parse-time graph (`graph.rs`): `core/lw/graph/LWDependencyGraph.java`
//!   plus `LWNode.java`. Left/right dependent lookups rescan the head array in
//!   index order, matching `getListOf{..}Dependents`.
//!
//! - Transition system (`transition.rs`): `parser/algorithm/nivre/
//!   ArcEager.java` + `NivreConfig.java` with `allow_root=true`,
//!   `allow_reduce=false`, `enforce_tree=false` (so the deterministic
//!   REDUCE/UNSHIFT branches never fire; every step is predicted, SHIFT is
//!   the default action when the k-best list is exhausted).
//! - Scoring (`scoring.rs`): `core/lw/parser/LWClassifier.java` +
//!   `ml/lib/MaltLiblinearModel.predict_one/predict` — f64 accumulation in
//!   the feature vector's built order (feature-position order, see
//!   `Lib.predict`), selection-sort ranking with strict `>` (first-argmax
//!   wins ties).
//!
//! # Exactness contract
//!
//! Bit-exact with the Java engine for a given blob: same packed keys
//! (`(position << 48) | code`), same vector build order, f64 dot products in
//! the same order, first-argmax-wins tie behavior. A later integration task
//! proves this against Java-captured goldens.
//!
//! # Known deviations
//!
//! - Training, options/flow/plugin machinery, k-best > 1 output, and
//!   pseudo-projectivity (pproj marking = none in the shipped model) are not
//!   ported; the deprojectivizer is therefore absent by design.
//! - Input is `&[(String, String)]` (form, postag) instead of CoNLL-X text;
//!   the wire layer lives in the integration task.
//! - The `LWDecisionModel` fallback `singleDecision.addDecision(1)` when a
//!   classifier file is missing is an error here: blobs always carry the
//!   single `odm0` model (decode enforces exactly one).
use crate::features::FeatureModel;
use crate::graph::Graph;
#[cfg(test)]
mod tests;
use crate::model::MaltModel;
use crate::scoring::predict;
use crate::transition::{apply, is_permissible, legal_default, Action, Config};

pub mod features;
pub mod graph;
pub mod model;
pub mod scoring;
pub mod symbol;
pub mod transition;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaltToken {
    pub head: u32,
    pub deprel: String,
}

#[derive(Debug)]
pub enum ParseError {
    Decode(String),
    Internal(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Decode(m) | ParseError::Internal(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for ParseError {}

/// Safety valve mirroring Java's unbounded `while (!config.isTerminalState())`:
/// arc-eager over n tokens performs at most a linear number of transitions;
/// 8n + 64 is far above that bound and only reachable on an engine bug.
fn max_steps(n: usize) -> usize {
    8 * n + 64
}

impl MaltModel {
    /// Parse one sentence given (form, postag) pairs. Deterministic and
    /// bit-exact against the Java engine on the same blob.
    pub fn parse_sentence(
        &self,
        tokens: &[(String, String)],
    ) -> Result<Vec<MaltToken>, ParseError> {
        let feature_model = FeatureModel::new(self).map_err(ParseError::Internal)?;
        let mut graph = Graph::new(tokens);
        let mut config = Config::new(graph.n_tokens(), self.allow_root, self.allow_reduce);

        let mut steps = 0;
        while !config.is_terminal() {
            steps += 1;
            if steps > max_steps(graph.n_tokens()) {
                return Err(ParseError::Internal(
                    "transition loop exceeded step bound".into(),
                ));
            }

            // LWDeterministicParser.parse: deterministic action first, else
            // predict; while illegal, walk down the k-best list; when
            // exhausted, the system default action (SHIFT, unlabeled).
            let mut action = legal_default()
                .unwrap_or_else(|| self.predict_action(&feature_model, &graph, &config));
            while !is_permissible(&action, &config, &graph) {
                match action.next_candidate() {
                    Some(a) => action = a,
                    None => {
                        action = Action::shift();
                        break;
                    }
                }
            }
            apply(&action, &mut config, &mut graph)?;
        }
        graph.link_all_trees_to_root(self.root_label_code);

        graph.tokens(self.tables())
    }

    /// LWDecisionModel.predict with one decision (odm0) and one_prediction:
    /// evaluate features, build the vector, score, keep the ranked labels.
    fn predict_action(&self, feature_model: &FeatureModel, graph: &Graph, config: &Config) -> Action {
        let values = feature_model.evaluate(self, graph, config);
        let nodes = self.build_feature_nodes(&values);
        let dm = self.decision_model();
        let ranked = predict(&dm, &nodes);
        Action::from_decisions(self, &ranked)
    }
}
