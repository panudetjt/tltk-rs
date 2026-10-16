//! Arc-eager transition system (nivreeager) with the shipped options
//! allow_root=true, allow_reduce=false, enforce_tree=false.
//! Ports `ArcEager.java` + `NivreConfig.java`.

use crate::graph::Graph;
use crate::model::MaltModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    Shift,
    Reduce,
    RightArc,
    LeftArc,
    #[allow(dead_code)]
    Unshift,
}

impl Transition {
    pub fn from_symbol(s: &str) -> Transition {
        match s {
            "SH" | "SHIFT" => Transition::Shift,
            "RE" | "REDUCE" => Transition::Reduce,
            "RA" | "RIGHTARC" => Transition::RightArc,
            "LA" | "LEFTARC" => Transition::LeftArc,
            "USH" | "UNSHIFT" => Transition::Unshift,
            other => panic!("unknown transition symbol {other}"),
        }
    }
}

/// A predicted action: transition plus deprel code for the arc labels.
/// `ranked` mirrors the k-best list (labels in predictor preference order);
/// the legality fallback walks it (`predictFromKBestList`).
#[derive(Debug, Clone)]
pub struct Action {
    pub transition: Transition,
    pub deprel: i32,
    /// Remaining candidates after this one (k-best tail).
    pub(crate) ranked_tail: Vec<(Transition, i32)>,
}

impl Action {
    pub fn shift() -> Action {
        Action {
            transition: Transition::Shift,
            deprel: -1,
            ranked_tail: Vec::new(),
        }
    }

    /// Build the action chain from ranked predictions, dropping symbols that
    /// do not exist in the combined decision table (`addDecision(-1)` guard:
    /// unknown -> decision -1 -> not permissible -> next candidate).
    pub fn from_decisions(model: &MaltModel, ranked: &[i32]) -> Action {
        let mut items = Vec::new();
        for &label in ranked {
            // symbol_of on the combined decision table: labels index the
            // `T.TRANS+A.DEPREL` table directly.
            if let Some(symbol) = model.tables().table("T.TRANS+A.DEPREL").and_then(|t| t.symbol_of(label)) {
                let (trans, deprel) = model.split_decision(symbol);
                items.push((trans, deprel));
            }
        }
        if items.is_empty() {
            return Action::shift();
        }
        let (transition, deprel) = items[0];
        Action {
            transition,
            deprel,
            ranked_tail: items[1..].to_vec(),
        }
    }

    pub fn next_candidate(&self) -> Option<Action> {
        let mut tail = self.ranked_tail.clone();
        let (transition, deprel) = tail.first().copied()?;
        tail.remove(0);
        Some(Action {
            transition,
            deprel,
            ranked_tail: tail,
        })
    }
}

pub struct Config {
    stack: Vec<u32>,
    input: Vec<u32>,
    /// Carried for parity with `NivreConfig`; the deterministic-action
    /// branch that reads it never fires with enforce_tree=false.
    #[allow(dead_code)]
    pub allow_root: bool,
    allow_reduce: bool,
}

impl Config {
    /// `NivreConfig.initialize`: stack = [root]; input = tokens n..1 (so the
    /// top of input is token 1).
    pub fn new(n_tokens: usize, allow_root: bool, allow_reduce: bool) -> Config {
        Config {
            stack: vec![0],
            input: (1..=n_tokens as u32).rev().collect(),
            allow_root,
            allow_reduce,
        }
    }

    /// `isTerminalState` (enforce_tree=false): input empty.
    pub fn is_terminal(&self) -> bool {
        self.input.is_empty()
    }
    pub fn stack_node(&self, k: usize) -> Option<u32> {
        if self.stack.len() > k {
            Some(self.stack[self.stack.len() - 1 - k])
        } else {
            None
        }
    }

    pub fn input_node(&self, k: usize) -> Option<u32> {
        if self.input.len() > k {
            Some(self.input[self.input.len() - 1 - k])
        } else {
            None
        }
    }

    fn stack_peek(&self) -> u32 {
        *self.stack.last().expect("stack never empty (root)")
    }

    fn input_peek(&self) -> u32 {
        *self.input.last().expect("caller checks non-empty")
    }
}

/// `ArcEager.getDeterministicAction` for the non-enforceTree branch: with
/// allow_root=true it is always None — every step is predicted.
pub fn legal_default() -> Option<Action> {
    None
}
pub fn is_permissible(action: &Action, config: &Config, graph: &Graph) -> bool {
    let trans = action.transition;
    let stack_peek = config.stack_peek();
    // (LEFTARC || RIGHTARC) must be labeled: our decisions always carry a
    // label; deprel == -1 means the combined symbol lacked a label part.
    if matches!(trans, Transition::LeftArc | Transition::RightArc) && action.deprel < 0 {
        return false;
    }
    if matches!(trans, Transition::LeftArc | Transition::Reduce) && stack_peek == 0 {
        return false; // stack peek is root
    }
    // LEFTARC && stackPeek.hasHead()
    if trans == Transition::LeftArc && graph.has_head(stack_peek) {
        return false;
    }
    // REDUCE && !stackPeek.hasHead() && !allow_reduce
    if trans == Transition::Reduce && !graph.has_head(stack_peek) && !config.allow_reduce {
        return false;
    }
    true
}
/// `ArcEager.apply` (non-enforceTree branch).
pub fn apply(
    action: &Action,
    config: &mut Config,
    graph: &mut Graph,
) -> Result<(), crate::ParseError> {
    match action.transition {
        Transition::LeftArc => {
            let s = config.stack_peek();
            let i = config.input_peek();
            graph.add_edge(i, s, action.deprel)?;
            config.stack.pop();
        }
        Transition::RightArc => {
            let s = config.stack_peek();
            let i = config.input_peek();
            graph.add_edge(s, i, action.deprel)?;
            let node = config.input.pop().expect("input non-empty");
            config.stack.push(node);
        }
        Transition::Reduce => {
            config.stack.pop();
        }
        // default (SHIFT)
        _ => {
            let node = config.input.pop().expect("input non-empty");
            config.stack.push(node);
        }
    }
    Ok(())
}
