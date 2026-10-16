//! Parse-time dependency graph: `LWDependencyGraph` + `LWNode`.
//! Nodes 1..n are tokens; node 0 is the root. Head array only — dependent
//! lists are recomputed on demand exactly like the Java scans (index order,
//! so leftmost = first, rightmost = last).

use crate::symbol::SymbolTables;
use crate::{MaltToken, ParseError};

pub struct Graph {
    n_tokens: usize,
    /// head[i] for node i (u32::MAX = unlinked); node 0 unused.
    head: Vec<u32>,
    deprel: Vec<i32>,
    forms: Vec<String>,
    postags: Vec<String>,
}

impl Graph {
    pub fn new(tokens: &[(String, String)]) -> Graph {
        let n = tokens.len();
        Graph {
            n_tokens: n,
            head: vec![u32::MAX; n + 1],
            deprel: vec![-1; n + 1],
            forms: tokens.iter().map(|t| t.0.clone()).collect(),
            postags: tokens.iter().map(|t| t.1.clone()).collect(),
        }
    }

    pub fn n_tokens(&self) -> usize {
        self.n_tokens
    }

    pub fn has_head(&self, i: u32) -> bool {
        self.head[i as usize] != u32::MAX
    }

    /// Node's head index; `None` when unlinked (root has none).
    pub fn head_of(&self, i: u32) -> Option<u32> {
        if self.has_head(i) {
            Some(self.head[i as usize])
        } else {
            None
        }
    }

    pub fn is_root(&self, i: u32) -> bool {
        i == 0
    }

    /// `LWDependencyGraph.addDependencyEdge`.
    pub fn add_edge(&mut self, head: u32, dependent: u32, deprel: i32) -> Result<(), ParseError> {
        if dependent == 0 || head as usize > self.n_tokens || dependent as usize > self.n_tokens {
            return Err(ParseError::Internal(format!(
                "illegal edge {head}->{dependent}"
            )));
        }
        if self.has_head(dependent) {
            return Err(ParseError::Internal(format!(
                "node {dependent} already has a head"
            )));
        }
        self.head[dependent as usize] = head;
        self.deprel[dependent as usize] = deprel;
        Ok(())
    }

    fn dependents_of(&self, i: u32) -> impl Iterator<Item = u32> + '_ {
        (1..=self.n_tokens as u32).filter(move |&d| self.head[d as usize] == i)
    }

    /// `getLeftmostDependent`.
    pub fn leftmost_dependent(&self, i: u32) -> Option<u32> {
        self.dependents_of(i).find(|&d| d < i)
    }

    /// `getRightmostDependent`.
    pub fn rightmost_dependent(&self, i: u32) -> Option<u32> {
        self.dependents_of(i).filter(|&d| d > i).max()
    }

    /// `linkAllTreesToRoot`: every node without a head gets head 0 with the
    /// default root edge label.
    pub fn link_all_trees_to_root(&mut self, root_label_code: i32) {
        for i in 1..=self.n_tokens as u32 {
            if !self.has_head(i) {
                self.head[i as usize] = 0;
                self.deprel[i as usize] = root_label_code;
            }
        }
    }

    pub fn deprel_code(&self, i: u32) -> i32 {
        self.deprel[i as usize]
    }

    pub fn form(&self, i: u32) -> &str {
        &self.forms[(i - 1) as usize]
    }

    pub fn postag(&self, i: u32) -> &str {
        &self.postags[(i - 1) as usize]
    }

    /// Output tokens (head 0 = root, deprel resolved through DEPREL table).
    pub fn tokens(&self, tables: &SymbolTables) -> Result<Vec<MaltToken>, ParseError> {
        (1..=self.n_tokens as u32)
            .map(|i| {
                let code = self.deprel[i as usize];
                let deprel = tables
                    .table("DEPREL")
                    .and_then(|t| t.symbol_of(code))
                    .ok_or_else(|| ParseError::Internal(format!("deprel code {code} unknown")))?;
                Ok(MaltToken {
                    head: self.head[i as usize],
                    deprel: deprel.to_string(),
                })
            })
            .collect()
    }
}
