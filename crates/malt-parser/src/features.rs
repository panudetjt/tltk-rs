//! Feature model: parse `feature_strings`, evaluate for a parser state.
//!
//! Ports `InputColumnFeature`/`OutputColumnFeature`, `NivreAddressFunction`
//! (`Stack[k]`/`Input[k]`), `DGraphAddressFunction` (`head`/`ldep`/`rdep`)
//! and `MergeFeature`/`Merge3Feature`. All our columns are STRING-typed, so
//! merges take the STRING branch: join component symbols with `~`, look the
//! joined string up in the MERGE2_*/MERGE3_* table; when every component is
//! null the value is the FIRST component's (null) code from its own table —
//! exactly `MergeFeature.update()`.
//!
//! A `None` result mirrors the liblinear-omit path: the resolved symbol has
//! no code in its table (unseen at training time) or the merged string is
//! absent from the merge table, so no feature-map key could exist either.
//! Java would produce a code via `addSymbol` at parse time and then omit at
//! the `getIndex == -1` check; returning None here is observably identical.

use crate::graph::Graph;
use crate::model::MaltModel;
use crate::transition::Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Column {
    Form,
    Postag,
    Deprel,
}

impl Column {
    fn parse(s: &str) -> Result<Column, String> {
        match s {
            "FORM" | "LEMMA" => Ok(Column::Form),
            "POSTAG" | "CPOSTAG" => Ok(Column::Postag),
            "DEPREL" => Ok(Column::Deprel),
            other => Err(format!("unsupported column {other}")),
        }
    }

    fn table(&self) -> &'static str {
        match self {
            Column::Form => "FORM",
            Column::Postag => "POSTAG",
            Column::Deprel => "DEPREL",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Address {
    Stack(usize),
    Input(usize),
    DGraph { sub: Sub, addr: Box<Address> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sub {
    Head,
    Ldep,
    Rdep,
}


#[derive(Debug, Clone, PartialEq, Eq)]
enum Feature {
    Column { col: Column, addr: Address },
    Merge2(Box<Feature>, Box<Feature>),
    Merge3(Box<Feature>, Box<Feature>, Box<Feature>),
}

pub struct FeatureModel {
    features: Vec<Feature>,
}

impl FeatureModel {
    pub fn new(model: &MaltModel) -> Result<FeatureModel, String> {
        // Java's SpecificationSubModel stores specs in a TreeSet<String>
        // after stripping ALL blanks — the wire order is that sorted order,
        // not the XML order. Position numbering follows it.
        let mut specs: Vec<String> = model
            .feature_strings()
            .iter()
            .map(|s| s.replace(' ', ""))
            .collect();
        specs.sort();
        specs.dedup();
        let features = specs
            .iter()
            .map(|s| parse_feature(s))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(FeatureModel { features })
    }

    /// Evaluate all features in spec order; `None` = feature omitted from
    /// the liblinear vector (see module doc).
    pub fn evaluate(
        &self,
        model: &MaltModel,
        graph: &Graph,
        config: &Config,
    ) -> Vec<Option<(i32, String)>> {
        self.features
            .iter()
            .map(|f| self.eval(model, f, graph, config))
            .collect()
    }

    fn eval(
        &self,
        model: &MaltModel,
        f: &Feature,
        graph: &Graph,
        config: &Config,
    ) -> Option<(i32, String)> {
        match f {
            Feature::Column { col, addr } => {
                let node = resolve_address(addr, graph, config);
                match (node, *col) {
                    // NO_NODE (no node) and ROOT_NODE share "#null#" = 0.
                    (None, _) | (Some(0), _) => Some((0, "#null#".to_string())),
                    (Some(i), Column::Form) => {
                        resolve_symbol(model, "FORM", graph.form(i))
                    }
                    (Some(i), Column::Postag) => {
                        resolve_symbol(model, "POSTAG", graph.postag(i))
                    }
                    (Some(i), Column::Deprel) => {
                        if graph.has_head(i) {
                            let code = graph.deprel_code(i);
                            let symbol = model
                                .tables()
                                .table("DEPREL")
                                .and_then(|t| t.symbol_of(code))
                                .map(str::to_string)?;
                            Some((code, symbol))
                        } else {
                            // NullValueId.NO_VALUE -> "#null#" = 0.
                            Some((0, "#null#".to_string()))
                        }
                    }
                }
            }
            Feature::Merge2(a, b) => {
                let (code_a, sym_a) = self.eval(model, a, graph, config)?;
                let (code_b, sym_b) = self.eval(model, b, graph, config)?;
                let name = format!("MERGE2_{}_{}", map_identifier(a), map_identifier(b));
                merge(model, &name, &[(code_a, sym_a), (code_b, sym_b)])
            }
            Feature::Merge3(a, b, c) => {
                let va = self.eval(model, a, graph, config)?;
                let vb = self.eval(model, b, graph, config)?;
                let vc = self.eval(model, c, graph, config)?;
                let name = format!(
                    "MERGE3_{}_{}_{}",
                    map_identifier(a),
                    map_identifier(b),
                    map_identifier(c)
                );
                merge(model, &name, &[va, vb, vc])
            }
        }
    }
}

/// MergeFeature/Merge3Feature.update, STRING type:
/// all-null => first component's code (from its own table, symbol kept);
/// otherwise join symbols with `~` and resolve through the merge table.
fn merge(
    model: &MaltModel,
    table: &str,
    values: &[(i32, String)],
) -> Option<(i32, String)> {
    let null = |v: &(i32, String)| v.1 == "#null#";
    if values.iter().all(null) {
        return Some(values[0].clone());
    }
    let merged = values
        .iter()
        .map(|v| v.1.as_str())
        .collect::<Vec<_>>()
        .join("~");
    resolve_symbol(model, table, &merged)
}

fn resolve_symbol(model: &MaltModel, table: &str, symbol: &str) -> Option<(i32, String)> {
    let code = model.tables().code_of(table, symbol).ok()?;
    Some((code, symbol.to_string()))
}

/// `NivreAddressFunction` + `DGraphAddressFunction`. Node 0 (root) has no
/// head and, during parsing, no dependents (root linking happens at the end).
fn resolve_address(addr: &Address, graph: &Graph, config: &Config) -> Option<u32> {
    match addr {
        Address::Stack(k) => config.stack_node(*k),
        Address::Input(k) => config.input_node(*k),
        Address::DGraph { sub, addr } => {
            let node = resolve_address(addr, graph, config)?;
            if node == 0 {
                return None;
            }
            match sub {
                Sub::Head => graph.head_of(node),
                Sub::Ldep => graph.leftmost_dependent(node),
                Sub::Rdep => graph.rightmost_dependent(node),
            }
        }
    }
}

fn parse_feature(s: &str) -> Result<Feature, String> {
    let s = s.trim();
    if let Some(inner) = strip(s, "Merge3(", ")") {
        let parts = split_args(&inner);
        if parts.len() != 3 {
            return Err(format!("Merge3 needs 3 args: {s}"));
        }
        return Ok(Feature::Merge3(
            Box::new(parse_feature(&parts[0])?),
            Box::new(parse_feature(&parts[1])?),
            Box::new(parse_feature(&parts[2])?),
        ));
    }
    if let Some(inner) = strip(s, "Merge(", ")") {
        let parts = split_args(&inner);
        if parts.len() != 2 {
            return Err(format!("Merge needs 2 args: {s}"));
        }
        return Ok(Feature::Merge2(
            Box::new(parse_feature(&parts[0])?),
            Box::new(parse_feature(&parts[1])?),
        ));
    }
    for prefix in ["InputColumn(", "OutputColumn("] {
        if let Some(inner) = strip(s, prefix, ")") {
            let parts = split_args(&inner);
            if parts.len() != 2 {
                return Err(format!("{prefix} needs 2 args: {s}"));
            }
            return Ok(Feature::Column {
                col: Column::parse(parts[0].trim())?,
                addr: parse_address(parts[1].trim())?,
            });
        }
    }
    Err(format!("unsupported feature {s}"))
}

fn parse_address(s: &str) -> Result<Address, String> {
    if let Some(inner) = strip(s, "Stack[", "]") {
        return inner
            .trim()
            .parse()
            .map(Address::Stack)
            .map_err(|_| format!("bad Stack index {s}"));
    }
    if let Some(inner) = strip(s, "Input[", "]") {
        return inner
            .trim()
            .parse()
            .map(Address::Input)
            .map_err(|_| format!("bad Input index {s}"));
    }
    if let Some(inner) = strip(s, "head(", ")") {
        return Ok(Address::DGraph {
            sub: Sub::Head,
            addr: Box::new(parse_address(inner.trim())?),
        });
    }
    if let Some(inner) = strip(s, "ldep(", ")") {
        return Ok(Address::DGraph {
            sub: Sub::Ldep,
            addr: Box::new(parse_address(inner.trim())?),
        });
    }
    if let Some(inner) = strip(s, "rdep(", ")") {
        return Ok(Address::DGraph {
            sub: Sub::Rdep,
            addr: Box::new(parse_address(inner.trim())?),
        });
    }
    Err(format!("unsupported address {s}"))
}

/// `MergeFeature.getMapIdentifier` = symbol table name = the column name for
/// column features, `MERGE2_*`/`MERGE3_*` for nested merges.
fn map_identifier(f: &Feature) -> String {
    match f {
        Feature::Column { col, .. } => col.table().to_string(),
        Feature::Merge2(..) => {
            let (a, b) = if let Feature::Merge2(a, b) = f {
                (a, b)
            } else {
                unreachable!()
            };
            format!("MERGE2_{}_{}", map_identifier(a), map_identifier(b))
        }
        Feature::Merge3(..) => {
            if let Feature::Merge3(a, b, c) = f {
                format!(
                    "MERGE3_{}_{}_{}",
                    map_identifier(a),
                    map_identifier(b),
                    map_identifier(c)
                )
            } else {
                unreachable!()
            }
        }
    }
}

fn strip(s: &str, prefix: &str, suffix: &str) -> Option<String> {
    let rest = s.strip_prefix(prefix)?;
    let rest = rest.strip_suffix(suffix)?;
    Some(rest.to_string())
}

/// Split top-level comma-separated args (nested parens never contain
/// top-level commas in this spec).
fn split_args(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => {
                parts.push(cur.trim().to_string());
                cur = String::new();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur.trim().to_string());
    }
    parts
}
