//! Symbol tables: `ParseSymbolTable` with `null_value=one` semantics.
//! Code 0 is the shared null symbol `#null#` (NO_NODE / ROOT_NODE / NO_VALUE).

use std::collections::HashMap;

pub const NULL_SYMBOL: &str = "#null#";

#[derive(Debug)]
pub struct SymbolTable {
    pub name: String,
    #[allow(dead_code)]
    pub header: Vec<String>,
    symbol_to_code: HashMap<String, i32>,
    code_to_symbol: HashMap<i32, String>,
}

impl SymbolTable {
    pub fn code_of(&self, symbol: &str) -> Option<i32> {
        self.symbol_to_code.get(symbol).copied()
    }

    #[allow(dead_code)]
    pub fn symbol_of(&self, code: i32) -> Option<&str> {
        self.code_to_symbol.get(&code).map(|s| s.as_str())
    }

    #[allow(dead_code)]
    pub fn null_code(&self) -> i32 {
        0
    }
}

#[derive(Debug, Default)]
pub struct SymbolTables {
    tables: HashMap<String, SymbolTable>,
}

impl SymbolTables {
    pub fn new(json: &[crate::model::SymbolTableJson]) -> Result<SymbolTables, String> {
        let mut tables = HashMap::new();
        for table in json {
            let mut symbol_to_code = HashMap::with_capacity(table.symbols.len());
            let mut code_to_symbol = HashMap::with_capacity(table.symbols.len());
            for sym in &table.symbols {
                symbol_to_code.insert(sym.symbol.clone(), sym.code);
                code_to_symbol.insert(sym.code, sym.symbol.clone());
            }
            // null_value=one: code 0 <-> "#null#" even when the converter
            // omitted it from the symbol list.
            symbol_to_code.entry(NULL_SYMBOL.to_string()).or_insert(0);
            code_to_symbol.entry(0).or_insert_with(|| NULL_SYMBOL.to_string());
            tables.insert(
                table.name.clone(),
                SymbolTable {
                    name: table.name.clone(),
                    header: table.header.clone(),
                    symbol_to_code,
                    code_to_symbol,
                },
            );
        }
        Ok(SymbolTables { tables })
    }

    pub fn table(&self, name: &str) -> Option<&SymbolTable> {
        self.tables.get(name)
    }

    pub fn code_of(&self, table: &str, symbol: &str) -> Result<i32, String> {
        self.tables
            .get(table)
            .ok_or_else(|| format!("missing symbol table {table}"))?
            .code_of(symbol)
            .ok_or_else(|| format!("symbol {symbol:?} not in table {table}"))
    }
}
