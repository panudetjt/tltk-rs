//! `tltk-web` — browser WebAssembly glue crate for `tltk-core`.
//!
//! Provides `Tltk`, a persistent session wrapper exposed to JavaScript via `wasm-bindgen`.
//!
//! # JS API
//!
//! ```javascript
//! import init, { Tltk } from './tltk_web.js';
//! await init();
//!
//! // tier-1 models (thdict, sylseg, sylvar, phs, pos, ner, sylrule.txt, thaisyl.txt)
//! const tltk = new Tltk(names, datas);
//! // or: new Tltk({ names, datas })
//! // or: new Tltk({ 'thdict.bin.gz': uint8Array, ... })
//!
//! // tier-2 models (edu.bin.gz, w2v.bin.gz)
//! tltk.attach(names, datas);
//! // or: tltk.attach({ names, datas })
//!
//! const res = tltk.process('segment', 'แมวกินปลา', 'medium');
//! ```

use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Tltk {
    sess: tltk_core::Session<'static>,
    inner: tltk_core::Tltk,
}

fn parse_blobs(
    arg0: JsValue,
    arg1: Option<Vec<js_sys::Uint8Array>>,
) -> Result<HashMap<String, Vec<u8>>, JsError> {
    let mut map = HashMap::new();
    if let Some(datas) = arg1 {
        let names_arr = js_sys::Array::from(&arg0);
        if names_arr.length() != datas.len() as u32 {
            return Err(JsError::new(&format!(
                "mismatched names ({}) and datas ({}) array lengths",
                names_arr.length(),
                datas.len()
            )));
        }
        for (i, data) in datas.into_iter().enumerate() {
            let name_val = names_arr.get(i as u32);
            let name = name_val
                .as_string()
                .ok_or_else(|| JsError::new(&format!("name at index {} is not a string", i)))?;
            map.insert(name, data.to_vec());
        }
        return Ok(map);
    }

    if arg0.is_object() {
        let has_names = js_sys::Reflect::has(&arg0, &"names".into()).unwrap_or(false);
        let has_datas = js_sys::Reflect::has(&arg0, &"datas".into()).unwrap_or(false);
        if has_names && has_datas {
            let names_val = js_sys::Reflect::get(&arg0, &"names".into())
                .map_err(|e| JsError::new(&format!("{:?}", e)))?;
            let datas_val = js_sys::Reflect::get(&arg0, &"datas".into())
                .map_err(|e| JsError::new(&format!("{:?}", e)))?;
            let names_arr = js_sys::Array::from(&names_val);
            let datas_arr = js_sys::Array::from(&datas_val);
            if names_arr.length() != datas_arr.length() {
                return Err(JsError::new(&format!(
                    "mismatched names ({}) and datas ({}) array lengths",
                    names_arr.length(),
                    datas_arr.length()
                )));
            }
            for i in 0..names_arr.length() {
                let name = names_arr
                    .get(i)
                    .as_string()
                    .ok_or_else(|| JsError::new(&format!("name at index {} is not a string", i)))?;
                let data_js = datas_arr.get(i);
                let u8_arr = js_sys::Uint8Array::new(&data_js);
                map.insert(name, u8_arr.to_vec());
            }
            return Ok(map);
        }

        let entries = js_sys::Object::entries(&js_sys::Object::from(arg0));
        for i in 0..entries.length() {
            let entry = js_sys::Array::from(&entries.get(i));
            let key = entry
                .get(0)
                .as_string()
                .ok_or_else(|| JsError::new(&format!("key at index {} is not a string", i)))?;
            let val = entry.get(1);
            let u8_arr = js_sys::Uint8Array::new(&val);
            map.insert(key, u8_arr.to_vec());
        }
        return Ok(map);
    }

    Err(JsError::new(
        "blobs must be (names, datas) or { names, datas } or a dictionary object",
    ))
}

#[wasm_bindgen]
impl Tltk {
    /// Construct a new `Tltk` instance from tier-1 model blobs.
    ///
    /// Accepts:
    /// - Two arrays: `new Tltk(names, datas)` where `names: string[]`, `datas: Uint8Array[]`
    /// - An object: `new Tltk({ names, datas })`
    /// - A dictionary: `new Tltk({ "thdict.bin.gz": uint8Array, ... })`
    #[wasm_bindgen(constructor)]
    pub fn new(
        names_or_obj: JsValue,
        maybe_datas: Option<Vec<js_sys::Uint8Array>>,
    ) -> Result<Tltk, JsError> {
        let blobs = parse_blobs(names_or_obj, maybe_datas)?;
        let models =
            tltk_core::model::Models::try_from_blobs(&blobs).map_err(|e| JsError::new(&e))?;
        let inner = tltk_core::Tltk::new(models);
        let sess = unsafe {
            let models_ref: &'static tltk_core::model::Models = &*(&*inner.models as *const _);
            tltk_core::Session {
                m: models_ref,
                total_word: models_ref.sylseg.total_word as i64,
                total_lex: models_ref.sylseg.total_lex as i64,
                bicount_ov: HashMap::new(),
                count_ov: HashMap::new(),
                colloc_bumped: std::collections::HashSet::new(),
                pronun: HashMap::new(),
            }
        };
        Ok(Tltk { sess, inner })
    }

    /// Attach tier-2 model blobs (edu.bin.gz, w2v.bin.gz, and ner.bin.gz if given).
    pub fn attach(
        &mut self,
        names_or_obj: JsValue,
        maybe_datas: Option<Vec<js_sys::Uint8Array>>,
    ) -> Result<(), JsError> {
        let blobs = parse_blobs(names_or_obj, maybe_datas)?;
        let models = std::sync::Arc::get_mut(&mut self.inner.models)
            .ok_or_else(|| JsError::new("failed to obtain mutable reference to models"))?;
        models.attach_blobs(&blobs).map_err(|e| JsError::new(&e))?;
        self.sess.m = unsafe { &*(models as *const _) };
        Ok(())
    }

    /// Process a TLTK function request.
    ///
    /// Mirrors `tltk_core::Session::process` with identical error strings.
    pub fn process(&mut self, function: &str, text: &str, cutoff: &str) -> Result<String, JsError> {
        self.sess
            .process(function, text, cutoff)
            .map_err(|e| JsError::new(&e.0))
    }
}
