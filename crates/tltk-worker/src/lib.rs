//! tltk-worker — Cloudflare Worker exposing the same JSON API as the
//! stichula validator: POST /process {text, function, cutoff} →
//! {success, function, input, result|error}.
//!
//! Model loading:
//! - small models (thdict/sylseg/sylvar/phs/pos/ner) are embedded in the WASM;
//! - edu + w2v are fetched once per isolate from the R2 bucket binding `MODELS`
//!   (keys: edu.bin.gz, w2v.bin.gz) and cached in a global. Without R2 bindings
//!   the corresponding functions error loudly.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;
use tltk_core::Tltk;
use worker::*;

static TLTK: Mutex<Option<Tltk>> = Mutex::new(None);
static INIT_FAIL: Mutex<Option<String>> = Mutex::new(None);

#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    router(req, env).await
}

async fn router(req: Request, env: Env) -> Result<Response> {
    let url = req.url()?;
    let path = url.path().to_string();
    match (req.method(), path.as_str()) {
        (Method::Post, "/process") | (Method::Post, "/py/tltk/process") => process(req, env).await,
        (Method::Get, "/") | (Method::Get, "/health") => Response::from_json(&serde_json::json!({
            "service": "tltk-rs worker",
            "functions": [
                "th2ipa","th2roman","word_segment","syl_segment","segment","g2p",
                "th2read","spell_variants","spoonerism","pos_tag","ner_tag",
                "similar_words","similarity","outofgroup","analogy","plotwords",
                "chunk","tnc_tag",
                "th2ipa_all","sylparse_all"
            ],
        })),
        _ => Response::error("Not Found", 404),
    }
}

async fn ensure_tltk(env: Env) -> Result<()> {
    if TLTK.lock().unwrap().is_some() {
        return Ok(());
    }
    if let Some(e) = INIT_FAIL.lock().unwrap().as_ref() {
        return Err(Error::RustError(format!("model init failed: {}", e)));
    }
    let mut extra: HashMap<String, Vec<u8>> = HashMap::new();
    match env.bucket("MODELS") {
        Ok(bucket) => {
            for key in ["edu.bin.gz", "w2v.bin.gz"] {
                match bucket.get(key).execute().await {
                    Ok(Some(obj)) => match obj.body() {
                        Some(body) => match body.bytes().await {
                            Ok(bytes) => {
                                extra.insert(key.to_string(), bytes.to_vec());
                            }
                            Err(e) => console_error!("read {} failed: {}", key, e),
                        },
                        None => console_error!("{} has no body", key),
                    },
                    Ok(None) => console_error!("{} not found in R2 MODELS", key),
                    Err(e) => console_error!("R2 get {} failed: {}", key, e),
                }
            }
        }
        Err(e) => console_error!("no MODELS R2 binding: {}", e),
    }
    let mut blobs = tltk_core::embedded::blobs();
    for (k, v) in extra {
        blobs.insert(k.trim_end_matches(".gz").to_string(), v);
    }
    let models = match tltk_core::model::Models::try_from_blobs(&blobs) {
        Ok(m) => m,
        Err(e) => {
            *INIT_FAIL.lock().unwrap() = Some(e.clone());
            return Err(Error::RustError(format!("model init failed: {}", e)));
        }
    };
    *TLTK.lock().unwrap() = Some(Tltk::new(models));
    Ok(())
}

async fn process(mut req: Request, env: Env) -> Result<Response> {
    ensure_tltk(env).await?;
    let body: Value = match req.json().await {
        Ok(v) => v,
        Err(e) => {
            return Response::from_json(&serde_json::json!({
                "success": false, "error": format!("bad json body: {}", e)
            }))
        }
    };
    let text = body.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let function = body
        .get("function")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let cutoff = body
        .get("cutoff")
        .and_then(|v| v.as_str())
        .unwrap_or("medium")
        .to_string();

    if text.trim().is_empty() {
        return Response::from_json(&serde_json::json!({
            "success": false, "error": "กรุณาใส่ข้อความ"
        }));
    }

    let tltk = TLTK.lock().unwrap().as_ref().expect("initialised above").clone_handle();
    let mut sess = tltk.session();
    let out = sess.process(&function, &text, &cutoff);
    let payload = match out {
        Ok(result) => serde_json::json!({
            "success": true, "function": function, "input": text, "result": result
        }),
        Err(e) => serde_json::json!({
            "success": false, "function": function, "input": text, "error": e.0
        }),
    };
    Response::from_json(&payload)
}

/// The Tltk handle is an Arc internally; borrow-then-process would fight the
/// Mutex guard across await points, so materialise a cheap clone instead.
trait CloneHandle {
    fn clone_handle(&self) -> Tltk;
}

impl CloneHandle for Tltk {
    fn clone_handle(&self) -> Tltk {
        Tltk { models: self.models.clone() }
    }
}
