/**
 * loader.js - Tiered WebAssembly model and module loader for TLTK
 *
 * Provides:
 * 1. Runtime SIMD / scalar wasm flavor resolution via probe.js
 * 2. Tier-1 model loading (8 core dictionary, syllable, POS, NER files ~5.4MB)
 * 3. Tier-2 on-demand model loading (edu.bin.gz + w2v.bin.gz ~17.8MB)
 * 4. Error boundary ensuring Tier-2 failures do not break Tier-1 capabilities
 * 5. Array-parallel blob binding contract: { names: string[], datas: Uint8Array[] }
 */

import { selectFlavor } from './probe.js';

// Tier-1: 10 small model and rule files needed for base inference
// (wordtier = textana lexicon, textclass = TextClass/txt2feat RF models)
export const TIER1_FILES = [
  'thdict.bin.gz',
  'sylseg.bin.gz',
  'sylvar.bin.gz',
  'phs.bin.gz',
  'pos.bin.gz',
  'ner.bin.gz',
  'wordtier.bin.gz',
  'textclass.bin.gz',
  'sylrule.txt',
  'thaisyl.txt'
];

// Tier-2: 2 large files (discourse segmentation and word2vec embeddings)
export const TIER2_FILES = [
  'edu.bin.gz',
  'w2v.bin.gz'
];

// Functions requiring Tier-2 models
export const TIER2_FUNCTIONS = new Set([
  'similar_words',
  'similarity',
  'outofgroup',
  'analogy',
  'plotwords',
  'segment'
]);

/**
 * Version note: Model assets default to 'data/<filename>'.
 * A query string such as '?v=8f42992' can be appended for immutable cache-busting.
 */
const DEFAULT_VERSION = '69e25a3';

/** Asset URL in the same versioned form getAssetUrl produces (SW prefetch). */
export function assetUrl(filename) {
  return `data/${filename}?v=${DEFAULT_VERSION}`;
}

/**
 * Loads an asset as a Uint8Array, supporting browser fetch with 'force-cache'
 * and Node.js filesystem reading for headless test/smoke scripts.
 */
async function fetchAsset(url, onChunk = null) {
  // If running in Node.js without an HTTP(S) protocol, read directly from the filesystem
  if (typeof window === 'undefined' && !url.startsWith('http://') && !url.startsWith('https://')) {
    const fs = await import('node:fs/promises');
    const filePath = url.split('?')[0];
    const data = await fs.readFile(filePath);
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }

  const res = await fetch(url, { cache: 'force-cache' });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${res.statusText} loading ${url}`);
  }

  // Stream reader for progress reporting if available
  if (onChunk && res.body && typeof res.body.getReader === 'function') {
    const contentLength = res.headers.get('content-length');
    const totalBytes = contentLength ? parseInt(contentLength, 10) : 0;
    const reader = res.body.getReader();
    const chunks = [];
    let receivedBytes = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      receivedBytes += value.length;
      onChunk({ receivedBytes, totalBytes });
    }

    const merged = new Uint8Array(receivedBytes);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  }

  const buffer = await res.arrayBuffer();
  return new Uint8Array(buffer);
}

/**
 * Instantiates the Tltk class adapting to either { names, datas } object shape
 * or (names, datas) argument list.
 */
function createTltk(TltkClass, names, datas) {
  if (TltkClass.length === 2) {
    return new TltkClass(names, datas);
  }
  try {
    return new TltkClass({ names, datas });
  } catch (err) {
    return new TltkClass(names, datas);
  }
}

/**
 * Attaches blobs to an existing Tltk instance adapting to both signature forms.
 */
function attachBlobs(tltkInstance, names, datas) {
  if (tltkInstance.attach && tltkInstance.attach.length === 2) {
    return tltkInstance.attach(names, datas);
  }
  try {
    return tltkInstance.attach({ names, datas });
  } catch (err) {
    return tltkInstance.attach(names, datas);
  }
}

/**
 * Manages the lifecycle, tiered fetching, and inference invocation of the TLTK WASM engine.
 */
export class TltkManager {
  constructor(options = {}) {
    this.wasmDir = options.wasmDir || './wasm';
    this.dataDir = options.dataDir || 'data';
    this.version = options.version !== undefined ? options.version : DEFAULT_VERSION;
    this.forceSimd = options.forceSimd !== undefined ? options.forceSimd : null;

    this.flavor = null;
    this.wasmMod = null;
    this.tltk = null;
    this.tier1Loaded = false;
    this.tier2Loaded = false;
    this.tier2Error = null;

    this._tier1Promise = null;
    this._tier2Promise = null;
  }

  /**
   * Builds the asset URL with optional cache-busting query parameter.
   */
  getAssetUrl(filename) {
    const base = `${this.dataDir}/${filename}`;
    return this.version ? `${base}?v=${this.version}` : base;
  }

  /**
   * Initializes the WASM module matching the selected runtime flavor (simd or scalar).
   */
  async initModule() {
    if (this.wasmMod) return this.wasmMod;

    this.flavor = selectFlavor(this.forceSimd);
    const jsPath = `${this.wasmDir}/${this.flavor}/tltk_web.js`;

    let mod;
    try {
      if (typeof document !== 'undefined') {
        // Browser: a bare/relative specifier resolves against THIS module's URL
        // (/js/), not the page root — resolve against the document base instead.
        mod = await import(new URL(jsPath, document.baseURI).href);
      } else {
        mod = await import(jsPath);
      }
    } catch (err) {
      // In node, a dynamic import of a relative path may need an explicit ./ prefix
      if (typeof document === 'undefined' && !jsPath.startsWith('.') && !jsPath.startsWith('/') && !jsPath.startsWith('http')) {
        mod = await import(`./${jsPath}`);
      } else {
        throw err;
      }
    }

    // Initialize wasm-bindgen target web output
    if (typeof mod.default === 'function') {
      const wasmPath = `${this.wasmDir}/${this.flavor}/tltk_web.wasm`;
      if (typeof window === 'undefined' && !wasmPath.startsWith('http://') && !wasmPath.startsWith('https://')) {
        const fs = await import('node:fs/promises');
        const wasmBuf = await fs.readFile(wasmPath);
        await mod.default(wasmBuf);
      } else {
        await mod.default(wasmPath);
      }
    }

    this.wasmMod = mod;
    return mod;
  }

  /**
   * Loads Tier-1 models and creates the persistent Tltk instance.
   * @param {Function} [onProgress] - Callback: ({ phase, fileIndex, totalFiles, fileName, percent })
   */
  async ensureTier1(onProgress = null) {
    if (this.tier1Loaded && this.tltk) return this.tltk;
    if (this._tier1Promise) return this._tier1Promise;

    this._tier1Promise = (async () => {
      await this.initModule();

      const names = [];
      const datas = [];
      const total = TIER1_FILES.length;

      for (let i = 0; i < total; i++) {
        const file = TIER1_FILES[i];
        if (onProgress) {
          onProgress({
            phase: 'tier1',
            fileIndex: i,
            totalFiles: total,
            fileName: file,
            percent: Math.round((i / total) * 100)
          });
        }

        const url = this.getAssetUrl(file);
        const data = await fetchAsset(url);
        names.push(file);
        datas.push(data);
      }

      if (onProgress) {
        onProgress({
          phase: 'tier1',
          fileIndex: total,
          totalFiles: total,
          fileName: 'complete',
          percent: 100
        });
      }

      const TltkClass = this.wasmMod.Tltk;
      if (!TltkClass) {
        throw new Error('Export Tltk class not found in loaded WASM module');
      }

      this.tltk = createTltk(TltkClass, names, datas);
      this.tier1Loaded = true;
      return this.tltk;
    })();

    return this._tier1Promise;
  }

  /**
   * Loads Tier-2 models and attaches them to the existing Tltk instance.
   * If Tier-2 loading fails, logs/stores the error without breaking Tier-1 functionality.
   * @param {Function} [onProgress] - Callback: ({ phase, fileIndex, totalFiles, fileName, percent })
   */
  async ensureTier2(onProgress = null) {
    if (this.tier2Loaded) return true;
    if (this._tier2Promise) return this._tier2Promise;

    // Must have Tier-1 loaded first
    await this.ensureTier1();

    this._tier2Promise = (async () => {
      try {
        const names = [];
        const datas = [];
        const total = TIER2_FILES.length;

        for (let i = 0; i < total; i++) {
          const file = TIER2_FILES[i];
          if (onProgress) {
            onProgress({
              phase: 'tier2',
              fileIndex: i,
              totalFiles: total,
              fileName: file,
              percent: Math.round((i / total) * 100)
            });
          }

          const url = this.getAssetUrl(file);
          const data = await fetchAsset(url);
          names.push(file);
          datas.push(data);
        }

        if (onProgress) {
          onProgress({
            phase: 'tier2',
            fileIndex: total,
            totalFiles: total,
            fileName: 'complete',
            percent: 100
          });
        }

        attachBlobs(this.tltk, names, datas);
        this.tier2Loaded = true;
        this.tier2Error = null;
        return true;
      } catch (err) {
        this.tier2Error = err;
        this._tier2Promise = null; // Allow retry on subsequent calls
        throw err;
      }
    })();

    return this._tier2Promise;
  }

  /**
   * Processes a task through the TLTK WASM session.
   * Automatically determines whether Tier-2 is required and loads it if needed.
   *
   * @param {string} functionName - One of the 16 TLTK functions
   * @param {string} text - Input text
   * @param {string} [cutoff] - Cutoff score for similar_words ('low'|'medium'|'high')
   * @param {Function} [onProgress] - Progress callback
   * @returns {Promise<string>} The result string (text or IMAGE:<base64>)
   */
  async process(functionName, text, cutoff = 'medium', onProgress = null) {
    await this.ensureTier1(onProgress);

    if (TIER2_FUNCTIONS.has(functionName) && !this.tier2Loaded) {
      await this.ensureTier2(onProgress);
    }

    return this.tltk.process(functionName, text, cutoff);
  }
}
