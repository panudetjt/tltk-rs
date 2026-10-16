/**
 * probe.js - Runtime WebAssembly SIMD capability detector
 *
 * Checks whether the current JavaScript environment / browser engine supports
 * WebAssembly 128-bit SIMD (Fixed-width SIMD / simd128).
 *
 * Inlines the canonical ~29-byte SIMD validation test module (wasm-feature-detect technique).
 * Caches the result in localStorage to skip re-validation on subsequent visits.
 */

// Minimal valid WebAssembly module containing a v128 instruction (i8x16.splat)
// Magic (4) + Version (4) + Type section (5) + Func section (2) + Code section (8) = 29 bytes
export const SIMD_PROBE_BYTES = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, // \0asm
  0x01, 0x00, 0x00, 0x00, // version 1
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, // Type section: 1 type, func () -> v128 (0x7b)
  0x03, 0x02, 0x01, 0x00, // Function section: 1 function, sig 0
  0x0a, 0x08, 0x01, 0x06, 0x00, 0x41, 0x00, 0xfd, 0x0f, 0x0b // Code: i32.const 0, i8x16.splat, end
]);

export const SIMD_CACHE_KEY = 'tltk_wasm_simd_supported';

/**
 * Probes the runtime for WebAssembly SIMD support.
 * @param {Storage|null} [storage] - Optional storage interface (defaults to global localStorage if present)
 * @returns {boolean} True if WebAssembly SIMD is supported, false otherwise
 */
export function probeSimd(storage = (typeof localStorage !== 'undefined' ? localStorage : null)) {
  if (storage) {
    try {
      const cached = storage.getItem(SIMD_CACHE_KEY);
      if (cached === 'true') return true;
      if (cached === 'false') return false;
    } catch {
      // Storage might be restricted (e.g. private mode, cookies disabled)
    }
  }

  let supported = false;
  try {
    if (typeof WebAssembly === 'object' && typeof WebAssembly.validate === 'function') {
      supported = WebAssembly.validate(SIMD_PROBE_BYTES);
    }
  } catch {
    supported = false;
  }

  if (storage) {
    try {
      storage.setItem(SIMD_CACHE_KEY, String(supported));
    } catch {
      // Ignore quota/security storage errors
    }
  }

  return supported;
}

/**
 * Selects the wasm build flavor ('simd' or 'scalar').
 * @param {boolean|null} [forceSimd] - If boolean, forces true (simd) or false (scalar)
 * @param {Storage|null} [storage] - Optional storage interface
 * @returns {'simd'|'scalar'} Flavor name
 */
export function selectFlavor(forceSimd = null, storage = (typeof localStorage !== 'undefined' ? localStorage : null)) {
  if (forceSimd !== null && typeof forceSimd === 'boolean') {
    return forceSimd ? 'simd' : 'scalar';
  }
  return probeSimd(storage) ? 'simd' : 'scalar';
}
