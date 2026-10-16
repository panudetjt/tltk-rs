/**
 * probe_test.mjs - Unit verification for runtime WebAssembly SIMD probe logic (AC2)
 *
 * Runs under Node 22 (simd-capable):
 * 1. Verifies that the probe function detects SIMD support and selects 'simd'.
 * 2. Verifies the fallback path by forcing the probe false and selecting 'scalar'.
 * 3. Verifies localStorage caching behavior.
 */

import { probeSimd, selectFlavor, SIMD_CACHE_KEY, SIMD_PROBE_BYTES } from './probe.js';

console.log('=== TLTK WebAssembly SIMD Probe Unit Verification ===\n');

// 1. Check raw byte validation under Node 22
const isValidWasm = WebAssembly.validate(SIMD_PROBE_BYTES);
console.log(`[AC2-1] WebAssembly.validate on probe module: ${isValidWasm}`);
if (!isValidWasm) {
  console.error('FAIL: Expected probe bytes to validate under Node 22');
  process.exit(1);
}

// 2. Default probe under Node 22 (no cache)
const detectedSimd = probeSimd(null);
const defaultFlavor = selectFlavor(null, null);
console.log(`[AC2-2] probeSimd(null) => ${detectedSimd}`);
console.log(`[AC2-3] selectFlavor(null, null) => "${defaultFlavor}"`);
if (!detectedSimd || defaultFlavor !== 'simd') {
  console.error(`FAIL: Expected 'simd' under Node 22, got detected=${detectedSimd}, flavor=${defaultFlavor}`);
  process.exit(1);
}

// 3. Fallback path: force probe false
const forcedScalar = selectFlavor(false, null);
console.log(`[AC2-4] selectFlavor(false, null) [forced false] => "${forcedScalar}"`);
if (forcedScalar !== 'scalar') {
  console.error(`FAIL: Expected 'scalar' when forced false, got "${forcedScalar}"`);
  process.exit(1);
}

// 4. Force true
const forcedSimd = selectFlavor(true, null);
console.log(`[AC2-5] selectFlavor(true, null) [forced true] => "${forcedSimd}"`);
if (forcedSimd !== 'simd') {
  console.error(`FAIL: Expected 'simd' when forced true, got "${forcedSimd}"`);
  process.exit(1);
}

// 5. Cache verification with simulated Storage
const mockStorage = {
  store: {},
  getItem(key) { return this.store[key] ?? null; },
  setItem(key, value) { this.store[key] = String(value); }
};

// First probe populates cache
const probedWithStorage = probeSimd(mockStorage);
const cachedVal = mockStorage.getItem(SIMD_CACHE_KEY);
console.log(`[AC2-6] Initial probe with storage => ${probedWithStorage}, cached: "${cachedVal}"`);
if (cachedVal !== 'true') {
  console.error(`FAIL: Expected cached value 'true', got "${cachedVal}"`);
  process.exit(1);
}

// Override cache to simulate a browser that previously cached 'false'
mockStorage.setItem(SIMD_CACHE_KEY, 'false');
const flavorFromFalseCache = selectFlavor(null, mockStorage);
console.log(`[AC2-7] selectFlavor with cached 'false' => "${flavorFromFalseCache}"`);
if (flavorFromFalseCache !== 'scalar') {
  console.error(`FAIL: Expected 'scalar' from cached false, got "${flavorFromFalseCache}"`);
  process.exit(1);
}

// Override cache to 'true'
mockStorage.setItem(SIMD_CACHE_KEY, 'true');
const flavorFromTrueCache = selectFlavor(null, mockStorage);
console.log(`[AC2-8] selectFlavor with cached 'true' => "${flavorFromTrueCache}"`);
if (flavorFromTrueCache !== 'simd') {
  console.error(`FAIL: Expected 'simd' from cached true, got "${flavorFromTrueCache}"`);
  process.exit(1);
}

console.log('\n[PROBE PROOF GREEN] All probe logic unit tests passed successfully.');
