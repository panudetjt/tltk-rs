// tltk-toolbox service worker — offline-first PWA shell + model precache.
// Strategy: precache the app shell at install; runtime cache-first for
// same-origin GETs (network fallback); /data/ and /wasm/ cache keys are
// query-stripped so the background prefetch (bare URLs) and the loader
// (versioned ?v= URLs) share one cache entry.
//
// Cache busting: navigation is network-first, so a new deploy ships a new
// index.html which pulls fresh app.js (its ?v= link changes) and the browser
// re-fetches sw.js itself on navigation (byte-diff = update). If model blobs
// ever change in place, bump DATA_REV to orphan the old entries.
const CACHE_PREFIX = 'tltk-toolbox-';
const CACHE = CACHE_PREFIX + 'v1';
const DATA_REV = 'r1';

const SHELL = [
  './',
  './css/style.css',
  './js/app.js',
  './js/loader.js',
  './js/probe.js',
  './manifest.webmanifest',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith(CACHE_PREFIX) && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

function cacheKey(url) {
  const u = new URL(url);
  if (u.pathname.startsWith('/data/') || u.pathname.startsWith('/wasm/')) {
    u.search = '';
  }
  return u.toString();
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // Navigations: network-first so deploys are picked up; offline falls back
  // to the cached shell.
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put('./', copy));
          return res;
        })
        .catch(() => caches.match('./')),
    );
    return;
  }

  // Everything else: cache-first, fill on miss, query-stripped for data/wasm.
  const key = cacheKey(req.url);
  event.respondWith(
    caches.match(key).then((hit) => {
      if (hit) return hit;
      return fetch(req).then((res) => {
        if (res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(key, copy));
        }
        return res;
      });
    }),
  );
});
