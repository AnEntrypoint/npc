// A service worker was once registered at this scope and its script is gone, so the
// browser keeps replaying a fetch handler that can only fail. Serving this file lets
// the next byte-check install a worker whose only job is to remove itself.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((k) => caches.delete(k)));
    await self.registration.unregister();
    const clients = await self.clients.matchAll({ type: 'window' });
    for (const c of clients) c.navigate(c.url);
  })());
});
self.addEventListener('fetch', (event) => event.respondWith(fetch(event.request)));
