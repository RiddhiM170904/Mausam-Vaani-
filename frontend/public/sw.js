const CACHE_NAME = 'mausam-vaani-v3';
const urlsToCache = [
  '/manifest.json',
  '/icons/icon-192x192.png',
  '/icons/icon-512x512.png'
];

// Install event - cache resources
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Opened cache');
        return cache.addAll(urlsToCache);
      })
  );
  self.skipWaiting();
});

// Fetch event - serve from cache when offline
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') {
    return;
  }

  const requestUrl = new URL(event.request.url);
  if (requestUrl.origin !== self.location.origin) {
    return;
  }

  event.respondWith(
    (async () => {
      const isNavigationRequest =
        event.request.mode === 'navigate' ||
        requestUrl.pathname === '/' ||
        requestUrl.pathname.endsWith('/index.html');

      // Always prefer network for HTML/navigation so new deployments don't keep stale hashed asset references.
      if (isNavigationRequest) {
        try {
          const freshPage = await fetch(event.request);
          const cache = await caches.open(CACHE_NAME);
          cache.put('/', freshPage.clone());
          return freshPage;
        } catch {
          const cachedPage = await caches.match('/') || await caches.match('/index.html');
          if (cachedPage) return cachedPage;
          return Response.error();
        }
      }

      if (requestUrl.pathname === '/manifest.json') {
        try {
          const fresh = await fetch(event.request);
          const cache = await caches.open(CACHE_NAME);
          cache.put(event.request, fresh.clone());
          return fresh;
        } catch {
          const cachedManifest = await caches.match(event.request);
          if (cachedManifest) return cachedManifest;
          return Response.error();
        }
      }

      const cached = await caches.match(event.request);
      if (cached) {
        return cached;
      }

      try {
        const fresh = await fetch(event.request);
        if (fresh && fresh.ok && /\.(?:js|css|png|jpg|jpeg|svg|webp|ico)$/i.test(requestUrl.pathname)) {
          const cache = await caches.open(CACHE_NAME);
          cache.put(event.request, fresh.clone());
        }
        return fresh;
      } catch {
        return Response.error();
      }
    })()
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Background sync for offline data
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    event.waitUntil(doBackgroundSync());
  }
});

function doBackgroundSync() {
  // Implement background sync logic here
  return Promise.resolve();
}

// Push notification handling
self.addEventListener('push', (event) => {
  let payload = {};
  if (event.data) {
    try {
      payload = event.data.json();
    } catch {
      payload = { body: event.data.text() };
    }
  }

  const title = payload?.title || 'Mausam Vaani';
  const options = {
    body: payload?.body || 'New weather update available!',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/icon-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: '2',
      ...(payload?.data || {}),
    },
    actions: [
      {
        action: 'explore',
        title: 'View Weather',
        icon: '/icons/icon-192x192.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/icons/icon-192x192.png'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

// Notification click handling
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  const targetUrl = event.notification?.data?.url || '/';

  if (event.action === 'explore') {
    event.waitUntil(clients.openWindow(targetUrl));
    return;
  }

  event.waitUntil(clients.openWindow(targetUrl));
});