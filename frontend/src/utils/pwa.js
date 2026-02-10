// Service Worker registration and PWA utilities

export const registerSW = async () => {
  if ('serviceWorker' in navigator) {
    try {
      const registration = await navigator.serviceWorker.register('/sw.js');
      
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing;
        if (newWorker) {
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'activated') {
              // New content is available, refresh the page
              if (confirm('New version available! Refresh to update?')) {
                window.location.reload();
              }
            }
          });
        }
      });
      
      console.log('SW registered: ', registration);
      return registration;
    } catch (registrationError) {
      console.log('SW registration failed: ', registrationError);
      return null;
    }
  }
  return null;
};

export const unregisterSW = async () => {
  if ('serviceWorker' in navigator) {
    const registration = await navigator.serviceWorker.ready;
    await registration.unregister();
  }
};

export const isStandalone = () => {
  return window.matchMedia('(display-mode: standalone)').matches ||
         window.navigator.standalone === true;
};

export const isInstallable = () => {
  return !isStandalone() && 'serviceWorker' in navigator;
};

export const requestNotificationPermission = async () => {
  if ('Notification' in window) {
    const permission = await Notification.requestPermission();
    return permission === 'granted';
  }
  return false;
};

export const subscribeUserToPush = async (registration, publicKey) => {
  try {
    const subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: publicKey
    });
    
    console.log('User is subscribed:', subscription);
    return subscription;
  } catch (error) {
    console.log('Failed to subscribe user: ', error);
    return null;
  }
};

export const checkForUpdate = async () => {
  if ('serviceWorker' in navigator) {
    const registration = await navigator.serviceWorker.ready;
    await registration.update();
  }
};

// Utility to track app usage for analytics
export const trackAppInstall = () => {
  if (window.gtag) {
    window.gtag('event', 'app_install', {
      event_category: 'PWA',
      event_label: 'App installed successfully'
    });
  }
};

// Network status detection
export const getNetworkStatus = () => {
  return {
    online: navigator.onLine,
    connection: navigator.connection || navigator.mozConnection || navigator.webkitConnection
  };
};

// Cache management utilities
export const clearAppCache = async () => {
  if ('caches' in window) {
    const cacheNames = await caches.keys();
    await Promise.all(
      cacheNames.map(name => caches.delete(name))
    );
  }
};

export const updateCache = async (cacheName, urls) => {
  if ('caches' in window) {
    const cache = await caches.open(cacheName);
    await cache.addAll(urls);
  }
};