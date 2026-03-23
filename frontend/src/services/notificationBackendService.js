import { registerSW } from "../utils/pwa";

const DEFAULT_BACKEND_URL = "http://localhost:5000";
const DEFAULT_TIMEZONE = "Asia/Kolkata";

function getBackendBaseUrl() {
  const configured =
    import.meta.env.VITE_NOTIFICATION_BACKEND_URL || DEFAULT_BACKEND_URL;
  return String(configured).replace(/\/+$/, "");
}

function getVapidPublicKey() {
  return (
    import.meta.env.VITE_NOTIFICATION_VAPID_PUBLIC_KEY ||
    import.meta.env.VITE_VAPID_PUBLIC_KEY ||
    ""
  );
}

function urlBase64ToUint8Array(base64String) {
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding)
    .replace(/-/g, "+")
    .replace(/_/g, "/");
  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);
  for (let i = 0; i < rawData.length; i += 1) {
    outputArray[i] = rawData.charCodeAt(i);
  }
  return outputArray;
}

function normalizeSubscription(subscription) {
  if (!subscription) return null;
  if (typeof subscription.toJSON === "function") {
    return subscription.toJSON();
  }
  return subscription;
}

function getDeviceInfo() {
  return navigator?.userAgent || "Unknown Device";
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${getBackendBaseUrl()}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const text = await response.text();
  const data = text ? JSON.parse(text) : {};

  if (!response.ok) {
    throw new Error(data?.message || "Notification backend request failed");
  }

  return data;
}

async function getOrRegisterServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    throw new Error("Service Worker is not supported in this browser");
  }

  let registration = await navigator.serviceWorker.getRegistration("/sw.js");
  if (!registration) {
    registration = await registerSW();
  }

  if (!registration) {
    throw new Error("Unable to initialize service worker registration");
  }

  return registration;
}

export async function ensurePushSubscription() {
  const registration = await getOrRegisterServiceWorker();
  let subscription = await registration.pushManager.getSubscription();

  if (subscription) {
    return subscription;
  }

  if (!("Notification" in window)) {
    throw new Error("Notifications are not supported in this browser");
  }

  if (Notification.permission !== "granted") {
    const permission = await Notification.requestPermission();
    if (permission !== "granted") {
      return null;
    }
  }

  const vapidPublicKey = getVapidPublicKey();
  if (!vapidPublicKey) {
    throw new Error(
      "Missing VITE_NOTIFICATION_VAPID_PUBLIC_KEY in frontend environment"
    );
  }

  subscription = await registration.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: urlBase64ToUint8Array(vapidPublicKey),
  });

  return subscription;
}

export async function registerPushSubscription({ userId }) {
  if (!userId) {
    throw new Error("Missing userId for push subscription registration");
  }

  const subscription = await ensurePushSubscription();
  if (!subscription) {
    return { ok: false, reason: "permission_denied" };
  }

  await requestJson("/api/subscriptions", {
    method: "POST",
    body: JSON.stringify({
      userId,
      subscription: normalizeSubscription(subscription),
      deviceInfo: getDeviceInfo(),
    }),
  });

  return { ok: true, subscription: normalizeSubscription(subscription) };
}

export async function saveBackendNotificationPreference({
  userId,
  enabled,
  dailyCount,
  timezone,
}) {
  if (!userId) {
    throw new Error("Missing userId for notification preferences");
  }

  await requestJson("/api/preferences", {
    method: "POST",
    body: JSON.stringify({
      userId,
      enabled: Boolean(enabled),
      dailyCount: Number(dailyCount) >= 9 ? 9 : 8,
      timezone: timezone || DEFAULT_TIMEZONE,
    }),
  });

  return { ok: true };
}

export async function unregisterPushSubscription() {
  if (!("serviceWorker" in navigator)) {
    return { ok: true, reason: "unsupported" };
  }

  const registration = await navigator.serviceWorker.getRegistration("/sw.js");
  if (!registration) {
    return { ok: true, reason: "no_registration" };
  }

  const subscription = await registration.pushManager.getSubscription();
  if (!subscription) {
    return { ok: true, reason: "no_subscription" };
  }

  const endpoint = subscription.endpoint;
  await subscription.unsubscribe();

  if (endpoint) {
    await requestJson("/api/subscriptions", {
      method: "DELETE",
      body: JSON.stringify({ endpoint }),
    });
  }

  return { ok: true };
}

export async function syncPushSubscriptionIfAvailable({ userId }) {
  if (!userId) return { ok: false, reason: "missing_user" };
  if (!("serviceWorker" in navigator) || !("Notification" in window)) {
    return { ok: false, reason: "unsupported" };
  }
  if (Notification.permission !== "granted") {
    return { ok: false, reason: "permission_not_granted" };
  }

  const registration = await getOrRegisterServiceWorker();
  const subscription = await registration.pushManager.getSubscription();
  if (!subscription) {
    return { ok: false, reason: "no_subscription" };
  }

  await requestJson("/api/subscriptions", {
    method: "POST",
    body: JSON.stringify({
      userId,
      subscription: normalizeSubscription(subscription),
      deviceInfo: getDeviceInfo(),
    }),
  });

  return { ok: true };
}
