import { insightService } from './insightService';
import dayjs from 'dayjs';
import customParseFormat from 'dayjs/plugin/customParseFormat';

dayjs.extend(customParseFormat);

const CONFIG_KEY = 'mv_local_notification_config_v1';
const DELIVERY_LOG_KEY = 'mv_local_notification_delivery_v1';
const AUTO_PERMISSION_PROMPT_KEY = 'mv_notification_auto_prompted_v1';
const DEFAULT_TIMEZONE = 'Asia/Kolkata';
const ENGINE_INTERVAL_MS = 15 * 60 * 1000;
const SLOT_GRACE_MS = 2 * 60 * 60 * 1000;

const DEFAULT_CONFIG = {
  enabled: true,
  dailyCount: 9,
  timezone: DEFAULT_TIMEZONE,
  scheduleDate: '',
  mode: 'personalized_random',
};

function pad2(n) {
  return String(n).padStart(2, '0');
}

function parseTimeToMinutes(t) {
  if (typeof t !== 'string') return null;
  const parts = t.split(':');
  if (parts.length !== 2) return null;
  const h = Number(parts[0]);
  const m = Number(parts[1]);
  if (!Number.isInteger(h) || !Number.isInteger(m)) return null;
  if (h < 0 || h > 23 || m < 0 || m > 59) return null;
  return h * 60 + m;
}

function formatMinutesToHHMM(minutes) {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${pad2(h)}:${pad2(m)}`;
}

function uniqueSortedTimes(times) {
  const seen = new Set();
  const normalized = [];
  for (const t of times) {
    const mins = parseTimeToMinutes(t);
    if (mins == null) continue;
    if (seen.has(mins)) continue;
    seen.add(mins);
    normalized.push(mins);
  }
  normalized.sort((a, b) => a - b);
  return normalized.map(formatMinutesToHHMM);
}

function hashSeed(input) {
  const text = String(input || 'mv');
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24);
  }
  return h >>> 0;
}

function seededRandomFactory(seedText) {
  let x = hashSeed(seedText) || 123456789;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return ((x >>> 0) % 1000000) / 1000000;
  };
}

function randomMinuteInRange(rng, startMin, endMin) {
  const span = Math.max(1, endMin - startMin + 1);
  return startMin + Math.floor(rng() * span);
}

function buildRandomBucketTimes(count, seedBase = todayDateKey()) {
  const safeCount = count >= 9 ? 9 : 8;
  const bucketRanges = {
    morning: [6 * 60 + 30, 10 * 60 + 30],
    afternoon: [12 * 60, 15 * 60 + 30],
    evening: [17 * 60, 20 * 60],
    night: [20 * 60 + 30, 22 * 60 + 30],
  };

  const distribution = safeCount === 9
    ? { morning: 3, afternoon: 2, evening: 2, night: 2 }
    : { morning: 2, afternoon: 2, evening: 2, night: 2 };

  const allTimes = [];
  Object.entries(distribution).forEach(([bucket, bucketCount]) => {
    const [start, end] = bucketRanges[bucket];
    const rng = seededRandomFactory(`${seedBase}:${bucket}:${safeCount}`);
    for (let i = 0; i < bucketCount; i += 1) {
      allTimes.push(formatMinutesToHHMM(randomMinuteInRange(rng, start, end)));
    }
  });

  return uniqueSortedTimes(allTimes).slice(0, safeCount);
}

function todayDateKey() {
  return dayjs().format('YYYY-MM-DD');
}

function getSlotTimeMsForToday(hhmm) {
  if (parseTimeToMinutes(hhmm) == null) return null;
  const parsed = dayjs(`${todayDateKey()} ${hhmm}`, 'YYYY-MM-DD HH:mm', true);
  if (!parsed.isValid()) return null;
  return parsed.valueOf();
}

function readJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function writeJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore localStorage failures.
  }
}

function weatherSummary(weatherData) {
  const current = weatherData?.current || {};
  const temp = Number(current?.temp ?? current?.temperature ?? 0);
  const cond = String(current?.condition || current?.weather?.main || current?.weather?.[0]?.main || 'weather').toLowerCase();
  const rainProb = Number(weatherData?.hourly?.[0]?.precip_probability ?? weatherData?.hourly?.[0]?.rain_probability ?? 0);

  if (rainProb >= 0.5 || cond.includes('rain') || cond.includes('storm')) {
    return 'Aaj baarish ka chance high hai 🌧️ Umbrella leke niklo aur travel thoda early plan karo.';
  }
  if (temp >= 30) {
    return 'Aaj garmi tez hai ☀️ Paani bottle saath rakho aur 12-4 PM dhoop avoid karo.';
  }
  if (temp <= 12) {
    return 'Aaj thandi hawa rahegi 🧥 Light jacket/cap handy rakhna best rahega.';
  }
  return 'Weather abhi kaafi acha hai 🙂 if kuch plan krna h to mausaam vaani pr dekh ke krna.';
}

function getFriendlyName(user) {
  const full = String(user?.name || user?.full_name || '').trim();
  if (!full) return 'dost';
  return full.split(/\s+/)[0] || 'dost';
}

function buildPersonaHint(user) {
  const persona = String(user?.persona || 'general').toLowerCase();

  if (persona.includes('driver') || persona.includes('delivery')) {
    return 'Road pe nikalne se pehle route + weather quick check kar lena 🚦';
  }
  if (persona.includes('farmer')) {
    return 'Field ka kaam weather window dekh ke plan karo 🌾';
  }
  if (persona.includes('worker')) {
    return 'Outdoor kaam me hydration break zarur lena 💧';
  }
  if (persona.includes('student')) {
    return 'College/school jaate waqt bottle aur weather-ready gear rakhna 🎒';
  }
  if (persona.includes('senior')) {
    return 'Aaj comfort ko priority do aur outdoor exposure short rakho 🤍';
  }
  return 'Aaj ka plan smart timing ke saath rakho ✅';
}

function generalReminderForSlot(slotIndex, weatherData, user) {
  const name = getFriendlyName(user);
  const personaHint = buildPersonaHint(user);
  const reminders = [
    `Good morning ${name} 🌤️ Aaj ka weather check karke day start karo. ${personaHint}`,
    `Hydration reminder ${name} 💧 Bottle refill kar lo, energy steady rahegi.`,
    `Nikalne se pehle quick weather + traffic check kar lo ${name} 🚕`,
    `Afternoon alert ${name} ☀️ Dhoop strong ho to shade break lena ya cap use karna.`,
    `Quick weather break ${name} ⏱️ Rain/wind update dubara check kar lo.`,
    `Evening commute tip ${name} 🌆 Rain chance ho to 10-15 min buffer rakhna.`,
    `Night prep ${name} 🌙 Kal ke liye umbrella/bottle ready rakh do.`,
    `Safety check ${name} 📱 Phone charged rakho aur emergency contact reachable ho.`,
    weatherSummary(weatherData),
  ];
  return reminders[slotIndex % reminders.length];
}

function getApiPermissionState() {
  if (!('Notification' in window)) return 'unsupported';
  return Notification.permission;
}

export function getNotificationConfig() {
  const raw = readJSON(CONFIG_KEY, DEFAULT_CONFIG);
  const dailyCount = raw?.dailyCount >= 9 ? 9 : 8;
  const today = todayDateKey();
  const storedTimes = Array.isArray(raw?.times) ? uniqueSortedTimes(raw.times) : [];
  const shouldRefreshDailySchedule = !storedTimes.length || raw?.scheduleDate !== today;
  const times = shouldRefreshDailySchedule
    ? buildRandomBucketTimes(dailyCount, `${today}:${raw?.timezone || DEFAULT_TIMEZONE}`)
    : storedTimes;

  const config = {
    enabled: Boolean(raw?.enabled),
    dailyCount,
    timezone: raw?.timezone || DEFAULT_TIMEZONE,
    mode: 'personalized_random',
    scheduleDate: today,
    times,
  };

  if (shouldRefreshDailySchedule || raw?.mode !== 'personalized_random') {
    writeJSON(CONFIG_KEY, config);
  }

  return config;
}

export function saveNotificationConfig(partial) {
  const current = getNotificationConfig();
  const nextCount = partial?.dailyCount >= 9 ? 9 : 8;
  const today = todayDateKey();
  const next = {
    ...current,
    ...partial,
    dailyCount: nextCount,
    scheduleDate: today,
    mode: 'personalized_random',
  };

  if (!Array.isArray(partial?.times) || partial?.regenerate) {
    next.times = buildRandomBucketTimes(nextCount, `${today}:${next.timezone}`);
  } else {
    next.times = uniqueSortedTimes(partial.times);
  }

  writeJSON(CONFIG_KEY, next);
  return next;
}

export async function ensureNotificationPermission() {
  if (!('Notification' in window)) return false;
  if (Notification.permission === 'granted') return true;
  const permission = await Notification.requestPermission();
  return permission === 'granted';
}

export async function autoRequestNotificationPermissionOnce() {
  if (!('Notification' in window)) return false;
  if (Notification.permission === 'granted') return true;
  if (Notification.permission === 'denied') return false;

  const alreadyPrompted = readJSON(AUTO_PERMISSION_PROMPT_KEY, false);
  if (alreadyPrompted) return false;

  writeJSON(AUTO_PERMISSION_PROMPT_KEY, true);
  return ensureNotificationPermission();
}

async function showBrowserNotification(title, body, data = {}) {
  const payload = {
    body,
    icon: '/icons/icon-192x192.png',
    badge: '/icons/icon-192x192.png',
    tag: data?.tag || `mv-${Date.now()}`,
    renotify: false,
    data,
  };

  try {
    const registration = await Promise.race([
      navigator.serviceWorker.ready,
      new Promise((resolve) => setTimeout(() => resolve(null), 2500)),
    ]);

    if (registration && typeof registration.showNotification === 'function') {
      await registration.showNotification(title, payload);
      return true;
    }
  } catch {
    // Fallback to direct Notification API below.
  }

  if ('Notification' in window && Notification.permission === 'granted') {
    new Notification(title, payload);
    return true;
  }

  return false;
}

function readDeliveryLog() {
  return readJSON(DELIVERY_LOG_KEY, {});
}

function saveDeliveryLog(log) {
  writeJSON(DELIVERY_LOG_KEY, log);
}

function cleanupDeliveryLog(log) {
  const keys = Object.keys(log || {});
  if (keys.length <= 7) return log;
  const sorted = keys.sort();
  const keep = sorted.slice(-7);
  const next = {};
  for (const k of keep) next[k] = log[k];
  return next;
}

function pickSlotToSend(times, sentMap) {
  const now = Date.now();
  const due = [];
  for (const t of times) {
    const ts = getSlotTimeMsForToday(t);
    if (ts == null) continue;
    if (sentMap[t]) continue;
    if (now < ts) continue;
    if (now - ts > SLOT_GRACE_MS) continue;
    due.push({ time: t, ts });
  }
  if (!due.length) return null;
  due.sort((a, b) => b.ts - a.ts);
  return due[0];
}

function isAiSlot(slotIndex) {
  return slotIndex % 2 === 1;
}

async function createAiMessage({ userId, persona, weatherData, location }) {
  try {
    const result = await insightService.getQuickInsight({
      userId: userId || null,
      persona: persona || 'general',
      weatherData,
      location,
      requirements: 'Create 1 personalized Hinglish notification in WhatsApp tone with 1 emoji. Max 110 chars. Must include one clear action.',
    });

    let text = String(result?.message || result?.recommendation || '').trim();
    if (text && !/[\u{1F300}-\u{1FAFF}]/u.test(text)) {
      text = `${text} ✅`;
    }
    if (text) return text;
  } catch {
    // Fall through to default message.
  }

  return weatherSummary(weatherData);
}

export function getNotificationStatus() {
  return {
    supported: 'Notification' in window && 'serviceWorker' in navigator,
    permission: getApiPermissionState(),
  };
}

export function getNextNotificationTime(config = getNotificationConfig()) {
  const now = dayjs();
  const times = config?.times || [];
  const slots = [];

  for (const t of times) {
    const ts = getSlotTimeMsForToday(t);
    if (ts != null && dayjs(ts).isAfter(now)) {
      slots.push(dayjs(ts));
    }
  }

  if (!slots.length && times.length) {
    const firstTomorrow = getSlotTimeMsForToday(times[0]);
    if (firstTomorrow != null) {
      slots.push(dayjs(firstTomorrow).add(1, 'day'));
    }
  }

  if (!slots.length) return null;
  slots.sort((a, b) => a.valueOf() - b.valueOf());
  return slots[0].toISOString();
}

export function regenerateNotificationSchedule(dailyCount = null) {
  const current = getNotificationConfig();
  const nextCount = dailyCount == null ? current.dailyCount : (Number(dailyCount) >= 9 ? 9 : 8);
  return saveNotificationConfig({ dailyCount: nextCount, regenerate: true });
}

export async function sendTestNotificationNow(message = null) {
  const status = getNotificationStatus();
  if (!status.supported) {
    return { ok: false, reason: 'unsupported' };
  }

  if (status.permission !== 'granted') {
    const granted = await ensureNotificationPermission();
    if (!granted) {
      return { ok: false, reason: 'permission_denied' };
    }
  }

  const ok = await showBrowserNotification(
    'Mausam Vaani Test',
    message || 'Test successful: local notification engine is active on this device.',
    {
      tag: `mv-test-${Date.now()}`,
      url: '/alerts',
      source: 'test',
    }
  );

  return {
    ok,
    reason: ok ? 'sent' : 'delivery_failed',
  };
}

export async function runLocalNotificationTick({ isLoggedIn, user, weatherData, location }) {
  const status = getNotificationStatus();
  if (!status.supported || status.permission !== 'granted') return { sent: false, reason: 'permission' };

  const config = getNotificationConfig();
  if (!config.enabled) return { sent: false, reason: 'disabled' };

  const dateKey = todayDateKey();
  const log = cleanupDeliveryLog(readDeliveryLog());
  const todayMap = log[dateKey] || {};
  const slot = pickSlotToSend(config.times, todayMap);

  if (!slot) {
    log[dateKey] = todayMap;
    saveDeliveryLog(log);
    return { sent: false, reason: 'no_due_slot' };
  }

  const slotIndex = config.times.indexOf(slot.time);
  const useAi = isAiSlot(slotIndex);

  let body = '';
  if (useAi && isLoggedIn) {
    body = await createAiMessage({
      userId: user?.id,
      persona: user?.persona,
      weatherData,
      location,
    });
  }

  if (!body) {
    body = generalReminderForSlot(slotIndex, weatherData, user);
  }

  const delivered = await showBrowserNotification('Mausam Vaani', body, {
    tag: `mv-local-${dateKey}-${slot.time}`,
    url: '/alerts',
    source: useAi && isLoggedIn ? 'ai' : 'general',
    slotTime: slot.time,
  });

  if (!delivered) {
    return { sent: false, reason: 'delivery_failed' };
  }

  // Mark all due slots as consumed so we do not dump backlog notifications.
  const now = Date.now();
  for (const t of config.times) {
    const tMs = getSlotTimeMsForToday(t);
    if (tMs != null && now >= tMs) {
      todayMap[t] = true;
    }
  }

  log[dateKey] = todayMap;
  saveDeliveryLog(log);

  return {
    sent: true,
    slot: slot.time,
    source: useAi && isLoggedIn ? 'ai' : 'general',
  };
}

export const LOCAL_NOTIFICATION_ENGINE_INTERVAL_MS = ENGINE_INTERVAL_MS;
