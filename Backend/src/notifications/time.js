function pad2(n) {
  return String(n).padStart(2, "0");
}

function todayDateKey(now = new Date()) {
  return `${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())}`;
}

function parseHHMM(value) {
  if (typeof value !== "string") return null;
  const [hRaw, mRaw] = value.split(":");
  const h = Number(hRaw);
  const m = Number(mRaw);
  if (!Number.isInteger(h) || !Number.isInteger(m)) return null;
  if (h < 0 || h > 23 || m < 0 || m > 59) return null;
  return h * 60 + m;
}

function formatHHMM(minutes) {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${pad2(h)}:${pad2(m)}`;
}

function hashSeed(input) {
  const text = String(input || "mv");
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

function uniqueSortedTimes(times) {
  const set = new Set();
  const out = [];
  for (const t of times) {
    const mins = parseHHMM(t);
    if (mins == null || set.has(mins)) continue;
    set.add(mins);
    out.push(mins);
  }
  out.sort((a, b) => a - b);
  return out.map(formatHHMM);
}

function buildDailyRandomSlots({ userId, dateKey, dailyCount }) {
  const safeCount = Number(dailyCount) >= 9 ? 9 : 8;
  const ranges = {
    morning: [6 * 60 + 30, 10 * 60 + 30],
    afternoon: [12 * 60, 15 * 60 + 30],
    evening: [17 * 60, 20 * 60],
    night: [20 * 60 + 30, 22 * 60 + 30],
  };

  const distribution = safeCount === 9
    ? { morning: 3, afternoon: 2, evening: 2, night: 2 }
    : { morning: 2, afternoon: 2, evening: 2, night: 2 };

  const list = [];
  for (const bucket of Object.keys(distribution)) {
    const rng = seededRandomFactory(`${userId}:${dateKey}:${bucket}:${safeCount}`);
    const [start, end] = ranges[bucket];
    const span = end - start + 1;
    for (let i = 0; i < distribution[bucket]; i += 1) {
      const mins = start + Math.floor(rng() * span);
      list.push(formatHHMM(mins));
    }
  }

  return uniqueSortedTimes(list).slice(0, safeCount);
}

function pickDueSlot(slots, sentMap, now = new Date(), graceMs = 2 * 60 * 60 * 1000) {
  const due = [];
  for (const time of slots) {
    if (sentMap[time]) continue;
    const mins = parseHHMM(time);
    if (mins == null) continue;
    const slotDate = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0);
    slotDate.setMinutes(mins);
    const ts = slotDate.getTime();
    const nowMs = now.getTime();
    if (nowMs < ts) continue;
    if (nowMs - ts > graceMs) continue;
    due.push({ time, ts });
  }

  due.sort((a, b) => b.ts - a.ts);
  return due[0] || null;
}

module.exports = {
  todayDateKey,
  buildDailyRandomSlots,
  pickDueSlot,
};
