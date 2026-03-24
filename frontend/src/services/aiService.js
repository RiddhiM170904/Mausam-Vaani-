// frontend/src/services/aiService.js
// Mausam Vaani — AI Service
// Three functions to call the AI backend routes

const BASE_URL = (
  import.meta.env.VITE_BACKEND_URL ||
  import.meta.env.VITE_NOTIFICATION_BACKEND_URL ||
  "http://localhost:5000"
).replace(/\/+$/, "");

// ── 1. Get AI prediction (hyperlocal + nowcast + crowd) ──
export async function getAIPrediction(weatherData, lat, lon) {
  try {
    const body = {
      temp:     weatherData?.current?.temp     ?? 0,
      humidity: weatherData?.current?.humidity ?? 0,
      wind:     weatherData?.current?.wind     ?? 0,
      rain:     weatherData?.hourly?.[0]?.pop  ?? weatherData?.hourly?.[0]?.rainProbability ?? 0,
      lat,
      lon,
    };

    const res = await fetch(`${BASE_URL}/api/ai/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("[aiService] getAIPrediction failed:", err.message);
    return null;
  }
}

// ── 2. Submit a crowd report ──
export async function submitCrowdReport(lat, lon, isRaining) {
  try {
    const res = await fetch(`${BASE_URL}/api/ai/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, isRaining }),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("[aiService] submitCrowdReport failed:", err.message);
    return null;
  }
}

// ── 3. Get nearby crowd reports ──
export async function getNearbyReports() {
  try {
    const res = await fetch(`${BASE_URL}/api/ai/reports`);
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("[aiService] getNearbyReports failed:", err.message);
    return { reports: [], count: 0 };
  }
}
