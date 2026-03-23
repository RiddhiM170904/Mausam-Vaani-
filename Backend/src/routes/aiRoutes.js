// Backend/routes/aiRoutes.js
// Mausam Vaani — AI Feature Routes
// Handles: hyperlocal adjustment, nowcasting, crowdsourcing, advice

const express = require("express");
const router = express.Router();

// ── In-memory crowd reports store ──
// Persists during server session, auto-clears reports older than 30 mins
let crowdData = [];

function cleanOldReports() {
  const thirtyMinsAgo = Date.now() - 30 * 60 * 1000;
  crowdData = crowdData.filter((r) => r.timestamp > thirtyMinsAgo);
}

function getNearbyReports(lat, lon, radius = 0.05) {
  cleanOldReports();
  return crowdData.filter(
    (r) => Math.abs(r.lat - lat) < radius && Math.abs(r.lon - lon) < radius
  );
}

// ── FEATURE 1: Hyperlocal Adjustment ──
function applyHyperlocal(temp, humidity, lat, lon) {
  const localTemp = parseFloat((temp + (lat % 1) - 0.5).toFixed(2));
  const localHumidity = parseFloat(
    Math.min(100, Math.max(0, humidity + (lon % 1) - 0.5)).toFixed(2)
  );
  return { localTemp, localHumidity };
}

// ── FEATURE 2: Nowcasting ──
function applyNowcasting(rain, localHumidity, wind) {
  let futureRain = rain;
  if (localHumidity > 90) {
    futureRain = Math.min(rain + 35, 100);
  } else if (localHumidity > 80 && wind < 10) {
    futureRain = Math.min(rain + 20, 100);
  }
  return parseFloat(futureRain.toFixed(2));
}

// ── FEATURE 3: Crowd correction ──
function applyCrowdCorrection(futureRain, lat, lon) {
  const nearby = getNearbyReports(lat, lon);
  if (nearby.length === 0) return { futureRain, nearbyReports: 0 };
  const rainingCount = nearby.filter((r) => r.isRaining).length;
  const majority = rainingCount > nearby.length / 2;
  if (majority) futureRain = 85;
  return { futureRain, nearbyReports: nearby.length };
}

// ── FEATURE 4: Advice generation ──
function generateAdvice(futureRain, localHumidity, wind) {
  if (futureRain > 70)
    return "Heavy rain likely. Delay outdoor activity and farming.";
  if (localHumidity > 85)
    return "High humidity. Risk of fungal crop disease. Apply fungicide.";
  if (wind > 25)
    return "Strong winds. Avoid travel on exposed roads.";
  return "Safe conditions. Good time for outdoor activities.";
}

// ─────────────────────────────────────
// POST /api/ai/predict
// Body: { temp, humidity, wind, rain, lat, lon }
// ─────────────────────────────────────
router.post("/predict", (req, res) => {
  try {
    let { temp, humidity, wind, rain, lat, lon } = req.body;

    // Validate
    if ([temp, humidity, wind, rain, lat, lon].some((v) => v == null)) {
      return res.status(400).json({ error: "Missing required fields: temp, humidity, wind, rain, lat, lon" });
    }

    temp = parseFloat(temp);
    humidity = parseFloat(humidity);
    wind = parseFloat(wind);
    rain = parseFloat(rain);
    lat = parseFloat(lat);
    lon = parseFloat(lon);

    // Apply pipeline
    const { localTemp, localHumidity } = applyHyperlocal(temp, humidity, lat, lon);
    let futureRain = applyNowcasting(rain, localHumidity, wind);
    const { futureRain: correctedRain, nearbyReports } = applyCrowdCorrection(futureRain, lat, lon);
    futureRain = correctedRain;
    const advice = generateAdvice(futureRain, localHumidity, wind);

    return res.json({
      localTemp,
      localHumidity,
      futureRain,
      advice,
      nearbyReports,
    });
  } catch (err) {
    console.error("[AI Predict Error]", err.message);
    return res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────
// POST /api/ai/report
// Body: { lat, lon, isRaining }
// ─────────────────────────────────────
router.post("/report", (req, res) => {
  try {
    const { lat, lon, isRaining } = req.body;

    if (lat == null || lon == null || isRaining == null) {
      return res.status(400).json({ error: "Missing fields: lat, lon, isRaining" });
    }

    cleanOldReports();
    crowdData.push({
      lat: parseFloat(lat),
      lon: parseFloat(lon),
      isRaining: Boolean(isRaining),
      timestamp: Date.now(),
    });

    return res.json({ success: true, totalReports: crowdData.length });
  } catch (err) {
    console.error("[AI Report Error]", err.message);
    return res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────
// GET /api/ai/reports
// Returns all reports from last 30 mins
// ─────────────────────────────────────
router.get("/reports", (req, res) => {
  try {
    cleanOldReports();
    return res.json({ reports: crowdData, count: crowdData.length });
  } catch (err) {
    console.error("[AI Reports Error]", err.message);
    return res.status(500).json({ error: err.message });
  }
});

module.exports = router;
