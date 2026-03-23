const express = require("express");
const { notificationConfig } = require("../notifications/config");
const {
  upsertSubscription,
  deactivateSubscriptionByEndpoint,
  savePreferences,
} = require("../notifications/subscriptionService");

const router = express.Router();

function ensureService(res) {
  if (notificationConfig.enabled) return true;
  res.status(503).json({
    ok: false,
    message: "Notification service not configured on backend",
  });
  return false;
}

router.post("/subscriptions", async (req, res, next) => {
  if (!ensureService(res)) return;

  try {
    const { userId, subscription, deviceInfo } = req.body || {};
    await upsertSubscription({ userId, subscription, deviceInfo });
    res.json({ ok: true, message: "Subscription saved" });
  } catch (err) {
    next(err);
  }
});

router.delete("/subscriptions", async (req, res, next) => {
  if (!ensureService(res)) return;

  try {
    const { endpoint } = req.body || {};
    await deactivateSubscriptionByEndpoint(endpoint);
    res.json({ ok: true, message: "Subscription deactivated" });
  } catch (err) {
    next(err);
  }
});

router.post("/preferences", async (req, res, next) => {
  if (!ensureService(res)) return;

  try {
    const { userId, enabled, dailyCount, timezone } = req.body || {};
    await savePreferences({ userId, enabled, dailyCount, timezone });
    res.json({ ok: true, message: "Preferences saved" });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
