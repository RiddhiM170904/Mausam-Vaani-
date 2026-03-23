const express = require("express");
const { notificationConfig } = require("../notifications/config");
const { runTick } = require("../notifications/schedulerService");
const { sendPush } = require("../notifications/pushService");

const router = express.Router();

function ensureService(res) {
  if (notificationConfig.enabled) return true;
  res.status(503).json({
    ok: false,
    message: "Notification service not configured on backend",
  });
  return false;
}

router.post("/notifications/test", async (req, res, next) => {
  if (!ensureService(res)) return;

  try {
    const { subscription, title, body, data } = req.body || {};
    const payload = {
      title: title || "Mausam Vaani Test",
      body: body || "Test notification from unified Backend",
      data: data || { url: "/alerts", source: "test" },
    };

    const result = await sendPush({ subscription, payload });
    if (!result.ok) {
      return res.status(400).json({ ok: false, message: "Push failed" });
    }

    return res.json({ ok: true, message: "Push sent" });
  } catch (err) {
    next(err);
  }
});

router.post("/notifications/run-now", async (req, res, next) => {
  if (!ensureService(res)) return;

  try {
    await runTick();
    res.json({ ok: true, message: "Scheduler run completed" });
  } catch (err) {
    next(err);
  }
});

router.get("/notifications/status", (req, res) => {
  res.json({
    ok: true,
    enabled: notificationConfig.enabled,
    timezone: notificationConfig.timezone,
  });
});

module.exports = router;
