const cron = require("node-cron");
const { notificationConfig } = require("./config");
const logger = require("./logger");
const { buildDailyRandomSlots, pickDueSlot, todayDateKey } = require("./time");
const {
  getActiveTargets,
  getDeliveryLogForDate,
  insertDeliveryLog,
} = require("./subscriptionService");
const { buildNotificationPayload } = require("./messageService");
const { sendPush } = require("./pushService");

let task = null;

async function runTick() {
  if (!notificationConfig.enabled) return;

  const now = new Date();
  const dateKey = todayDateKey(now);

  const targets = await getActiveTargets();
  if (!targets.length) {
    logger.info("scheduler_no_targets");
    return;
  }

  for (const target of targets) {
    const dailyCount = Number(target?.preference?.daily_count || 9) >= 9 ? 9 : 8;
    const slots = buildDailyRandomSlots({
      userId: target.userId,
      dateKey,
      dailyCount,
    });

    const sentMap = await getDeliveryLogForDate({ userId: target.userId, dateKey });
    const dueSlot = pickDueSlot(slots, sentMap, now);
    if (!dueSlot) continue;

    const slotIndex = Math.max(0, slots.indexOf(dueSlot.time));
    const message = buildNotificationPayload({
      slotIndex,
      userProfile: { user_id: target.userId },
      weather: null,
    });

    const payload = {
      title: message.title,
      body: message.body,
      data: {
        url: "/alerts",
        source: message.source,
        slotTime: dueSlot.time,
      },
    };

    const result = await sendPush({
      subscription: target.subscription,
      payload,
    });

    if (result.ok) {
      await insertDeliveryLog({
        userId: target.userId,
        slotDate: dateKey,
        slotTime: dueSlot.time,
        source: message.source,
        title: message.title,
        body: message.body,
      });
      logger.info("scheduler_sent", {
        userId: target.userId,
        slotTime: dueSlot.time,
        source: message.source,
      });
    }
  }
}

function startScheduler() {
  if (!notificationConfig.enabled) {
    logger.warn("scheduler_disabled_missing_env", {
      required: [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "VAPID_SUBJECT",
        "VAPID_PUBLIC_KEY",
        "VAPID_PRIVATE_KEY",
      ],
    });
    return null;
  }

  if (task) return task;

  task = cron.schedule(
    notificationConfig.schedulerCron,
    async () => {
      try {
        await runTick();
      } catch (err) {
        logger.error("scheduler_tick_failed", { message: err?.message });
      }
    },
    { timezone: notificationConfig.timezone }
  );

  logger.info("scheduler_started", {
    timezone: notificationConfig.timezone,
    cron: notificationConfig.schedulerCron,
  });

  return task;
}

module.exports = { startScheduler, runTick };
