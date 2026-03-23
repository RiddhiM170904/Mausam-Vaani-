const { webpush } = require("./webpush");
const { notificationConfig } = require("./config");
const { deactivateSubscriptionByEndpoint } = require("./subscriptionService");
const logger = require("./logger");

async function sendPush({ subscription, payload }) {
  if (!notificationConfig.enabled) {
    const err = new Error("Notification service is not configured");
    err.status = 503;
    throw err;
  }

  try {
    await webpush.sendNotification(subscription, JSON.stringify(payload));
    return { ok: true };
  } catch (err) {
    const statusCode = err?.statusCode || 0;
    const endpoint = subscription?.endpoint;

    logger.warn("push_send_failed", {
      statusCode,
      endpoint,
      message: err?.message,
    });

    if (statusCode === 404 || statusCode === 410) {
      try {
        await deactivateSubscriptionByEndpoint(endpoint);
      } catch (deactivateErr) {
        logger.error("deactivate_subscription_failed", {
          endpoint,
          message: deactivateErr?.message,
        });
      }
    }

    return { ok: false, error: err };
  }
}

module.exports = { sendPush };
