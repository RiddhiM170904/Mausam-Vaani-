const webpush = require("web-push");
const { notificationConfig } = require("./config");

if (notificationConfig.enabled) {
  webpush.setVapidDetails(
    notificationConfig.vapidSubject,
    notificationConfig.vapidPublicKey,
    notificationConfig.vapidPrivateKey
  );
}

module.exports = { webpush };
