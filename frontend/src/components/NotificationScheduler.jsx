import { useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import {
  getNotificationConfig,
} from '../services/localNotificationService';
import {
  saveBackendNotificationPreference,
  syncPushSubscriptionIfAvailable,
} from '../services/notificationBackendService';

/**
 * Keeps backend subscription + preference state in sync after login.
 */
export default function NotificationScheduler() {
  const { isLoggedIn, user } = useAuth();

  useEffect(() => {
    let cancelled = false;

    const syncBackendState = async () => {
      if (!isLoggedIn || !user?.id) {
        return;
      }

      const cfg = getNotificationConfig();

      if (cancelled) return;

      try {
        await saveBackendNotificationPreference({
          userId: user.id,
          enabled: cfg.enabled,
          dailyCount: cfg.dailyCount,
          timezone: cfg.timezone,
        });

        if (cfg.enabled) {
          await syncPushSubscriptionIfAvailable({ userId: user.id });
        }
      } catch {
        // Keep sync silent on failures.
      }
    };

    syncBackendState();

    return () => {
      cancelled = true;
    };
  }, [isLoggedIn, user?.id]);

  return null;
}
