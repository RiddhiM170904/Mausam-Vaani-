import { useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import useLocation from '../hooks/useLocation';
import { weatherService } from '../services/weatherService';
import {
  LOCAL_NOTIFICATION_ENGINE_INTERVAL_MS,
  autoRequestNotificationPermissionOnce,
  getNotificationConfig,
  saveNotificationConfig,
  runLocalNotificationTick,
} from '../services/localNotificationService';

/**
 * Background local notification engine.
 * Runs while app is open and triggers scheduled browser notifications.
 */
export default function NotificationScheduler() {
  const { isLoggedIn, user } = useAuth();
  const { location } = useLocation();

  useEffect(() => {
    let cancelled = false;

    const initialize = async () => {
      const cfg = getNotificationConfig();
      if (!cfg.enabled) {
        saveNotificationConfig({ enabled: true });
      }
      await autoRequestNotificationPermissionOnce();
    };

    const tick = async () => {
      if (cancelled) return;
      try {
        let weatherData = null;
        const lat = Number(location?.lat);
        const lon = Number(location?.lon);

        if (Number.isFinite(lat) && Number.isFinite(lon)) {
          weatherData = await weatherService.getFullWeatherCached(lat, lon);
        }

        await runLocalNotificationTick({
          isLoggedIn,
          user,
          weatherData,
          location,
        });
      } catch {
        // Keep scheduler silent on failures.
      }
    };

    initialize();
    tick();

    const id = setInterval(() => {
      if (document.visibilityState === 'visible') {
        tick();
      }
    }, LOCAL_NOTIFICATION_ENGINE_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [isLoggedIn, user?.id, user?.persona, location?.lat, location?.lon, location?.city]);

  return null;
}
