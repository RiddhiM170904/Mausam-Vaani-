export function buildNotifications(weather, userProfile) {
  const alerts = [];
  const userType = String(userProfile?.user_type || 'general').toLowerCase();

  if (Number(weather?.rain_probability) >= 0.6) {
    alerts.push({
      type: 'rain_alert',
      priority: 'high',
      condition: 'rain_probability > 0.6',
      message: 'High rain probability. Carry rain gear and avoid non-urgent travel.',
    });
  }

  if (Number(weather?.aqi) > 150) {
    alerts.push({
      type: 'aqi_alert',
      priority: 'high',
      condition: 'aqi > 150',
      message: 'Air quality is unhealthy. Reduce prolonged outdoor exposure.',
    });
  }

  if (Number(weather?.temp) >= 36) {
    alerts.push({
      type: 'heat_alert',
      priority: 'medium',
      condition: 'temp >= 36',
      message: 'Heat stress risk is high. Hydrate and avoid peak afternoon sun.',
    });
  }

  if ((userType === 'driver' || userType === 'delivery') && Number(weather?.wind_speed) >= 14) {
    alerts.push({
      type: 'wind_travel_warning',
      priority: 'medium',
      condition: 'driver_or_delivery + wind_speed >= 14',
      message: 'Strong winds can impact travel safety. Reduce speed and stay alert.',
    });
  }

  return alerts;
}
