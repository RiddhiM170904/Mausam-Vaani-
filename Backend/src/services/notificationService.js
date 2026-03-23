function buildTriggerNotifications({ weather, userProfile }) {
  const notifications = [];
  const userType = userProfile?.user_type || 'general';

  if (weather.rain_probability >= 0.6) {
    notifications.push({
      type: 'rain_alert',
      priority: 'high',
      title: 'Rain Alert',
      message: `Rain probability is ${Math.round(weather.rain_probability * 100)}%. Plan travel with rain protection.`,
      channels: ['push'],
      condition: 'rain_probability > 0.6',
    });
  }

  if (weather.aqi > 150) {
    notifications.push({
      type: 'aqi_alert',
      priority: 'high',
      title: 'Air Quality Alert',
      message: `AQI is ${weather.aqi}. Limit prolonged outdoor activity and use a mask if needed.`,
      channels: ['push'],
      condition: 'aqi > 150',
    });
  }

  if (weather.temp >= 36) {
    notifications.push({
      type: 'heat_alert',
      priority: 'medium',
      title: 'Heat Stress Warning',
      message: 'High temperature detected. Hydrate frequently and avoid peak afternoon exposure.',
      channels: ['push'],
      condition: 'temp >= 36',
    });
  }

  if (userType === 'driver' && weather.wind_speed >= 14) {
    notifications.push({
      type: 'wind_travel_warning',
      priority: 'medium',
      title: 'Travel Caution',
      message: 'Strong winds may affect vehicle stability. Ride carefully and reduce speed.',
      channels: ['push'],
      condition: 'driver + wind_speed >= 14',
    });
  }

  return notifications;
}

module.exports = {
  buildTriggerNotifications,
};
