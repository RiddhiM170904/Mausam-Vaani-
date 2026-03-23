export const RAG_RULES = [
  {
    id: 'aqi_unhealthy',
    severity: 'high',
    when: { aqi: { gte: 150 } },
    text: 'AQI above 150 is unhealthy for prolonged outdoor exposure. Reduce outdoor time and use a mask when needed.',
  },
  {
    id: 'rain_bike_risk',
    severity: 'high',
    when: {
      rain_probability: { gte: 0.6 },
      user_types: ['driver', 'delivery'],
      vehicles: ['bike', 'two_wheeler', 'scooter'],
    },
    text: 'Rain and two-wheeler travel increase accident risk. Slow down, wear rain protection, and avoid unnecessary trips.',
  },
  {
    id: 'heat_dehydration',
    severity: 'medium',
    when: { temp: { gte: 34 } },
    text: 'Heat can cause dehydration. Drink water frequently and avoid direct sun during afternoon hours.',
  },
  {
    id: 'wind_travel_warning',
    severity: 'medium',
    when: { wind_speed: { gte: 14 } },
    text: 'Strong winds can impact travel safety. Keep speed moderate and stay alert for road hazards.',
  },
];
