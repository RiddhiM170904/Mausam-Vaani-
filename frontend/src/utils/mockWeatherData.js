/**
 * Mock Weather Data Generator
 * Generates 168 hours of historical weather data for testing
 */

/**
 * City coordinates and base weather patterns
 */
const CITIES = {
    Delhi: {
        latitude: 28.6139,
        longitude: 77.2090,
        baseTemp: 28,
        tempVariation: 8,
        baseHumidity: 65,
        monsoonMonth: [7, 8, 9]
    },
    Mumbai: {
        latitude: 19.0760,
        longitude: 72.8777,
        baseTemp: 28,
        tempVariation: 5,
        baseHumidity: 80,
        monsoonMonth: [6, 7, 8, 9]
    },
    Bengaluru: {
        latitude: 12.9716,
        longitude: 77.5946,
        baseTemp: 24,
        tempVariation: 6,
        baseHumidity: 70,
        monsoonMonth: [6, 7, 8]
    }
};

/**
 * Generate realistic temperature based on hour of day
 * @param {number} hour - Hour of day (0-23)
 * @param {number} baseTemp - Base temperature
 * @param {number} variation - Temperature variation
 * @returns {number} Temperature in Celsius
 */
function generateTemperature(hour, baseTemp, variation) {
    // Temperature follows sinusoidal pattern (cooler at night, warmer in afternoon)
    const hourFactor = Math.sin(((hour - 6) / 24) * 2 * Math.PI);
    const randomFactor = (Math.random() - 0.5) * 2;
    return baseTemp + (hourFactor * variation) + randomFactor;
}

/**
 * Generate realistic humidity based on temperature
 * @param {number} temp - Temperature
 * @param {number} baseHumidity - Base humidity
 * @returns {number} Humidity percentage (0-100)
 */
function generateHumidity(temp, baseHumidity) {
    // Humidity inversely related to temperature
    const tempFactor = (30 - temp) * 0.5;
    const randomFactor = (Math.random() - 0.5) * 10;
    const humidity = baseHumidity + tempFactor + randomFactor;
    return Math.max(30, Math.min(100, humidity));
}

/**
 * Generate rainfall based on month
 * @param {number} month - Month (1-12)
 * @param {Array} monsoonMonths - Monsoon months for the city
 * @returns {number} Rainfall in mm
 */
function generateRainfall(month, monsoonMonths) {
    if (monsoonMonths.includes(month)) {
        // Higher chance of rain during monsoon
        return Math.random() > 0.4 ? Math.random() * 15 : 0;
    } else {
        // Lower chance of rain otherwise
        return Math.random() > 0.8 ? Math.random() * 5 : 0;
    }
}

/**
 * Generate mock historical weather data
 * @param {string} cityName - City name (Delhi, Mumbai, Bengaluru)
 * @param {number} hours - Number of hours of data to generate (default: 168)
 * @returns {Object} Historical weather data
 */
export function generateMockWeatherData(cityName = 'Delhi', hours = 168) {
    const city = CITIES[cityName];

    if (!city) {
        throw new Error(`City ${cityName} not found. Available: ${Object.keys(CITIES).join(', ')}`);
    }

    const now = new Date();
    const currentMonth = now.getMonth() + 1; // 1-12

    const data = {
        timestamp: [],
        temperature: [],
        humidity: [],
        wind_speed: [],
        rainfall: [],
        pressure: [],
        cloud_cover: [],
        latitude: [],
        longitude: []
    };

    // Generate data for past 'hours' hours
    for (let i = hours - 1; i >= 0; i--) {
        const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
        const hour = timestamp.getHours();

        // Generate weather parameters
        const temp = generateTemperature(hour, city.baseTemp, city.tempVariation);
        const humidity = generateHumidity(temp, city.baseHumidity);
        const rainfall = generateRainfall(currentMonth, city.monsoonMonth);
        const windSpeed = 2 + Math.random() * 8; // 2-10 km/h
        const pressure = 1005 + (Math.random() - 0.5) * 10; // 1000-1010 hPa
        const cloudCover = rainfall > 0 ? 60 + Math.random() * 40 : Math.random() * 60;

        // Add to data arrays
        data.timestamp.push(timestamp.toISOString().slice(0, 16).replace('T', ' '));
        data.temperature.push(parseFloat(temp.toFixed(1)));
        data.humidity.push(parseFloat(humidity.toFixed(1)));
        data.wind_speed.push(parseFloat(windSpeed.toFixed(1)));
        data.rainfall.push(parseFloat(rainfall.toFixed(1)));
        data.pressure.push(parseFloat(pressure.toFixed(1)));
        data.cloud_cover.push(parseFloat(cloudCover.toFixed(1)));
        data.latitude.push(city.latitude);
        data.longitude.push(city.longitude);
    }

    return data;
}

/**
 * Get city information
 * @param {string} cityName - City name
 * @returns {Object} City info with coordinates
 */
export function getCityInfo(cityName) {
    const city = CITIES[cityName];
    if (!city) {
        return null;
    }

    return {
        name: cityName,
        latitude: city.latitude,
        longitude: city.longitude
    };
}

/**
 * Get list of available cities
 * @returns {Array<string>} List of city names
 */
export function getAvailableCities() {
    return Object.keys(CITIES);
}

export default {
    generateMockWeatherData,
    getCityInfo,
    getAvailableCities
};
