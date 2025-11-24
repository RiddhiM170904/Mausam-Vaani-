/**
 * Weather Forecast Display Component
 * Shows 24-hour weather predictions with visual cards
 */

import { Cloud, CloudRain, Sun, Wind, Droplets, Gauge } from 'lucide-react';

const WeatherForecast = ({ forecast }) => {
    if (!forecast || forecast.length === 0) {
        return null;
    }

    /**
     * Get weather icon based on conditions
     */
    const getWeatherIcon = (temp, rainfall, cloudCover) => {
        if (rainfall > 5) {
            return <CloudRain className="w-8 h-8 text-blue-600" />;
        } else if (cloudCover > 70) {
            return <Cloud className="w-8 h-8 text-gray-600" />;
        } else {
            return <Sun className="w-8 h-8 text-yellow-500" />;
        }
    };

    /**
     * Format timestamp to readable format
     */
    const formatTime = (timestamp) => {
        const date = new Date(timestamp);
        const hours = date.getHours();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        const displayHours = hours % 12 || 12;
        return `${displayHours} ${ampm}`;
    };

    /**
     * Format date
     */
    const formatDate = (timestamp) => {
        const date = new Date(timestamp);
        return date.toLocaleDateString('en-IN', {
            month: 'short',
            day: 'numeric'
        });
    };

    // Show only first 24 hours
    const displayForecast = forecast.slice(0, 24);

    return (
        <div className="bg-white rounded-2xl shadow-lg p-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <Cloud className="w-6 h-6 text-blue-600" />
                24-Hour Weather Forecast
            </h3>

            {/* Hourly forecast grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4 max-h-[600px] overflow-y-auto">
                {displayForecast.map((hourData, index) => (
                    <div
                        key={index}
                        className="bg-gradient-to-br from-blue-50 to-sky-50 rounded-xl p-4 border border-blue-100 hover:shadow-md transition-shadow"
                    >
                        {/* Time */}
                        <div className="text-center mb-2">
                            <p className="text-sm font-semibold text-gray-700">
                                {formatTime(hourData.timestamp)}
                            </p>
                            <p className="text-xs text-gray-500">
                                {formatDate(hourData.timestamp)}
                            </p>
                        </div>

                        {/* Weather Icon */}
                        <div className="flex justify-center mb-3">
                            {getWeatherIcon(
                                hourData.temperature,
                                hourData.rainfall,
                                hourData.cloud_cover
                            )}
                        </div>

                        {/* Temperature */}
                        <div className="text-center mb-3">
                            <p className="text-2xl font-bold text-gray-800">
                                {Math.round(hourData.temperature)}°C
                            </p>
                        </div>

                        {/* Additional metrics */}
                        <div className="space-y-1 text-xs">
                            <div className="flex items-center justify-between">
                                <span className="flex items-center gap-1 text-gray-600">
                                    <Droplets className="w-3 h-3" />
                                    Humidity
                                </span>
                                <span className="font-semibold text-gray-700">
                                    {Math.round(hourData.humidity)}%
                                </span>
                            </div>

                            <div className="flex items-center justify-between">
                                <span className="flex items-center gap-1 text-gray-600">
                                    <Wind className="w-3 h-3" />
                                    Wind
                                </span>
                                <span className="font-semibold text-gray-700">
                                    {hourData.wind_speed.toFixed(1)} km/h
                                </span>
                            </div>

                            {hourData.rainfall > 0 && (
                                <div className="flex items-center justify-between">
                                    <span className="flex items-center gap-1 text-blue-600">
                                        <CloudRain className="w-3 h-3" />
                                        Rain
                                    </span>
                                    <span className="font-semibold text-blue-700">
                                        {hourData.rainfall.toFixed(1)} mm
                                    </span>
                                </div>
                            )}

                            <div className="flex items-center justify-between">
                                <span className="flex items-center gap-1 text-gray-600">
                                    <Gauge className="w-3 h-3" />
                                    Pressure
                                </span>
                                <span className="font-semibold text-gray-700">
                                    {Math.round(hourData.pressure)} hPa
                                </span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Summary Stats */}
            <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-lg p-4 border border-orange-100">
                    <p className="text-sm text-gray-600 mb-1">Max Temp</p>
                    <p className="text-2xl font-bold text-orange-600">
                        {Math.round(Math.max(...displayForecast.map(h => h.temperature)))}°C
                    </p>
                </div>

                <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-4 border border-blue-100">
                    <p className="text-sm text-gray-600 mb-1">Min Temp</p>
                    <p className="text-2xl font-bold text-blue-600">
                        {Math.round(Math.min(...displayForecast.map(h => h.temperature)))}°C
                    </p>
                </div>

                <div className="bg-gradient-to-br from-sky-50 to-blue-50 rounded-lg p-4 border border-sky-100">
                    <p className="text-sm text-gray-600 mb-1">Total Rainfall</p>
                    <p className="text-2xl font-bold text-sky-600">
                        {displayForecast.reduce((sum, h) => sum + h.rainfall, 0).toFixed(1)} mm
                    </p>
                </div>

                <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-4 border border-purple-100">
                    <p className="text-sm text-gray-600 mb-1">Avg Humidity</p>
                    <p className="text-2xl font-bold text-purple-600">
                        {Math.round(displayForecast.reduce((sum, h) => sum + h.humidity, 0) / displayForecast.length)}%
                    </p>
                </div>
            </div>
        </div>
    );
};

export default WeatherForecast;
