/**
 * Insight Card Component
 * Displays personalized weather insights from Gemini AI
 */

import { Sparkles, MapPin, Briefcase, Info } from 'lucide-react';

const InsightCard = ({ insightData }) => {
    if (!insightData) {
        return null;
    }

    const { location, weather, personalized_insight } = insightData;

    /**
     * Get condition color
     */
    const getConditionColor = (condition) => {
        const colors = {
            'Heavy Rain': 'bg-blue-600',
            'Light Rain': 'bg-blue-400',
            'Clear': 'bg-green-500',
            'Cloudy': 'bg-gray-500',
            'Very Hot': 'bg-red-500'
        };
        return colors[condition] || 'bg-gray-500';
    };

    return (
        <div className="bg-gradient-to-br from-purple-50 via-white to-pink-50 rounded-2xl shadow-lg p-6 border border-purple-100">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <div className="bg-purple-600 p-3 rounded-xl">
                    <Sparkles className="w-6 h-6 text-white" />
                </div>
                <div>
                    <h3 className="text-2xl font-bold text-gray-800">
                        AI-Powered Insights
                    </h3>
                    <p className="text-sm text-gray-600">
                        Personalized recommendations by Gemini AI
                    </p>
                </div>
            </div>

            {/* Location Info */}
            <div className="bg-white rounded-xl p-4 mb-4 border border-gray-200">
                <div className="flex items-start gap-3">
                    <MapPin className="w-5 h-5 text-purple-600 mt-1" />
                    <div>
                        <p className="font-semibold text-gray-800">{location.city}</p>
                        <p className="text-sm text-gray-600">
                            {location.latitude.toFixed(4)}° N, {location.longitude.toFixed(4)}° E
                        </p>
                    </div>
                </div>
            </div>

            {/* Weather Condition Badge */}
            <div className="mb-4">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-gray-200">
                    <div className={`w-3 h-3 rounded-full ${getConditionColor(weather.condition)}`}></div>
                    <span className="font-semibold text-gray-800">{weather.condition}</span>
                </div>
            </div>

            {/* Current Weather Stats */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-6">
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <p className="text-xs text-gray-600 mb-1">Temperature</p>
                    <p className="text-xl font-bold text-orange-600">
                        {Math.round(weather.current.temperature)}°C
                    </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <p className="text-xs text-gray-600 mb-1">Humidity</p>
                    <p className="text-xl font-bold text-blue-600">
                        {Math.round(weather.current.humidity)}%
                    </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <p className="text-xs text-gray-600 mb-1">Wind Speed</p>
                    <p className="text-xl font-bold text-cyan-600">
                        {weather.current.wind_speed.toFixed(1)} km/h
                    </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <p className="text-xs text-gray-600 mb-1">Rainfall</p>
                    <p className="text-xl font-bold text-sky-600">
                        {weather.current.rainfall.toFixed(1)} mm
                    </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <p className="text-xs text-gray-600 mb-1">Pressure</p>
                    <p className="text-xl font-bold text-purple-600">
                        {Math.round(weather.current.pressure)} hPa
                    </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <p className="text-xs text-gray-600 mb-1">Cloud Cover</p>
                    <p className="text-xl font-bold text-gray-600">
                        {Math.round(weather.current.cloud_cover)}%
                    </p>
                </div>
            </div>

            {/* Personalized Insight */}
            <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl p-6 text-white">
                <div className="flex items-start gap-3 mb-4">
                    <Briefcase className="w-6 h-6 flex-shrink-0 mt-1" />
                    <div>
                        <h4 className="text-lg font-semibold mb-2">
                            Recommendation for You
                        </h4>
                        <p className="text-white/90 leading-relaxed text-base">
                            {personalized_insight}
                        </p>
                    </div>
                </div>
            </div>

            {/* Info Footer */}
            <div className="mt-4 flex items-center gap-2 text-sm text-gray-600">
                <Info className="w-4 h-4" />
                <p>
                    Insights are generated in real-time based on current weather conditions and your profile.
                </p>
            </div>
        </div>
    );
};

export default InsightCard;
