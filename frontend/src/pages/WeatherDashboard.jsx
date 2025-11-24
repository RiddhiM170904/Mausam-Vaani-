/**
 * Weather Dashboard Page
 * Main page for weather predictions and personalized insights
 */

import { useState } from 'react';
import { MapPin, User, Briefcase, Loader, AlertCircle, CheckCircle } from 'lucide-react';
import apiService from '../services/api';
import { generateMockWeatherData, getCityInfo, getAvailableCities } from '../utils/mockWeatherData';
import WeatherForecast from '../components/WeatherForecast';
import InsightCard from '../components/InsightCard';

const WeatherDashboard = () => {
    // State management
    const [selectedCity, setSelectedCity] = useState('Delhi');
    const [userProfession, setUserProfession] = useState('Farmer');
    const [userContext, setUserContext] = useState({});
    const [contextInput, setContextInput] = useState('Rice'); // For crop/activity input

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [backendStatus, setBackendStatus] = useState('unknown');

    const [weatherForecast, setWeatherForecast] = useState(null);
    const [personalizedInsight, setPersonalizedInsight] = useState(null);

    // Available options
    const cities = getAvailableCities();
    const professions = [
        'Farmer',
        'Commuter',
        'Construction Worker',
        'Outdoor Sports',
        'Tourism',
        'Business',
        'General'
    ];

    /**
     * Check backend health on mount
     */
    const checkBackendHealth = async () => {
        const result = await apiService.checkHealth();
        if (result.success) {
            setBackendStatus('online');
        } else {
            setBackendStatus('offline');
        }
    };

    /**
     * Handle profession change and update context
     */
    const handleProfessionChange = (profession) => {
        setUserProfession(profession);

        // Set default context based on profession
        if (profession === 'Farmer') {
            setUserContext({ crop: contextInput || 'Rice' });
        } else if (profession === 'Commuter') {
            setUserContext({ transport: contextInput || 'Car' });
        } else if (profession === 'Construction Worker') {
            setUserContext({ project_type: contextInput || 'Building' });
        } else if (profession === 'Outdoor Sports') {
            setUserContext({ activity: contextInput || 'Cricket' });
        } else {
            setUserContext({});
        }
    };

    /**
     * Update context based on input
     */
    const handleContextInputChange = (value) => {
        setContextInput(value);

        if (userProfession === 'Farmer') {
            setUserContext({ crop: value });
        } else if (userProfession === 'Commuter') {
            setUserContext({ transport: value });
        } else if (userProfession === 'Construction Worker') {
            setUserContext({ project_type: value });
        } else if (userProfession === 'Outdoor Sports') {
            setUserContext({ activity: value });
        }
    };

    /**
     * Get context field label based on profession
     */
    const getContextLabel = () => {
        const labels = {
            'Farmer': 'Crop Type',
            'Commuter': 'Transport Mode',
            'Construction Worker': 'Project Type',
            'Outdoor Sports': 'Activity',
            'Tourism': 'Trip Type',
            'Business': 'Business Type'
        };
        return labels[userProfession] || 'Context';
    };

    /**
     * Get context field placeholder
     */
    const getContextPlaceholder = () => {
        const placeholders = {
            'Farmer': 'e.g., Rice, Wheat, Cotton',
            'Commuter': 'e.g., Car, Bike, Public Transport',
            'Construction Worker': 'e.g., Building, Road, Bridge',
            'Outdoor Sports': 'e.g., Cricket, Football, Running',
            'Tourism': 'e.g., Sightseeing, Adventure',
            'Business': 'e.g., Retail, Food, Services'
        };
        return placeholders[userProfession] || 'Enter context';
    };

    /**
     * Main function to get weather insights
     */
    const getWeatherInsights = async () => {
        setLoading(true);
        setError(null);
        setWeatherForecast(null);
        setPersonalizedInsight(null);

        // Check backend health first
        await checkBackendHealth();

        try {
            // Get city info
            const cityInfo = getCityInfo(selectedCity);
            if (!cityInfo) {
                throw new Error('Invalid city selected');
            }

            // Generate mock historical data (168 hours)
            console.log('Generating mock historical data...');
            const historicalData = generateMockWeatherData(selectedCity, 168);

            // Get personalized insight
            console.log('Calling API for insights...');
            const result = await apiService.getInsight({
                latitude: cityInfo.latitude,
                longitude: cityInfo.longitude,
                city: cityInfo.name,
                userProfession: userProfession,
                userContext: userContext,
                historicalData: historicalData,
                forecastSteps: 24
            });

            if (result.success) {
                // Extract forecast data
                if (result.data.forecast) {
                    setWeatherForecast(result.data.forecast);
                }

                // Set insight data
                setPersonalizedInsight(result.data);

                setError(null);
            } else {
                setError(result.error || 'Failed to get weather insights');
            }
        } catch (err) {
            console.error('Error:', err);
            setError(err.message || 'An unexpected error occurred');
        } finally {
            setLoading(false);
        }
    };

    /**
     * Get backend status badge
     */
    const getBackendStatusBadge = () => {
        if (backendStatus === 'online') {
            return (
                <div className="flex items-center gap-2 px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                    <CheckCircle className="w-4 h-4" />
                    <span>Backend Online</span>
                </div>
            );
        } else if (backendStatus === 'offline') {
            return (
                <div className="flex items-center gap-2 px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm">
                    <AlertCircle className="w-4 h-4" />
                    <span>Backend Offline</span>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-sky-50 py-12 px-4">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="text-center mb-12">
                    <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
                        AI Weather Dashboard
                    </h1>
                    <p className="text-lg text-gray-600 mb-4">
                        Get hyperlocal weather predictions and personalized insights
                    </p>
                    {getBackendStatusBadge()}
                </div>

                {/* Input Form */}
                <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
                    <h2 className="text-2xl font-bold text-gray-800 mb-6">
                        Tell us about yourself
                    </h2>

                    <div className="grid md:grid-cols-2 gap-6">
                        {/* City Selection */}
                        <div>
                            <label className="block text-sm font-semibold text-gray-700 mb-2">
                                <MapPin className="w-4 h-4 inline mr-1" />
                                Select Location
                            </label>
                            <select
                                value={selectedCity}
                                onChange={(e) => setSelectedCity(e.target.value)}
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                            >
                                {cities.map((city) => (
                                    <option key={city} value={city}>
                                        {city}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Profession Selection */}
                        <div>
                            <label className="block text-sm font-semibold text-gray-700 mb-2">
                                <User className="w-4 h-4 inline mr-1" />
                                Your Profession
                            </label>
                            <select
                                value={userProfession}
                                onChange={(e) => handleProfessionChange(e.target.value)}
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                            >
                                {professions.map((profession) => (
                                    <option key={profession} value={profession}>
                                        {profession}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Context Input (conditional) */}
                        {userProfession !== 'General' && (
                            <div className="md:col-span-2">
                                <label className="block text-sm font-semibold text-gray-700 mb-2">
                                    <Briefcase className="w-4 h-4 inline mr-1" />
                                    {getContextLabel()}
                                </label>
                                <input
                                    type="text"
                                    value={contextInput}
                                    onChange={(e) => handleContextInputChange(e.target.value)}
                                    placeholder={getContextPlaceholder()}
                                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                                />
                            </div>
                        )}
                    </div>

                    {/* Submit Button */}
                    <button
                        onClick={getWeatherInsights}
                        disabled={loading}
                        className="mt-6 w-full bg-gradient-to-r from-blue-600 to-sky-600 text-white font-semibold py-4 px-6 rounded-lg hover:from-blue-700 hover:to-sky-700 transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                        {loading ? (
                            <>
                                <Loader className="w-5 h-5 animate-spin" />
                                Getting Weather Insights...
                            </>
                        ) : (
                            <>
                                Get Weather Insights
                            </>
                        )}
                    </button>
                </div>

                {/* Error Message */}
                {error && (
                    <div className="bg-red-50 border border-red-200 rounded-xl p-6 mb-8">
                        <div className="flex items-start gap-3">
                            <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-1" />
                            <div>
                                <h3 className="font-semibold text-red-800 mb-1">Error</h3>
                                <p className="text-red-700">{error}</p>
                                <p className="text-sm text-red-600 mt-2">
                                    Make sure the AI Backend is running on http://localhost:5000
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Results */}
                {personalizedInsight && (
                    <div className="space-y-8">
                        {/* Personalized Insight */}
                        <InsightCard insightData={personalizedInsight} />

                        {/* Weather Forecast */}
                        {weatherForecast && (
                            <WeatherForecast forecast={weatherForecast} />
                        )}
                    </div>
                )}

                {/* Instructions (show when no results) */}
                {!loading && !personalizedInsight && !error && (
                    <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                        <h3 className="font-semibold text-blue-800 mb-3">
                            How to use this dashboard
                        </h3>
                        <ol className="list-decimal list-inside space-y-2 text-blue-700">
                            <li>Select your city from the dropdown</li>
                            <li>Choose your profession</li>
                            <li>Add relevant context (crop type, transport mode, etc.)</li>
                            <li>Click "Get Weather Insights" to see predictions</li>
                        </ol>
                        <p className="text-sm text-blue-600 mt-4">
                            <strong>Note:</strong> Make sure the AI Backend is running on port 5000 before using this dashboard.
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default WeatherDashboard;
