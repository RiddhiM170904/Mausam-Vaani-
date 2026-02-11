import { useState, useEffect } from 'react';
import { MapPin, Search, Navigation, X, Check, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from './GlassCard';
import { weatherService } from '../services/weatherService';
import useLocation from '../hooks/useLocation';

const LocationSelector = ({ isOpen, onClose, onLocationSelect }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentLocationLoading, setCurrentLocationLoading] = useState(false);
  const [error, setError] = useState('');
  
  const { getCurrentLocation, currentLocation, updateUserLocation } = useLocation();

  // Search cities
  const handleSearch = async (query) => {
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const results = await weatherService.searchCity(query);
      setSearchResults(results.slice(0, 5)); // Limit to 5 results
    } catch (err) {
      setError('Failed to search cities');
      setSearchResults([]);
    }
    
    setLoading(false);
  };

  // Use current location
  const handleUseCurrentLocation = async () => {
    setCurrentLocationLoading(true);
    setError('');
    
    try {
      const location = await getCurrentLocation();
      if (location) {
        // Get city name for the coordinates
        const cityName = await weatherService.reverseGeocode(location.lat, location.lon);
        
        const locationData = {
          lat: location.lat,
          lon: location.lon,
          city: cityName,
          isCurrentLocation: true
        };

        // Update in database if user is logged in
        const saved = await updateUserLocation(locationData);
        
        onLocationSelect(locationData);
        onClose();
      }
    } catch (err) {
      setError('Could not access your location. Please check permissions.');
    }
    
    setCurrentLocationLoading(false);
  };

  // Select a searched city
  const handleCitySelect = async (city) => {
    const locationData = {
      lat: city.lat,
      lon: city.lon,
      city: city.name,
      state: city.state,
      country: city.country,
      isCurrentLocation: false
    };

    // Update in database if user is logged in  
    const saved = await updateUserLocation(locationData);
    
    onLocationSelect(locationData);
    onClose();
  };

  useEffect(() => {
    const debounceTimer = setTimeout(() => {
      handleSearch(searchQuery);
    }, 300);

    return () => clearTimeout(debounceTimer);
  }, [searchQuery]);

  // Debug logging
  useEffect(() => {
    console.log('LocationSelector isOpen changed:', isOpen);
    if (isOpen) {
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden';
    } else {
      // Restore body scroll when modal is closed
      document.body.style.overflow = 'unset';
    }
    
    // Cleanup function to restore scroll when component unmounts
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-lg z-[100] flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          onClick={(e) => e.stopPropagation()}
          className="w-full h-[50%] max-w-md mx-auto"
        >
          <GlassCard className="p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <MapPin className="w-6 h-6 text-blue-400" />
                <h2 className="text-xl font-semibold text-white">Set Location</h2>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 transition-colors hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Current Location Button */}
            <button
              onClick={handleUseCurrentLocation}
              disabled={currentLocationLoading}
              className="flex items-center justify-center w-full gap-2 p-3 mb-4 text-blue-300 transition-colors border bg-blue-500/10 border-blue-500/30 rounded-xl hover:bg-blue-500/20 disabled:opacity-50"
            >
              {currentLocationLoading ? (
                <div className="w-4 h-4 border-2 border-blue-300 rounded-full border-t-transparent animate-spin" />
              ) : (
                <Navigation className="w-4 h-4" />
              )}
              Use Current Location
            </button>

            {/* Search */}
            <div className="mb-4">
              <div className="relative">
                <Search className="absolute w-4 h-4 text-gray-400 transform -translate-y-1/2 left-3 top-1/2" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search for a city..."
                  className="w-full py-3 pl-10 pr-4 text-white placeholder-gray-500 border border-gray-700 bg-gray-800/50 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                {loading && (
                  <div className="absolute transform -translate-y-1/2 right-3 top-1/2">
                    <div className="w-4 h-4 border-2 border-gray-400 rounded-full border-t-transparent animate-spin" />
                  </div>
                )}
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="flex items-center gap-2 p-3 mb-4 border bg-red-500/10 border-red-500/30 rounded-xl">
                <AlertCircle className="flex-shrink-0 w-4 h-4 text-red-400" />
                <p className="text-sm text-red-400">{error}</p>
              </div>
            )}

            {/* Search Results */}
            <div className="space-y-2 overflow-y-auto max-h-64">
              {searchResults.map((city, index) => (
                <button
                  key={`${city.lat}-${city.lon}-${index}`}
                  onClick={() => handleCitySelect(city)}
                  className="w-full p-3 text-left transition-colors border bg-gray-800/30 hover:bg-gray-700/50 border-gray-700/50 rounded-xl group"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-white">{city.name}</p>
                      <p className="text-sm text-gray-400">
                        {city.state && `${city.state}, `}{city.country}
                      </p>
                    </div>
                    <Check className="w-4 h-4 text-green-400 transition-opacity opacity-0 group-hover:opacity-100" />
                  </div>
                </button>
              ))}
              
              {searchQuery.length >= 2 && searchResults.length === 0 && !loading && (
                <p className="py-4 text-center text-gray-500">No cities found</p>
              )}
            </div>

            {/* Popular Cities (when no search) */}
            {searchQuery.length < 2 && (
              <div>
                <p className="mb-3 text-sm text-gray-400">Popular Cities</p>
                <div className="flex flex-col gap-2">
                  {[
                    { name: "New Delhi", state: "Delhi", country: "IN", lat: 28.6139, lon: 77.209 },
                    { name: "Mumbai", state: "Maharashtra", country: "IN", lat: 19.0760, lon: 72.8777 },
                    { name: "Bangalore", state: "Karnataka", country: "IN", lat: 12.9716, lon: 77.5946 },
                    { name: "Chennai", state: "Tamil Nadu", country: "IN", lat: 13.0827, lon: 80.2707 },
                  ].map((city) => (
                    <button
                      key={`${city.lat}-${city.lon}`}
                      onClick={() => handleCitySelect(city)}
                      className="w-full p-3 text-left transition-colors border bg-gray-800/30 hover:bg-gray-700/50 border-gray-700/50 rounded-xl group"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-white">{city.name}</p>
                          <p className="text-sm text-gray-400">{city.state}</p>
                        </div>
                        <Check className="w-4 h-4 text-green-400 transition-opacity opacity-0 group-hover:opacity-100" />
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </GlassCard>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default LocationSelector;