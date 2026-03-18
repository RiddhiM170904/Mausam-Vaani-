import { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";
import { supabase } from "../services/supabaseClient";

/**
 * Enhanced Location hook with user preferences and location management.
 * Priorities: 
 * 1. User's saved location (if logged in)
 * 2. Current device location 
 * 3. Fallback to New Delhi
 */
export default function useLocation() {
  const [location, setLocation] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [currentLocation, setCurrentLocation] = useState(null);
  const [savedLocation, setSavedLocation] = useState(null);
  const { user, isLoggedIn, refreshProfile } = useAuth();

  // Get current device location
  const getCurrentLocation = () => {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error("Geolocation not supported"));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const coords = {
            lat: pos.coords.latitude,
            lon: pos.coords.longitude,
            city: "Current Location"
          };
          setCurrentLocation(coords);
          resolve(coords);
        },
        (err) => {
          reject(err);
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000, // 5 min cache
        }
      );
    });
  };

  // Get user's saved location
  const getSavedLocation = async () => {
    if (!isLoggedIn || !user?.location?.coordinates?.latitude) {
      return null;
    }

    return {
      lat: user.location.coordinates.latitude,
      lon: user.location.coordinates.longitude,
      city: user.location.city || user.location.name || "Saved Location"
    };
  };

  // Update user's preferred location
  const updateUserLocation = async (locationData) => {
    if (!isLoggedIn || !user?.id) return false;

    try {
      const locationPayload = {
        city: locationData.city,
        state: locationData.state || "",
        district: locationData.district || "",
        pincode: locationData.pincode || "",
        coordinates: {
          latitude: locationData.lat,
          longitude: locationData.lon,
        },
      };

      const { error } = await supabase
        .from("users")
        .update({
          location: locationPayload,
          locations: [locationPayload],
          use_current_location: false,
        })
        .eq("id", user.id);

      if (error) {
        console.error("Failed to update location:", error.message);
        return false;
      }

      setSavedLocation({
        lat: locationData.lat,
        lon: locationData.lon,
        city: locationData.city,
      });
      setLocation({
        lat: locationData.lat,
        lon: locationData.lon,
        city: locationData.city,
      });

      await refreshProfile();
      return true;
    } catch (err) {
      console.error('Failed to update location:', err);
      return false;
    }
  };

  // Set location with priority logic
  const setLocationPriority = async () => {
    setLoading(true);
    let finalLocation = null;

    try {
      // Check for saved location first (logged in users)
      if (isLoggedIn) {
        const saved = await getSavedLocation();
        if (saved) {
          finalLocation = saved;
          setSavedLocation(saved);
        }
      }

      // If no saved location, try to get current location
      if (!finalLocation) {
        try {
          const current = await getCurrentLocation();
          finalLocation = current;
        } catch (locError) {
          console.warn('Location access denied:', locError.message);
          setError("Location access denied. Using default location.");
          
          // Fallback: New Delhi
          finalLocation = { 
            lat: 28.6139, 
            lon: 77.209, 
            city: "New Delhi" 
          };
        }
      }

      setLocation(finalLocation);
    } catch (err) {
      console.error('Location error:', err);
      setError(err.message);
      
      // Final fallback
      setLocation({ 
        lat: 28.6139, 
        lon: 77.209, 
        city: "New Delhi" 
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setLocationPriority();
  }, [isLoggedIn, user]);

  return { 
    location, 
    currentLocation,
    savedLocation,
    error, 
    loading,
    updateUserLocation,
    getCurrentLocation: () => getCurrentLocation().catch(() => null),
    refreshLocation: setLocationPriority
  };
}
