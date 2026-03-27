import { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";
import { supabase } from "../services/supabaseClient";
import { weatherService } from "../services/weatherService";

const HYPERLOCAL_CACHE_KEY = "mv_last_hyperlocal_location";
const DEFAULT_FALLBACK_COORDS = { lat: 20.5937, lon: 78.9629 };

const toRadians = (deg) => (deg * Math.PI) / 180;

const distanceInMeters = (lat1, lon1, lat2, lon2) => {
  const R = 6371000;
  const dLat = toRadians(Number(lat2) - Number(lat1));
  const dLon = toRadians(Number(lon2) - Number(lon1));
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRadians(Number(lat1))) *
      Math.cos(toRadians(Number(lat2))) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
};

const readCachedHyperlocal = () => {
  try {
    const raw = localStorage.getItem(HYPERLOCAL_CACHE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
};

const writeCachedHyperlocal = (locationData) => {
  try {
    localStorage.setItem(HYPERLOCAL_CACHE_KEY, JSON.stringify(locationData));
  } catch {
    // Ignore cache write failures.
  }
};

/**
 * Enhanced Location hook with user preferences and location management.
 * Priorities:
 * 1. Current device location (hyperlocal)
 * 2. User's saved location (if device location unavailable)
 * 3. Last resolved cached location
 * 4. Dynamic fallback via reverse geocode
 */
export default function useLocation() {
  const [location, setLocation] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [currentLocation, setCurrentLocation] = useState(null);
  const [savedLocation, setSavedLocation] = useState(null);
  const [locationSource, setLocationSource] = useState("fallback");
  const { user, isLoggedIn, refreshProfile } = useAuth();

  // Get current device location with hyperlocal place lookup.
  const getCurrentLocation = () => {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error("Geolocation not supported"));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        async (pos) => {
          const lat = pos.coords.latitude;
          const lon = pos.coords.longitude;

          const nearbyPlace = await weatherService.getNearbyPlaceName(lat, lon, { radius: 1500 });

          const coords = {
            lat,
            lon,
            city: nearbyPlace?.name || "Unknown",
            formattedAddress: nearbyPlace?.formattedAddress || null,
            placeId: nearbyPlace?.placeId || null,
          };

          if (nearbyPlace?.name) {
            writeCachedHyperlocal(coords);
          } else {
            const cached = readCachedHyperlocal();
            const isCloseToCached =
              cached?.lat != null &&
              cached?.lon != null &&
              distanceInMeters(lat, lon, cached.lat, cached.lon) <= 250;

            if (isCloseToCached && cached?.city) {
              coords.city = cached.city;
              coords.formattedAddress = cached.formattedAddress || null;
              coords.placeId = cached.placeId || null;
            }
          }

          if (!nearbyPlace && coords.city === "Unknown") {
            const fallbackCity = await weatherService.reverseGeocode(lat, lon);
            coords.city = fallbackCity || "Unknown";
          }

          setCurrentLocation(coords);
          resolve(coords);
        },
        (err) => {
          reject(err);
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 0,
        }
      );
    });
  };

  // Get user's saved location.
  const getSavedLocation = async () => {
    if (!isLoggedIn || !user?.location?.coordinates?.latitude) {
      return null;
    }

    return {
      lat: user.location.coordinates.latitude,
      lon: user.location.coordinates.longitude,
      city: user.location.city || user.location.name || "Saved Location",
    };
  };

  // Update user's preferred location.
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

      const { error: updateError } = await supabase
        .from("users")
        .update({
          location: locationPayload,
          locations: [locationPayload],
          use_current_location: Boolean(locationData?.isCurrentLocation),
        })
        .eq("id", user.id);

      if (updateError) {
        console.error("Failed to update location:", updateError.message);
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
        formattedAddress: locationData.formattedAddress || null,
        placeId: locationData.placeId || null,
      });
      setLocationSource(locationData?.isCurrentLocation ? "current" : "saved");

      await refreshProfile();
      return true;
    } catch (err) {
      console.error("Failed to update location:", err);
      return false;
    }
  };

  // Set location with priority logic.
  const setLocationPriority = async () => {
    setLoading(true);
    setError(null);
    let finalLocation = null;

    try {
      // Keep saved location available for UI metadata.
      if (isLoggedIn) {
        const saved = await getSavedLocation();
        setSavedLocation(saved);
      }

      // Priority 1: live location if browser permission is available.
      try {
        finalLocation = await getCurrentLocation();
        setLocationSource("current");
      } catch (locError) {
        console.warn("Live location unavailable:", locError.message);
      }

      // Priority 2: saved profile location.
      if (!finalLocation && isLoggedIn) {
        const saved = await getSavedLocation();
        if (saved) {
          finalLocation = saved;
          setLocationSource("saved");
        }
      }

      // Priority 3: cached last known hyperlocal location.
      if (!finalLocation) {
        const cached = readCachedHyperlocal();
        if (cached?.lat != null && cached?.lon != null) {
          finalLocation = cached;
          setLocationSource("cached");
        }
      }

      // Priority 4: dynamic fallback from reverse geocode.
      if (!finalLocation) {
        const nearbyPlace = await weatherService.getNearbyPlaceName(
          DEFAULT_FALLBACK_COORDS.lat,
          DEFAULT_FALLBACK_COORDS.lon,
          { radius: 5000 }
        );
        finalLocation = {
          lat: DEFAULT_FALLBACK_COORDS.lat,
          lon: DEFAULT_FALLBACK_COORDS.lon,
          city: nearbyPlace?.name || "Unknown",
          formattedAddress: nearbyPlace?.formattedAddress || null,
          placeId: nearbyPlace?.placeId || null,
        };
        setLocationSource("fallback");
      }

      setLocation(finalLocation);
    } catch (err) {
      console.error("Location error:", err);
      setError(err.message);
      setLocationSource("fallback");
      setLocation({
        lat: DEFAULT_FALLBACK_COORDS.lat,
        lon: DEFAULT_FALLBACK_COORDS.lon,
        city: "Unknown",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setLocationPriority();
  }, [
    isLoggedIn,
    user?.id,
  ]);

  return {
    location,
    currentLocation,
    savedLocation,
    locationSource,
    error,
    loading,
    updateUserLocation,
    setManualLocation: (locationData) => {
      if (!locationData?.lat || !locationData?.lon) return;
      setLocation({
        lat: locationData.lat,
        lon: locationData.lon,
        city: locationData.city || "Unknown",
        formattedAddress: locationData.formattedAddress || null,
        placeId: locationData.placeId || null,
      });
      setLocationSource(locationData?.isCurrentLocation ? "current" : "saved");
    },
    getCurrentLocation: () => getCurrentLocation().catch(() => null),
    refreshLocation: setLocationPriority,
  };
}
