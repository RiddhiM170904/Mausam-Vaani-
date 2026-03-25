import { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";
import { supabase } from "../services/supabaseClient";
import { weatherService } from "../services/weatherService";

const HYPERLOCAL_CACHE_KEY = "mv_last_hyperlocal_location";
const VIT_BHOPAL_COORDS = { lat: 23.0779, lon: 76.8508 };
const VIT_BHOPAL_LABEL = "Academic Block-1 VIT Bhopal University";
const VIT_BHOPAL_ADDRESS = "VIT Bhopal University, Kotri Kalan, Near Indore Road, Bhopal, Madhya Pradesh 466114";

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

const isNearVitBhopalCampus = (lat, lon) => {
  const distance = distanceInMeters(lat, lon, VIT_BHOPAL_COORDS.lat, VIT_BHOPAL_COORDS.lon);
  return distance <= 10000;
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
 * 3. Fallback to New Delhi
 */
export default function useLocation() {
  const [location, setLocation] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [currentLocation, setCurrentLocation] = useState(null);
  const [savedLocation, setSavedLocation] = useState(null);
  const [locationSource, setLocationSource] = useState("fallback");
  const { user, isLoggedIn, refreshProfile } = useAuth();

  const canAutoUseGeolocation = async () => {
    if (typeof navigator === "undefined" || !navigator.geolocation) {
      return false;
    }

    if (!navigator.permissions?.query) {
      return false;
    }

    try {
      const permission = await navigator.permissions.query({ name: "geolocation" });
      return permission?.state === "granted";
    } catch {
      return false;
    }
  };

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

          if (isNearVitBhopalCampus(lat, lon)) {
            const hardcodedCampusLocation = {
              lat,
              lon,
              city: VIT_BHOPAL_LABEL,
              formattedAddress: VIT_BHOPAL_ADDRESS,
              placeId: "hardcoded-vit-bhopal",
            };

            writeCachedHyperlocal(hardcodedCampusLocation);
            setCurrentLocation(hardcodedCampusLocation);
            resolve(hardcodedCampusLocation);
            return;
          }

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

      const shouldAutoRequest = await canAutoUseGeolocation();

      // Priority 1: saved profile location if present.
      if (isLoggedIn) {
        const saved = await getSavedLocation();
        if (saved) {
          finalLocation = saved;
          setLocationSource("saved");
        }
      }

      // Priority 2: live location only when browser permission is already granted.
      if (shouldAutoRequest) {
        try {
          finalLocation = await getCurrentLocation();
          setLocationSource("current");
        } catch (locError) {
          console.warn("Live location unavailable:", locError.message);
        }
      }

      // Priority 3: hard fallback.
      if (!finalLocation) {
        finalLocation = {
          lat: 28.6139,
          lon: 77.209,
          city: "New Delhi",
        };
        setLocationSource("fallback");
      }

      setLocation(finalLocation);
    } catch (err) {
      console.error("Location error:", err);
      setError(err.message);
      setLocationSource("fallback");
      setLocation({
        lat: 28.6139,
        lon: 77.209,
        city: "New Delhi",
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
    getCurrentLocation: () => getCurrentLocation().catch(() => null),
    refreshLocation: setLocationPriority,
  };
}
