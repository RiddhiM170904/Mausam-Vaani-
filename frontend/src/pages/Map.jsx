import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import useLocation from "../hooks/useLocation";
import GlassCard from "../components/GlassCard";
import Loader from "../components/Loader";

/**
 * Interactive map page with Leaflet.
 * Shows weather overlay at current location.
 */
export default function MapPage() {
  const { location } = useLocation();
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const [layer, setLayer] = useState("temp_new");

  const OWM_KEY = import.meta.env.VITE_OWM_KEY || "";

  const layers = [
    { id: "temp_new", label: "Temperature" },
    { id: "precipitation_new", label: "Precipitation" },
    { id: "clouds_new", label: "Clouds" },
    { id: "wind_new", label: "Wind" },
  ];

  useEffect(() => {
    if (!location || mapInstanceRef.current) return;

    // Dynamic import for Leaflet
    Promise.all([
      import("leaflet"),
      import("leaflet/dist/leaflet.css"),
    ]).then(([L]) => {
      const map = L.default.map(mapRef.current, {
        center: [location.lat, location.lon],
        zoom: 8,
        zoomControl: false,
      });

      // Dark basemap
      L.default
        .tileLayer(
          "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
          {
            attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
            maxZoom: 18,
          }
        )
        .addTo(map);

      // Weather overlay
      if (OWM_KEY) {
        L.default
          .tileLayer(
            `https://tile.openweathermap.org/map/${layer}/{z}/{x}/{y}.png?appid=${OWM_KEY}`,
            { opacity: 0.6 }
          )
          .addTo(map);
      }

      // Current location marker
      L.default
        .circleMarker([location.lat, location.lon], {
          radius: 8,
          fillColor: "#818cf8",
          color: "#4f46e5",
          weight: 2,
          fillOpacity: 0.8,
        })
        .addTo(map)
        .bindPopup("📍 You are here");

      mapInstanceRef.current = map;
    });

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [location, layer, OWM_KEY]);

  if (!location) return <Loader text="Getting your location..." />;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-4"
    >
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Weather Map</h1>
        <div className="flex gap-1 p-1 rounded-xl bg-white/[0.06] overflow-x-auto hide-scrollbar">
          {layers.map((l) => (
            <button
              key={l.id}
              onClick={() => setLayer(l.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                layer === l.id
                  ? "bg-indigo-500/30 text-indigo-300"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {l.label}
            </button>
          ))}
        </div>
      </div>

      <GlassCard className="overflow-hidden" hover={false}>
        <div
          ref={mapRef}
          className="w-full h-[60vh] sm:h-[70vh] rounded-3xl"
          style={{ minHeight: 400 }}
        />
      </GlassCard>

      {!OWM_KEY && (
        <p className="text-center text-gray-600 text-xs">
          Add VITE_OWM_KEY in .env for weather overlays
        </p>
      )}
    </motion.div>
  );
}
