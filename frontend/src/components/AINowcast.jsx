// frontend/src/components/AINowcast.jsx
// Mausam Vaani — AI Nowcast Card
// Shows hyperlocal prediction, nowcast, crowd reports, and advice

import { useState, useEffect } from "react";
import { getAIPrediction, submitCrowdReport, getNearbyReports } from "../services/aiService";

export default function AINowcast({ lat, lon, weatherData }) {
  const [prediction, setPrediction]     = useState(null);
  const [nearbyCount, setNearbyCount]   = useState(0);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState(null);
  const [reported, setReported]         = useState(false);
  const [reporting, setReporting]       = useState(false);

  useEffect(() => {
    if (lat && lon && weatherData) {
      loadPrediction();
    }
  }, [lat, lon, weatherData]);

  async function loadPrediction() {
    setLoading(true);
    setError(null);
    try {
      const [pred, reports] = await Promise.all([
        getAIPrediction(weatherData, lat, lon),
        getNearbyReports(),
      ]);
      if (!pred) throw new Error("No prediction returned");
      setPrediction(pred);
      setNearbyCount(reports?.count ?? 0);
    } catch (err) {
      setError("AI analysis unavailable");
    } finally {
      setLoading(false);
    }
  }

  async function handleReport(isRaining) {
    setReporting(true);
    await submitCrowdReport(lat, lon, isRaining);
    setReporting(false);
    setReported(true);
    // Reset after 3 seconds and refresh prediction
    setTimeout(() => {
      setReported(false);
      loadPrediction();
    }, 3000);
  }

  // Rain color coding
  function rainColor(rain) {
    if (rain < 30) return "text-emerald-300";
    if (rain < 70) return "text-yellow-300";
    return "text-red-400";
  }

  function rainLabel(rain) {
    if (rain < 30) return "Low";
    if (rain < 70) return "Moderate";
    return "High";
  }

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm p-4 sm:p-5 space-y-4">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[10px] uppercase tracking-widest text-gray-500">Mausam Vaani AI</p>
          <p className="text-sm font-semibold text-white">Hyperlocal Nowcast</p>
          <p className="text-xs text-gray-500">Adjusted for your exact location</p>
        </div>
        <button
          onClick={loadPrediction}
          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-indigo-500/20 text-indigo-300 hover:bg-indigo-500/30 transition-all"
        >
          Refresh
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <p className="text-sm text-gray-400 animate-pulse">Analyzing weather for your location...</p>
      )}

      {/* Error */}
      {error && !loading && (
        <p className="text-sm text-red-400">{error}. Is the backend running on port 5000?</p>
      )}

      {/* Prediction */}
      {prediction && !loading && (
        <>
          {/* Stats row */}
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-xl bg-white/5 border border-white/8 px-3 py-3 text-center">
              <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Rain Risk</p>
              <p className={`text-xl font-bold ${rainColor(prediction.futureRain)}`}>
                {prediction.futureRain}%
              </p>
              <p className={`text-[10px] mt-1 ${rainColor(prediction.futureRain)}`}>
                {rainLabel(prediction.futureRain)}
              </p>
            </div>

            <div className="rounded-xl bg-white/5 border border-white/8 px-3 py-3 text-center">
              <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Local Temp</p>
              <p className="text-xl font-bold text-blue-300">
                {prediction.localTemp}°C
              </p>
              <p className="text-[10px] text-gray-500 mt-1">Adjusted</p>
            </div>

            <div className="rounded-xl bg-white/5 border border-white/8 px-3 py-3 text-center">
              <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Humidity</p>
              <p className="text-xl font-bold text-cyan-300">
                {prediction.localHumidity}%
              </p>
              <p className="text-[10px] text-gray-500 mt-1">Local</p>
            </div>
          </div>

          {/* Advice */}
          <div className="rounded-xl bg-indigo-500/10 border border-indigo-500/20 px-4 py-3">
            <p className="text-[10px] uppercase tracking-wider text-indigo-400 mb-1">Advisory</p>
            <p className="text-sm text-white">{prediction.advice}</p>
          </div>

          {/* Crowd reports count */}
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400"></div>
            <p className="text-xs text-gray-400">
              {nearbyCount > 0
                ? `${nearbyCount} nearby user report${nearbyCount > 1 ? "s" : ""} in last 30 mins`
                : "No nearby reports yet — be the first!"}
            </p>
          </div>

          {/* Mausam Mitra buttons */}
          <div>
            <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">
              Mausam Mitra — Report your weather
            </p>
            {reported ? (
              <div className="rounded-xl bg-emerald-500/10 border border-emerald-500/20 px-4 py-3 text-center">
                <p className="text-sm text-emerald-300">Thanks for reporting! Your data helps others.</p>
              </div>
            ) : (
              <div className="flex gap-2">
                <button
                  onClick={() => handleReport(true)}
                  disabled={reporting}
                  className="flex-1 py-2 rounded-xl text-xs font-medium bg-blue-500/20 text-blue-300 border border-blue-500/20 hover:bg-blue-500/30 transition-all disabled:opacity-50"
                >
                  {reporting ? "Submitting..." : "It's raining here"}
                </button>
                <button
                  onClick={() => handleReport(false)}
                  disabled={reporting}
                  className="flex-1 py-2 rounded-xl text-xs font-medium bg-amber-500/20 text-amber-300 border border-amber-500/20 hover:bg-amber-500/30 transition-all disabled:opacity-50"
                >
                  {reporting ? "Submitting..." : "No rain here"}
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
