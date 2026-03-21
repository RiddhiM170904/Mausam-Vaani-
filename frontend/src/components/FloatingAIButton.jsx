import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { Bell, Sparkles } from "lucide-react";

const VISIBLE_PATHS = new Set([
  "/",
  "/forecast",
  "/alerts",
  "/profile",
  "/insights",
  "/map",
]);

export default function FloatingAIButton() {
  const { pathname } = useLocation();

  if (!VISIBLE_PATHS.has(pathname)) return null;

  return (
    <div className="fixed z-40 flex flex-col items-center gap-4 bottom-24 right-4 md:bottom-8">
     
        <Link
          to="/alerts"
          className="relative inline-flex items-center justify-center text-white transition-transform border rounded-full shadow-lg group h-11 w-11 border-rose-400/45 bg-linear-to-br from-rose-500 to-orange-500 shadow-rose-500/30 hover:scale-105"
          aria-label="Open Alerts"
        >
          <span className="absolute inset-0 rounded-full bg-rose-400/25 animate-ping"></span>
          <Bell className="relative w-6 h-6" />
        </Link>
      

      <Link
        to="/assistant"
        className="relative inline-flex items-center justify-center text-white transition-transform border rounded-full shadow-lg group h-14 w-14 border-indigo-400/50 bg-linear-to-br from-indigo-500 to-cyan-500 shadow-indigo-500/30 hover:scale-105"
        aria-label="Open AI Assistant"
      >
        <span className="absolute inset-0 rounded-full bg-indigo-400/30 animate-ping"></span>
        <Sparkles className="w-6 h-6" />
      </Link>
    </div>
  );
}