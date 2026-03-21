import { Link, useLocation } from "react-router-dom";
import { Sparkles } from "lucide-react";

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
    <Link
      to="/assistant"
      className="group fixed bottom-24 right-4 z-40 inline-flex h-14 w-14 items-center justify-center rounded-full border border-indigo-400/50 bg-linear-to-br from-indigo-500 to-cyan-500 text-white shadow-lg shadow-indigo-500/30 transition-transform hover:scale-105 md:bottom-8"
      aria-label="Open AI Assistant"
    >
      <span className="absolute inset-0 rounded-full bg-indigo-400/30 animate-ping"></span>
      <Sparkles className="h-6 w-6" />
    </Link>
  );
}