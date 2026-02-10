import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import {
  WiDaySunny,
  WiRaindrops,
} from "react-icons/wi";
import {
  HiOutlineMapPin,
  HiOutlineBell,
  HiOutlineUser,
  HiOutlineSparkles,
} from "react-icons/hi2";
import { useAuth } from "../context/AuthContext";
import useLocation from "../hooks/useLocation";
import LocationSelector from "./LocationSelector";
import InstallButton from "./InstallButton";

export default function Navbar() {
  const { isLoggedIn } = useAuth();
  const { location, refreshLocation } = useLocation();
  const [showLocationSelector, setShowLocationSelector] = useState(false);
  const navigate = useNavigate();

  const handleLocationClick = () => {
    console.log('Location button clicked in navbar, opening selector...');
    setShowLocationSelector(true);
  };

  const handleLocationSelect = (locationData) => {
    console.log('Location selected in navbar:', locationData);
    refreshLocation();
    setShowLocationSelector(false);
  };

  return (
    <nav className="sticky top-0 z-50 bg-black/80 backdrop-blur-xl border-b border-white/[0.05]">
      <div className="px-4 mx-auto max-w-7xl sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <NavLink to="/" className="flex items-center gap-2 group">
            <span className="text-2xl">⛅</span>
            <span className="text-lg font-bold text-transparent bg-gradient-to-r from-indigo-300 to-purple-300 bg-clip-text">
              Mausam Vaani
            </span>
          </NavLink>

          {/* Desktop Nav */}
          <div className="items-center hidden gap-1 md:flex">
            {[
              { to: "/", label: "Home", icon: <WiDaySunny size={20} /> },
              { to: "/forecast", label: "Forecast", icon: <WiRaindrops size={20} /> },
              { to: "/map", label: "Map", icon: <HiOutlineMapPin size={18} /> },
              { to: "/alerts", label: "Alerts", icon: <HiOutlineBell size={18} /> },
              { to: "/planner", label: "Planner", icon: <HiOutlineSparkles size={18} /> },
            ].map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) =>
                  `flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? "bg-white/[0.12] text-white"
                      : "text-gray-400 hover:text-white hover:bg-white/[0.06]"
                  }`
                }
              >
                {link.icon}
                {link.label}
              </NavLink>
            ))}
          </div>

          {/* Right side */}
          <div className="flex items-center gap-2">
            {/* Location Button - Works for all states and screen sizes */}
            <div className="flex items-center gap-2">
              {/* Desktop: Show current location + button */}
              <div className="items-center hidden gap-2 sm:flex">
                {location && (
                  <div className="flex items-center gap-1.5 text-sm text-gray-300 px-2 py-1">
                    <HiOutlineMapPin className="text-blue-400" size={14} />
                    <span className="max-w-[100px] truncate text-xs">{location.city}</span>
                  </div>
                )}
                <button
                  onClick={handleLocationClick}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors bg-blue-500/20 text-blue-300 hover:bg-blue-500/30 border border-blue-500/30 h-[50%] backdrop-blur-lg"
                  title={location ? "Change location" : "Set your location"}
                >
                  <HiOutlineMapPin size={14} />
                  {location ? "Change" : "Set Location"}
                </button>
              </div>
              
              {/* Mobile: Single location button */}
              <button
                onClick={handleLocationClick}
                className="p-2 text-blue-300 transition-colors border rounded-lg sm:hidden bg-blue-500/20 hover:bg-blue-500/30 border-blue-500/30"
                title={location ? `Current: ${location.city} - Tap to change` : "Set your location"}
              >
                <HiOutlineMapPin size={18} />
              </button>
            </div>

            {/* Install App Button */}
            <InstallButton variant="outline" size="large" />

            {/* Profile / Login */}
            {isLoggedIn ? (
              <NavLink
                to="/profile"
                className="p-2 rounded-xl hover:bg-white/[0.08] transition-colors text-gray-300 hover:text-white"
              >
                <HiOutlineUser size={20} />
              </NavLink>
            ) : (
              <button
                onClick={() => navigate("/login")}
                className="px-4 py-1.5 rounded-xl bg-indigo-500/20 text-indigo-300 text-sm font-medium hover:bg-indigo-500/30 transition-colors border border-indigo-500/30"
              >
                Sign in
              </button>
            )}
          </div>
        </div>
      </div>
      
      {/* Location Selector Modal */}
      <LocationSelector 
        isOpen={showLocationSelector}
        onClose={() => {
          console.log('Closing location selector from navbar...');
          setShowLocationSelector(false);
        }}
        onLocationSelect={handleLocationSelect}
      />
    </nav>
  );
}
