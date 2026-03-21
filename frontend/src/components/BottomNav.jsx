import { NavLink } from "react-router-dom";
import {
  HiOutlineHome,
  HiOutlineCalendarDays,
  HiOutlineMap,
  HiOutlineBell,
  HiOutlineSparkles,
  HiOutlineUser,
} from "react-icons/hi2";

const links = [
  { to: "/", icon: HiOutlineHome, label: "Home" },
  { to: "/forecast", icon: HiOutlineCalendarDays, label: "Forecast" },
  { to: "/alerts", icon: HiOutlineBell, label: "Alerts" },
  { to: "/insights", icon: HiOutlineSparkles, label: "Insights" },
  { to: "/map", icon: HiOutlineMap, label: "Map" },
  { to: "/profile", icon: HiOutlineUser, label: "Profile" },
];

export default function BottomNav() {
  return (
    <nav className="fixed bottom-0 inset-x-0 z-50 border-t border-indigo-400/15 bg-slate-950/80 backdrop-blur-xl md:hidden">
      <div className="flex h-16 items-center gap-1 overflow-x-auto px-2 hide-scrollbar">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              `flex min-w-17 flex-col items-center gap-0.5 rounded-xl px-2 py-1.5 transition-all duration-200 ${
                isActive
                  ? "bg-indigo-400/10 text-indigo-300"
                  : "text-gray-500 hover:bg-indigo-400/8 hover:text-gray-300"
              }`
            }
          >
            <link.icon size={20} />
            <span className="text-[9px] font-medium">{link.label}</span>
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
