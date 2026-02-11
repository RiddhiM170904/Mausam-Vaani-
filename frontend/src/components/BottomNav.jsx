import { NavLink } from "react-router-dom";
import { HiOutlineHome, HiOutlineCalendarDays, HiOutlineMap, HiOutlineBell, HiOutlineSparkles } from "react-icons/hi2";

const links = [
  { to: "/", icon: HiOutlineHome, label: "Home" },
  { to: "/forecast", icon: HiOutlineCalendarDays, label: "Forecast" },
  { to: "/map", icon: HiOutlineMap, label: "Map" },
  { to: "/alerts", icon: HiOutlineBell, label: "Alerts" },
  { to: "/planner", icon: HiOutlineSparkles, label: "Planner" },
];

export default function BottomNav() {
  return (
    <nav className="fixed bottom-0 inset-x-0 z-50 md:hidden bg-black/90 backdrop-blur-xl border-t border-white/5">
      <div className="flex items-center justify-around h-16 px-2">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              `flex flex-col items-center gap-0.5 px-2 py-1.5 rounded-xl transition-all duration-200 ${
                isActive
                  ? "text-indigo-400"
                  : "text-gray-500 hover:text-gray-300"
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
