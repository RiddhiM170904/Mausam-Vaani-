import { useEffect, useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import {
  HiOutlineHome,
  HiOutlineCalendarDays,
  HiOutlineMap,
  HiOutlineSparkles,
  HiOutlineUser,
} from "react-icons/hi2";

const links = [
  { to: "/", icon: HiOutlineHome, label: "Home" },
  { to: "/forecast", icon: HiOutlineCalendarDays, label: "Forecast" },
  { to: "/planner", icon: HiOutlineCalendarDays, label: "Planner" },
  { to: "/insights", icon: HiOutlineSparkles, label: "Insights" },
  { to: "/map", icon: HiOutlineMap, label: "Map" },
  { to: "/profile", icon: HiOutlineUser, label: "Profile" },
];

export default function BottomNav() {
  const [windowWidth, setWindowWidth] = useState(() =>
    typeof window !== "undefined" ? window.innerWidth : 390
  );

  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const iconSize = useMemo(() => {
    if (windowWidth <= 340) return 17;
    if (windowWidth <= 390) return 18;
    return 20;
  }, [windowWidth]);

  const labelClass = useMemo(() => {
    if (windowWidth <= 340) return "text-[8px]";
    if (windowWidth <= 390) return "text-[9px]";
    return "text-[10px]";
  }, [windowWidth]);

  return (
    <motion.nav
      initial={{ y: 40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="fixed bottom-0 inset-x-0 z-50 border-t border-indigo-400/15 bg-slate-950/80 pb-[env(safe-area-inset-bottom)] backdrop-blur-xl md:hidden"
    >
      <div
        className="grid h-16 items-center gap-1 px-2"
        style={{ gridTemplateColumns: `repeat(${links.length}, minmax(0, 1fr))` }}
      >
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              `group relative flex w-full flex-col items-center gap-0.5 rounded-xl px-1.5 py-1.5 transition-all duration-200 ${
                isActive
                  ? "bg-indigo-400/10 text-indigo-300"
                  : "text-gray-500 hover:bg-indigo-400/8 hover:text-gray-300"
              }`
            }
          >
            {({ isActive }) => (
              <motion.div
                whileTap={{ scale: 0.92 }}
                className="flex flex-col items-center gap-0.5"
              >
                <motion.div
                  animate={{ y: isActive ? -1 : 0, scale: isActive ? 1.05 : 1 }}
                  transition={{ type: "spring", stiffness: 380, damping: 24 }}
                >
                  <link.icon size={iconSize} />
                </motion.div>
                <span className={`${labelClass} font-medium leading-none`}>{link.label}</span>
                {isActive && (
                  <motion.span
                    layoutId="bottom-nav-active-dot"
                    className="mt-0.5 h-1 w-1 rounded-full bg-indigo-300"
                    transition={{ type: "spring", stiffness: 420, damping: 30 }}
                  />
                )}
              </motion.div>
            )}
          </NavLink>
        ))}
      </div>
    </motion.nav>
  );
}
