import { createContext, useContext, useState, useEffect } from "react";

const ThemeContext = createContext(null);

const THEMES = {
  dark: {
    name: "dark",
    bg: "from-black via-gray-950 to-black",
    card: "bg-white/[0.05]",
    text: "text-white",
    textSecondary: "text-gray-400",
  },
  midnight: {
    name: "midnight",
    bg: "from-slate-950 via-gray-900 to-black",
    card: "bg-white/[0.05]",
    text: "text-white",
    textSecondary: "text-slate-400",
  },
};

export function ThemeProvider({ children }) {
  const [themeName, setThemeName] = useState(() => {
    return localStorage.getItem("mv_theme") || "dark";
  });

  const theme = THEMES[themeName] || THEMES.dark;

  const toggleTheme = () => {
    const next = themeName === "dark" ? "midnight" : "dark";
    setThemeName(next);
    localStorage.setItem("mv_theme", next);
  };

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", themeName);
  }, [themeName]);

  return (
    <ThemeContext.Provider value={{ theme, themeName, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}

export default ThemeContext;
