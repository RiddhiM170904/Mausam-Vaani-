import { createContext, useContext, useEffect, useState } from "react";
import { isSupabaseConfigured, supabase } from "../services/supabaseClient";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem("mv_user");
    if (stored) {
      try {
        setUser(JSON.parse(stored));
      } catch {
        localStorage.removeItem("mv_user");
      }
    }
    setLoading(false);
  }, []);

  const refreshProfile = async () => {
    if (!user?.id || !supabase) return null;
    const { data, error } = await supabase
      .from("users")
      .select("*")
      .eq("id", user.id)
      .maybeSingle();

    if (error) {
      console.error("Profile refresh failed:", error.message);
      return null;
    }

    const updatedUser = data || null;

    if (updatedUser) {
      setUser(updatedUser);
      localStorage.setItem("mv_user", JSON.stringify(updatedUser));
    }
    return updatedUser;
  };

  const login = (userData) => {
    setUser(userData);
    localStorage.setItem("mv_user", JSON.stringify(userData));
  };

  const logout = async () => {
    setUser(null);
    localStorage.removeItem("mv_user");
  };

  const isLoggedIn = !!user;

  return (
    <AuthContext.Provider
      value={{ user, loading, isLoggedIn, login, logout, refreshProfile }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

export default AuthContext;
