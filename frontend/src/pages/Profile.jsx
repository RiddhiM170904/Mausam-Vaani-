import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import toast from "react-hot-toast";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import GlassCard from "../components/GlassCard";
import {
  HiOutlineArrowRightOnRectangle,
  HiOutlinePaintBrush,
  HiOutlineBell,
  HiOutlineLanguage,
  HiOutlineUser,
} from "react-icons/hi2";
import {
  ensureNotificationPermission,
  getNotificationConfig,
  getNotificationStatus,
  saveNotificationConfig,
} from "../services/localNotificationService";
import {
  registerPushSubscription,
  saveBackendNotificationPreference,
  unregisterPushSubscription,
} from "../services/notificationBackendService";

export default function Profile() {
  const navigate = useNavigate();
  const { user, logout, isLoggedIn } = useAuth();
  const { themeName, toggleTheme } = useTheme();
  const [notifConfig, setNotifConfig] = useState(() => getNotificationConfig());
  const [notifStatus, setNotifStatus] = useState(() => getNotificationStatus());

  if (!isLoggedIn) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex flex-col items-center justify-center py-20 text-center"
      >
        <HiOutlineUser size={48} className="text-gray-600 mb-4" />
        <h2 className="text-xl font-bold text-white mb-2">Not signed in</h2>
        <p className="text-gray-500 text-sm mb-6">Sign in to manage your profile</p>
        <button
          onClick={() => navigate("/login")}
          className="px-6 py-2.5 rounded-xl bg-indigo-500 text-white font-semibold hover:bg-indigo-600 transition-colors"
        >
          Sign In
        </button>
      </motion.div>
    );
  }

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  const refreshNotifState = () => {
    setNotifConfig(getNotificationConfig());
    setNotifStatus(getNotificationStatus());
  };

  const handleToggleNotifications = async () => {
    if (notifConfig.enabled) {
      try {
        await unregisterPushSubscription();
        await saveBackendNotificationPreference({
          userId: user?.id,
          enabled: false,
          dailyCount: notifConfig.dailyCount,
          timezone: notifConfig.timezone,
        });
      } catch (err) {
        toast.error(err?.message || "Failed to disable backend notifications");
      }

      setNotifConfig(saveNotificationConfig({ enabled: false }));
      setNotifStatus(getNotificationStatus());
      toast("App notifications disabled", { icon: "🔕" });
      return;
    }

    const granted = await ensureNotificationPermission();

    if (!granted) {
      setNotifConfig(saveNotificationConfig({ enabled: false }));
      setNotifStatus(getNotificationStatus());
      toast.error("Permission blocked. Enable notifications from browser settings.");
      refreshNotifState();
      return;
    }

    try {
      const subscriptionResult = await registerPushSubscription({ userId: user?.id });
      if (!subscriptionResult.ok) {
        throw new Error("Push subscription was not granted");
      }

      await saveBackendNotificationPreference({
        userId: user?.id,
        enabled: true,
        dailyCount: notifConfig.dailyCount,
        timezone: notifConfig.timezone,
      });

      setNotifConfig(saveNotificationConfig({ enabled: true }));
      setNotifStatus(getNotificationStatus());
      toast.success("App notifications enabled ✅");
    } catch (err) {
      setNotifConfig(saveNotificationConfig({ enabled: false }));
      setNotifStatus(getNotificationStatus());
      toast.error(err?.message || "Failed to enable backend notifications");
    }

    refreshNotifState();
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6 max-w-lg mx-auto"
    >
      <h1 className="text-2xl font-bold text-white">Profile</h1>

      {/* User info */}
      <GlassCard className="p-6" hover={false}>
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-2xl bg-indigo-500/20 flex items-center justify-center text-2xl">
            👤
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">{user?.name || "User"}</h3>
            <p className="text-sm text-gray-500">
              {user?.phone ? `+91 ${user.phone}` : "Phone not set"}
            </p>
            <span className="inline-block mt-1 px-2 py-0.5 rounded-full bg-indigo-500/15 text-indigo-300 text-xs font-medium capitalize">
              {user?.persona || "general"}
            </span>
          </div>
        </div>
      </GlassCard>

      {/* Settings */}
      <div className="space-y-2 px-1 sm:px-0">
        <h3 className="text-sm font-semibold text-gray-400 px-1">Settings</h3>

        {/* Theme */}
        <GlassCard
          className="p-4 flex items-center justify-between cursor-pointer"
          onClick={toggleTheme}
        >
          <div className="flex items-center gap-3">
            <HiOutlinePaintBrush className="text-purple-400" size={20} />
            <div>
              <p className="text-sm font-medium text-white">Theme</p>
              <p className="text-xs text-gray-500 capitalize">{themeName}</p>
            </div>
          </div>
          <span className="text-xs text-gray-500">Tap to switch</span>
        </GlassCard>

        {/* Language */}
        <GlassCard className="p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <HiOutlineLanguage className="text-blue-400" size={20} />
            <div>
              <p className="text-sm font-medium text-white">Language</p>
              <p className="text-xs text-gray-500">{user?.language === "hi" ? "हिंदी" : "English"}</p>
            </div>
          </div>
        </GlassCard>

        {/* Notifications */}
        <GlassCard className="p-4 flex items-center justify-between" hover={false}>
          <div className="flex items-start gap-3">
            <HiOutlineBell className="text-yellow-400 mt-0.5" size={20} />
            <div className="flex items-center gap-3">
              <div>
                <p className="text-sm font-medium text-white">App Notifications</p>
                <p className="text-xs text-gray-500">
                  {notifStatus.permission === "granted"
                    ? "Weather and AI alerts are active"
                    : "Enable to receive weather and AI alerts"}
                </p>
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={handleToggleNotifications}
            className={`w-10 h-5 rounded-full relative transition-colors ${
              notifConfig.enabled ? "bg-indigo-500" : "bg-gray-600"
            }`}
            aria-label="Toggle app notifications"
          >
            <span
              className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                notifConfig.enabled ? "translate-x-5" : "translate-x-0.5"
              }`}
            />
          </button>
        </GlassCard>
      </div>

      {/* Logout */}
      <button
        onClick={handleLogout}
        className="w-full py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 font-semibold hover:bg-red-500/20 transition-colors flex items-center justify-center gap-2"
      >
        <HiOutlineArrowRightOnRectangle size={18} />
        Sign Out
      </button>
    </motion.div>
  );
}
