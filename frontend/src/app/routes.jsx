import { Routes, Route } from "react-router-dom";
import Home from "../pages/Home";
import Forecast from "../pages/Forecast";
import MapPage from "../pages/Map";
import Alerts from "../pages/Alerts";
import AIInsights from "../pages/AIInsights";
import AIChat from "../pages/AIChat";
import Login from "../pages/Login";
import Signup from "../pages/Signup";
import Profile from "../pages/Profile";

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/forecast" element={<Forecast />} />
      <Route path="/map" element={<MapPage />} />
      <Route path="/alerts" element={<Alerts />} />
      <Route path="/insights" element={<AIInsights />} />
      <Route path="/assistant" element={<AIChat />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/profile" element={<Profile />} />
    </Routes>
  );
}
