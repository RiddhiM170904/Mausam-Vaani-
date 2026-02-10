import { BrowserRouter } from "react-router-dom";
import { useEffect } from "react";
import { AuthProvider } from "./context/AuthContext";
import { ThemeProvider } from "./context/ThemeContext";
import Navbar from "./components/Navbar";
import BottomNav from "./components/BottomNav";
import InstallPrompt from "./components/InstallPrompt";
import OfflineIndicator from "./components/OfflineIndicator";
import AppRoutes from "./app/routes";
import useLocation from "./hooks/useLocation";
import { registerSW } from "./utils/pwa";

function App() {
  useEffect(() => {
    // Register service worker for PWA functionality
    registerSW();

    // Track app launch for analytics
    if (window.matchMedia('(display-mode: standalone)').matches) {
      console.log('App launched in standalone mode');
      // Track PWA usage if analytics is available
      if (window.gtag) {
        window.gtag('event', 'pwa_launch', {
          event_category: 'PWA',
          event_label: 'App launched from home screen'
        });
      }
    }
  }, []);

  return (
    <AuthProvider>
      <ThemeProvider>
        <BrowserRouter>
          <div className="flex flex-col min-h-screen bg-gradient-to-br from-black via-gray-950 to-black">
            <Navbar />

            {/* Offline Indicator */}
            <OfflineIndicator />

            {/* Main content */}
            <main className="flex-1 w-full px-4 py-6 pb-24 mx-auto max-w-7xl sm:px-6 lg:px-8 md:pb-6">
              <AppRoutes />
            </main>

            <BottomNav />
            
            {/* PWA Install Prompt */}
            <InstallPrompt />
          </div>
        </BrowserRouter>
      </ThemeProvider>
    </AuthProvider>
  );
}

export default App;
