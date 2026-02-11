import { useState, useEffect } from 'react';
import { X, Download, Smartphone, Monitor } from 'lucide-react';

const InstallPrompt = () => {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [isVisible, setIsVisible] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    // Check if app is already installed
    const checkIfInstalled = () => {
      if (window.matchMedia('(display-mode: standalone)').matches) {
        setIsInstalled(true);
        return;
      }
      
      if (window.navigator.standalone === true) {
        setIsInstalled(true);
        return;
      }
    };

    checkIfInstalled();

    // Listen for the beforeinstallprompt event
    const handleBeforeInstallPrompt = (e) => {
      // Prevent the mini-infobar from appearing on mobile
      e.preventDefault();
      // Save the event so it can be triggered later
      setDeferredPrompt(e);
      
      // Show install prompt if not already installed and user hasn't dismissed it recently
      const dismissed = localStorage.getItem('installPromptDismissed');
      const dismissedTime = dismissed ? parseInt(dismissed) : 0;
      const dayInMs = 24 * 60 * 60 * 1000;
      
      if (!isInstalled && (!dismissed || (Date.now() - dismissedTime) > dayInMs * 7)) {
        setTimeout(() => setIsVisible(true), 2000); // Show after 2 seconds
      }
    };

    // Listen for app installed event
    const handleAppInstalled = () => {
      setIsInstalled(true);
      setIsVisible(false);
      setDeferredPrompt(null);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, [isInstalled]);

  const handleInstallClick = async () => {
    if (!deferredPrompt) return;

    // Show the install prompt
    deferredPrompt.prompt();

    // Wait for the user to respond to the prompt
    const { outcome } = await deferredPrompt.userChoice;
    
    if (outcome === 'accepted') {
      console.log('User accepted the install prompt');
    } else {
      console.log('User dismissed the install prompt');
    }

    // Clear the deferredPrompt variable
    setDeferredPrompt(null);
    setIsVisible(false);
  };

  const handleDismiss = () => {
    setIsVisible(false);
    localStorage.setItem('installPromptDismissed', Date.now().toString());
  };

  const isMobile = () => {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  };

  if (!isVisible || isInstalled) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gradient-to-br from-gray-900 to-black border border-gray-700 rounded-2xl max-w-md w-full p-6 relative overflow-hidden">
        {/* Background glow effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-2xl"></div>
        
        {/* Close button */}
        <button
          onClick={handleDismiss}
          className="absolute top-4 right-4 p-1 rounded-lg hover:bg-gray-800 transition-colors"
          aria-label="Close"
        >
          <X className="w-5 h-5 text-gray-400" />
        </button>

        {/* Content */}
        <div className="relative z-10">
          {/* Icon */}
          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mb-4">
            {isMobile() ? (
              <Smartphone className="w-8 h-8 text-white" />
            ) : (
              <Monitor className="w-8 h-8 text-white" />
            )}
          </div>

          {/* Title and description */}
          <h3 className="text-xl font-bold text-white mb-2">
            Install Mausam Vaani
          </h3>
          <p className="text-gray-300 text-sm mb-6">
            Get instant access to weather updates right from your {isMobile() ? 'home screen' : 'desktop'}. 
            Works offline and sends notifications for weather alerts.
          </p>

          {/* Features */}
          <div className="space-y-2 mb-6">
            <div className="flex items-center text-sm text-gray-300">
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mr-3"></div>
              Works offline
            </div>
            <div className="flex items-center text-sm text-gray-300">
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mr-3"></div>
              Push notifications for weather alerts
            </div>
            <div className="flex items-center text-sm text-gray-300">
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mr-3"></div>
              Fast and reliable performance
            </div>
          </div>

          {/* Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleDismiss}
              className="flex-1 px-4 py-2.5 text-gray-300 border border-gray-600 rounded-xl hover:bg-gray-800 transition-colors text-sm font-medium"
            >
              Not now
            </button>
            <button
              onClick={handleInstallClick}
              className="flex-1 px-4 py-2.5 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl hover:from-blue-600 hover:to-purple-700 transition-colors text-sm font-medium flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              Install
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InstallPrompt;