import { useState, useEffect } from 'react';
import { Download, Smartphone } from 'lucide-react';

const InstallButton = ({ variant = 'default', size = 'medium' }) => {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [isInstallable, setIsInstallable] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    // Check if app is already installed
    if (window.matchMedia('(display-mode: standalone)').matches || 
        window.navigator.standalone === true) {
      setIsInstalled(true);
      return;
    }

    const handleBeforeInstallPrompt = (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
      setIsInstallable(true);
    };

    const handleAppInstalled = () => {
      setIsInstalled(true);
      setIsInstallable(false);
      setDeferredPrompt(null);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt) return;

    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    
    if (outcome === 'accepted') {
      console.log('User accepted the install prompt');
    }

    setDeferredPrompt(null);
    setIsInstallable(false);
  };

  // Don't show button if not installable or already installed
  if (!isInstallable || isInstalled) return null;

  const baseClasses = "flex items-center gap-2 transition-colors rounded-lg font-medium";
  
  const variants = {
    default: "bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white",
    outline: "border border-gray-600 text-gray-300 hover:bg-gray-800",
    ghost: "text-gray-400 hover:text-white hover:bg-gray-800"
  };

  const sizes = {
    small: "px-3 py-1.5 text-sm",
    medium: "px-4 py-2 text-sm",
    large: "px-6 py-3 text-base"
  };

  const iconSizes = {
    small: "w-3 h-3",
    medium: "w-4 h-4", 
    large: "w-5 h-5"
  };

  return (
    <button
      onClick={handleInstall}
      className={`${baseClasses} ${variants[variant]} ${sizes[size]}`}
      title="Install Mausam Vaani as an app"
    >
      <Download className={iconSizes[size]} />
      <span className="hidden sm:inline">Install App</span>
      <span className="sm:hidden">Install</span>
    </button>
  );
};

export default InstallButton;