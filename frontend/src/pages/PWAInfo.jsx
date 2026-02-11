import { useState, useEffect } from 'react';
import { Download, Wifi, Bell, Smartphone, Monitor, RefreshCw, Zap } from 'lucide-react';
import InstallButton from '../components/InstallButton';
import { isStandalone, getNetworkStatus } from '../utils/pwa';

const PWAInfo = () => {
  const [networkStatus, setNetworkStatus] = useState(getNetworkStatus());
  const [isAppInstalled, setIsAppInstalled] = useState(false);

  useEffect(() => {
    setIsAppInstalled(isStandalone());

    const updateNetworkStatus = () => {
      setNetworkStatus(getNetworkStatus());
    };

    window.addEventListener('online', updateNetworkStatus);
    window.addEventListener('offline', updateNetworkStatus);

    return () => {
      window.removeEventListener('online', updateNetworkStatus);
      window.removeEventListener('offline', updateNetworkStatus);
    };
  }, []);

  const features = [
    {
      icon: <Smartphone className="w-6 h-6" />,
      title: "Native App Experience",
      description: "Install on your home screen for quick access like a native app"
    },
    {
      icon: <Wifi className="w-6 h-6" />,
      title: "Works Offline",
      description: "Access your weather data even when you're not connected to the internet"
    },
    {
      icon: <Bell className="w-6 h-6" />,
      title: "Push Notifications",
      description: "Get instant weather alerts and updates directly to your device"
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: "Fast Loading",
      description: "Cached resources ensure lightning-fast app startup and navigation"
    },
    {
      icon: <RefreshCw className="w-6 h-6" />,
      title: "Auto Updates",
      description: "Always stay up-to-date with the latest features and improvements"
    },
    {
      icon: <Monitor className="w-6 h-6" />,
      title: "Cross Platform",
      description: "Works seamlessly on mobile, tablet, and desktop devices"
    }
  ];

  return (
    <div className="min-h-screen p-4 bg-gradient-to-br from-black via-gray-950 to-black">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <div className="flex items-center justify-center w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl">
            <Download className="w-10 h-10 text-white" />
          </div>
          <h1 className="mb-4 text-4xl font-bold text-white">
            Install Mausam Vaani
          </h1>
          <p className="max-w-2xl mx-auto text-lg text-gray-300">
            Get the best weather experience with our Progressive Web App. 
            Fast, reliable, and works offline.
          </p>
        </div>

        {/* App Status */}
        <div className="p-6 mb-8 border border-gray-700 bg-gradient-to-r from-gray-900/50 to-gray-800/50 backdrop-blur-sm rounded-2xl">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <div className="text-center">
              <div className={`w-4 h-4 rounded-full mx-auto mb-2 ${
                isAppInstalled ? 'bg-green-500' : 'bg-gray-500'
              }`}></div>
              <p className="text-sm text-gray-300">
                App Status: <span className={isAppInstalled ? 'text-green-400' : 'text-gray-400'}>
                  {isAppInstalled ? 'Installed' : 'Not Installed'}
                </span>
              </p>
            </div>
            <div className="text-center">
              <div className={`w-4 h-4 rounded-full mx-auto mb-2 ${
                networkStatus.online ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
              <p className="text-sm text-gray-300">
                Network: <span className={networkStatus.online ? 'text-green-400' : 'text-red-400'}>
                  {networkStatus.online ? 'Online' : 'Offline'}
                </span>
              </p>
            </div>
            <div className="text-center">
              <div className="w-4 h-4 mx-auto mb-2 bg-blue-500 rounded-full"></div>
              <p className="text-sm text-gray-300">
                PWA: <span className="text-blue-400">Enabled</span>
              </p>
            </div>
          </div>
        </div>

        {/* Install Section */}
        {!isAppInstalled && (
          <div className="p-8 mb-8 text-center border bg-gradient-to-r from-blue-900/20 to-purple-900/20 border-blue-500/30 rounded-2xl">
            <h2 className="mb-4 text-2xl font-bold text-white">
              Ready to Install?
            </h2>
            <p className="mb-6 text-gray-300">
              Install Mausam Vaani for the best experience. It's free and takes just a few seconds.
            </p>
            <InstallButton variant="default" size="large" />
          </div>
        )}

        {/* Features Grid */}
        <div className="grid grid-cols-1 gap-6 mb-8 md:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="p-6 transition-colors border border-gray-700 bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-sm rounded-xl hover:border-blue-500/30"
            >
              <div className="mb-4 text-blue-400">
                {feature.icon}
              </div>
              <h3 className="mb-2 text-lg font-semibold text-white">
                {feature.title}
              </h3>
              <p className="text-sm text-gray-300">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        {/* Installation Instructions */}
        <div className="p-8 border border-gray-700 bg-gradient-to-r from-gray-900/50 to-gray-800/50 backdrop-blur-sm rounded-2xl">
          <h2 className="mb-6 text-2xl font-bold text-white">How to Install</h2>
          
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
            {/* Mobile Instructions */}
            <div>
              <h3 className="flex items-center gap-2 mb-4 text-lg font-semibold text-white">
                <Smartphone className="w-5 h-5" />
                Mobile (iOS/Android)
              </h3>
              <ol className="space-y-3 text-gray-300">
                <li className="flex items-start gap-3">
                  <span className="flex items-center justify-center flex-shrink-0 w-6 h-6 text-sm font-medium text-white bg-blue-500 rounded-full">1</span>
                  <span>Tap the install button above or the one in the navigation</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex items-center justify-center flex-shrink-0 w-6 h-6 text-sm font-medium text-white bg-blue-500 rounded-full">2</span>
                  <span>Follow your browser's installation prompts</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex items-center justify-center flex-shrink-0 w-6 h-6 text-sm font-medium text-white bg-blue-500 rounded-full">3</span>
                  <span>Find the app icon on your home screen</span>
                </li>
              </ol>
            </div>

            {/* Desktop Instructions */}
            <div>
              <h3 className="flex items-center gap-2 mb-4 text-lg font-semibold text-white">
                <Monitor className="w-5 h-5" />
                Desktop
              </h3>
              <ol className="space-y-3 text-gray-300">
                <li className="flex items-start gap-3">
                  <span className="flex items-center justify-center flex-shrink-0 w-6 h-6 text-sm font-medium text-white bg-purple-500 rounded-full">1</span>
                  <span>Click the install button in your browser's address bar</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex items-center justify-center flex-shrink-0 w-6 h-6 text-sm font-medium text-white bg-purple-500 rounded-full">2</span>
                  <span>Or use the install button in our navigation menu</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex items-center justify-center flex-shrink-0 w-6 h-6 text-sm font-medium text-white bg-purple-500 rounded-full">3</span>
                  <span>Access the app from your desktop or applications menu</span>
                </li>
              </ol>
            </div>
          </div>

          <div className="p-4 mt-8 border rounded-lg bg-blue-500/10 border-blue-500/30">
            <p className="text-sm text-blue-300">
              <strong>Note:</strong> PWA installation is supported on Chrome, Firefox, Edge, and Safari. 
              The install option will appear automatically when available.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PWAInfo;