import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { hashPassword, isSupabaseConfigured, supabase } from '../services/supabaseClient';
import { 
  User, 
  Phone, 
  Globe, 
  UserCog, 
  CloudRain, 
  Clock, 
  MapPin, 
  Bell,
  ChevronRight,
  ChevronLeft,
  Check,
  Smartphone,
  Car,
  Briefcase,
  HardHat,
  Tractor,
  Package,
  Users,
  GraduationCap,
  Store,
  MoreHorizontal
} from 'lucide-react';

const Signup = () => {
  const navigate = useNavigate();
  const { login } = useAuth();

  // Form state
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState({});
  
  // Form data
  const [formData, setFormData] = useState({
    // Step 1: Basic Info
    name: '',
    phone: '',
    password: '',
    language: 'en',
    
    // Step 2: Persona
    persona: '',
    otherPersonaText: '',
    
    // Step 3: Weather Risks
    weatherRisks: [],
    
    // Step 4: Schedule
    activeHours: {
      startTime: '09:00',
      endTime: '18:00',
      enabled: true
    },
    
    // Step 5: Location
    locations: [],
    useCurrentLocation: true,
    
    // Step 6: Notifications
    notificationPreferences: 'severe_only'
  });

  const totalSteps = 6;

  // Language options
  const languages = [
    { code: 'en', name: 'English', nativeName: 'English' },
    { code: 'hi', name: 'Hindi', nativeName: 'हिंदी' },
    { code: 'mr', name: 'Marathi', nativeName: 'मराठी' },
    { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்' },
    { code: 'te', name: 'Telugu', nativeName: 'తెలుగు' },
    { code: 'bn', name: 'Bengali', nativeName: 'বাংলা' },
    { code: 'gu', name: 'Gujarati', nativeName: 'ગુજરાતી' },
    { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ' },
    { code: 'ml', name: 'Malayalam', nativeName: 'മലയാളം' },
    { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ' }
  ];

  // Persona options
  const personas = [
    { id: 'general', label: 'General citizen', icon: <User className="w-6 h-6" />, description: 'Everyday weather needs' },
    { id: 'driver', label: 'Driver / Traveler', icon: <Car className="w-6 h-6" />, description: 'Road conditions & visibility' },
    { id: 'worker', label: 'Worker / Construction', icon: <HardHat className="w-6 h-6" />, description: 'Outdoor work safety' },
    { id: 'office_employee', label: 'Office employee', icon: <Briefcase className="w-6 h-6" />, description: 'Daily commute planning' },
    { id: 'farmer', label: 'Farmer', icon: <Tractor className="w-6 h-6" />, description: 'Crop & irrigation planning' },
    { id: 'delivery', label: 'Delivery / Logistics', icon: <Package className="w-6 h-6" />, description: 'Delivery route planning' },
    { id: 'senior_citizen', label: 'Senior citizen', icon: <Users className="w-6 h-6" />, description: 'Health-focused alerts' },
    { id: 'student', label: 'Student', icon: <GraduationCap className="w-6 h-6" />, description: 'School & activity planning' },
    { id: 'business_owner', label: 'Shop / Business owner', icon: <Store className="w-6 h-6" />, description: 'Business operation planning' },
    { id: 'other', label: 'Other', icon: <MoreHorizontal className="w-6 h-6" />, description: 'Custom description' }
  ];

  // Weather risks
  const weatherRisks = [
    { id: 'heavy_rain', label: 'Heavy rain', description: 'Flooding, waterlogging alerts' },
    { id: 'flood', label: 'Flood/waterlogging', description: 'Severe water accumulation' },
    { id: 'heatwave', label: 'Heatwave', description: 'Extreme temperature warnings' },
    { id: 'strong_winds', label: 'Strong winds', description: 'High wind speed alerts' },
    { id: 'fog', label: 'Fog', description: 'Low visibility conditions' },
    { id: 'cold_waves', label: 'Cold waves', description: 'Extreme cold temperatures' },
    { id: 'air_quality', label: 'Air quality', description: 'Pollution & AQI alerts' },
    { id: 'storms', label: 'Storms', description: 'Thunderstorms & cyclones' },
    { id: 'all', label: 'All weather events', description: 'Complete weather monitoring' }
  ];

  // Notification preferences
  const notificationOptions = [
    { id: 'severe_only', label: 'Severe alerts only', description: 'Critical weather warnings' },
    { id: 'all_updates', label: 'All updates', description: 'Comprehensive weather information' },
    { id: 'daily_summary', label: 'Daily summary only', description: 'Once-a-day weather overview' },
    { id: 'none', label: 'No notifications', description: 'Manual app checking only' }
  ];

  // Get user's location
  const getCurrentLocation = () => {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation not supported'));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude
          });
        },
        (error) => reject(error),
        { enableHighAccuracy: true, timeout: 10000 }
      );
    });
  };

  // Update form data
  const updateFormData = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ''
      }));
    }
  };

  // Validate current step
  const validateStep = (step) => {
    const newErrors = {};

    switch (step) {
      case 1:
        if (!formData.name.trim()) newErrors.name = 'Name is required';
        if (!formData.phone.trim()) newErrors.phone = 'Phone number is required';
        else if (!/^[6-9]\d{9}$/.test(formData.phone)) newErrors.phone = 'Enter valid 10-digit phone number';
        if (!formData.password) newErrors.password = 'Password is required';
        else if (formData.password.length < 6) newErrors.password = 'Password must be at least 6 characters';
        break;
      
      case 2:
        if (!formData.persona) newErrors.persona = 'Please select your role';
        if (formData.persona === 'other' && !formData.otherPersonaText.trim()) {
          newErrors.otherPersonaText = 'Please describe your work/role';
        }
        break;
      
      case 3:
        if (formData.weatherRisks.length === 0) newErrors.weatherRisks = 'Please select at least one weather concern';
        break;
      
      case 5:
        if (!formData.useCurrentLocation && formData.locations.length === 0) {
          newErrors.locations = 'Please add at least one location';
        }
        break;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle next step
  const handleNext = () => {
    if (validateStep(currentStep)) {
      if (currentStep < totalSteps) {
        setCurrentStep(prev => prev + 1);
      } else {
        handleSubmit();
      }
    }
  };

  // Handle previous step
  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep(prev => prev - 1);
    }
  };

  // Handle weather risk selection
  const toggleWeatherRisk = (riskId) => {
    if (riskId === 'all') {
      // If 'all' is selected, clear others and select all
      if (formData.weatherRisks.includes('all')) {
        updateFormData('weatherRisks', []);
      } else {
        updateFormData('weatherRisks', ['all']);
      }
    } else {
      // Remove 'all' if other specific risks are selected
      const newRisks = formData.weatherRisks.filter(risk => risk !== 'all');
      
      if (newRisks.includes(riskId)) {
        updateFormData('weatherRisks', newRisks.filter(risk => risk !== riskId));
      } else {
        updateFormData('weatherRisks', [...newRisks, riskId]);
      }
    }
  };

  // Handle location setup
  const handleLocationSetup = async () => {
    if (!formData.useCurrentLocation) {
      return formData.locations?.[0] || null;
    }

    try {
      const coords = await getCurrentLocation();
      
      // Add current location to locations array
      const currentLocation = {
        name: 'Current Location',
        city: 'Current Location',
        type: 'current',
        coordinates: coords,
        isPrimary: true
      };
      
      updateFormData('locations', [currentLocation]);
      return currentLocation;
    } catch (error) {
      console.error('Location error:', error);
      return null;
    }
  };

  // Handle form submission
  const handleSubmit = async () => {
    if (!validateStep(6)) return;

    setIsLoading(true);
    
    try {
      if (!isSupabaseConfigured || !supabase) {
        setErrors({ general: 'Supabase is not configured. Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.' });
        return;
      }

      const primaryLocation = await handleLocationSetup();
      const locationsForProfile = formData.useCurrentLocation
        ? (primaryLocation ? [primaryLocation] : [])
        : formData.locations;
      const resolvedPrimaryLocation = primaryLocation || locationsForProfile[0] || null;
      const passwordHash = await hashPassword(formData.password);

      const { data, error } = await supabase
        .from('users')
        .insert({
          name: formData.name,
          phone: formData.phone,
          password_hash: passwordHash,
          language: formData.language,
          persona: formData.persona,
          other_persona_text: formData.otherPersonaText,
          weather_risks: formData.weatherRisks,
          active_hours: formData.activeHours,
          use_current_location: formData.useCurrentLocation,
          locations: locationsForProfile,
          location: resolvedPrimaryLocation,
          notification_preferences: formData.notificationPreferences,
          planner_profile_completed: false,
          role: 'user',
        })
        .select('*')
        .single();

      if (error) {
        setErrors({ general: error.message || 'Registration failed' });
        return;
      }

      login(data);
      navigate('/planner');
    } catch (error) {
      console.error('Registration error:', error);
      setErrors({ general: 'Network error. Please check your connection.' });
    } finally {
      setIsLoading(false);
    }
  };

  // Render step indicator
  const renderStepIndicator = () => (
    <div className="flex items-center justify-center mb-8">
      {Array.from({ length: totalSteps }, (_, index) => {
        const stepNumber = index + 1;
        const isActive = stepNumber === currentStep;
        const isCompleted = stepNumber < currentStep;
        
        return (
          <div key={stepNumber} className="flex items-center">
            <div className={`
              w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all
              ${isCompleted ? 'bg-green-500 text-white' : 
                isActive ? 'bg-blue-500 text-white' : 
                'bg-gray-700 text-gray-400'}
            `}>
              {isCompleted ? <Check className="w-4 h-4" /> : stepNumber}
            </div>
            
            {stepNumber < totalSteps && (
              <div className={`w-8 h-0.5 mx-2 transition-all ${
                stepNumber < currentStep ? 'bg-green-500' : 'bg-gray-700'
              }`} />
            )}
          </div>
        );
      })}
    </div>
  );

  // Render step 1: Basic Info
  const renderStep1 = () => (
    <div className="space-y-6">
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl">
          <User className="w-8 h-8 text-white" />
        </div>
        <h2 className="mb-2 text-2xl font-bold text-white">Welcome to Mausam Vaani</h2>
        <p className="text-gray-400">Let's start with your basic information</p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block mb-2 text-sm font-medium text-gray-300">Full Name</label>
          <input
            type="text"
            value={formData.name}
            onChange={(e) => updateFormData('name', e.target.value)}
            className="w-full px-4 py-3 text-white placeholder-gray-500 bg-gray-800 border border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter your full name"
          />
          {errors.name && <p className="mt-1 text-sm text-red-400">{errors.name}</p>}
        </div>

        <div>
          <label className="block mb-2 text-sm font-medium text-gray-300">Phone Number</label>
          <div className="relative">
            <Phone className="absolute w-5 h-5 text-gray-400 transform -translate-y-1/2 left-3 top-1/2" />
            <input
              type="tel"
              value={formData.phone}
              onChange={(e) => updateFormData('phone', e.target.value.replace(/\D/g, ''))}
              className="w-full py-3 pl-12 pr-4 text-white placeholder-gray-500 bg-gray-800 border border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="10-digit mobile number"
              maxLength="10"
            />
          </div>
          {errors.phone && <p className="mt-1 text-sm text-red-400">{errors.phone}</p>}
        </div>

        <div>
          <label className="block mb-2 text-sm font-medium text-gray-300">Password</label>
          <input
            type="password"
            value={formData.password}
            onChange={(e) => updateFormData('password', e.target.value)}
            className="w-full px-4 py-3 text-white placeholder-gray-500 bg-gray-800 border border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Create a password (min 6 characters)"
          />
          {errors.password && <p className="mt-1 text-sm text-red-400">{errors.password}</p>}
        </div>

        <div>
          <label className="block mb-2 text-sm font-medium text-gray-300">Preferred Language</label>
          <div className="relative">
            <Globe className="absolute w-5 h-5 text-gray-400 transform -translate-y-1/2 left-3 top-1/2" />
            <select
              value={formData.language}
              onChange={(e) => updateFormData('language', e.target.value)}
              className="w-full py-3 pl-12 pr-4 text-white bg-gray-800 border border-gray-700 appearance-none rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {languages.map(lang => (
                <option key={lang.code} value={lang.code}>
                  {lang.name} ({lang.nativeName})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </div>
  );

  // Render step 2: Persona
  const renderStep2 = () => (
    <div className="space-y-6">
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl">
          <UserCog className="w-8 h-8 text-white" />
        </div>
        <h2 className="mb-2 text-2xl font-bold text-white">What describes you best?</h2>
        <p className="text-gray-400">This helps us provide relevant weather advice</p>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {personas.map(persona => (
          <button
            key={persona.id}
            onClick={() => updateFormData('persona', persona.id)}
            className={`p-4 rounded-xl border text-left transition-all ${
              formData.persona === persona.id
                ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-600'
            }`}
          >
            <div className="flex items-center gap-3">
              <div className={`${formData.persona === persona.id ? 'text-blue-400' : 'text-gray-400'}`}>
                {persona.icon}
              </div>
              <div className="flex-1">
                <div className="font-medium">{persona.label}</div>
                <div className="text-sm text-gray-500">{persona.description}</div>
              </div>
            </div>
          </button>
        ))}
      </div>

      {formData.persona === 'other' && (
        <div className="mt-4">
          <label className="block mb-2 text-sm font-medium text-gray-300">Describe your work/role</label>
          <input
            type="text"
            value={formData.otherPersonaText}
            onChange={(e) => updateFormData('otherPersonaText', e.target.value)}
            className="w-full px-4 py-3 text-white placeholder-gray-500 bg-gray-800 border border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="e.g., Doctor, Teacher, etc."
            maxLength="100"
          />
          {errors.otherPersonaText && <p className="mt-1 text-sm text-red-400">{errors.otherPersonaText}</p>}
        </div>
      )}

      {errors.persona && <p className="text-sm text-red-400">{errors.persona}</p>}
    </div>
  );

  // Render step 3: Weather Risks
  const renderStep3 = () => (
    <div className="space-y-6">
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-green-500 to-blue-600 rounded-2xl">
          <CloudRain className="w-8 h-8 text-white" />
        </div>
        <h2 className="mb-2 text-2xl font-bold text-white">Weather Concerns</h2>
        <p className="text-gray-400">What weather situations affect you the most?</p>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {weatherRisks.map(risk => (
          <button
            key={risk.id}
            onClick={() => toggleWeatherRisk(risk.id)}
            className={`p-4 rounded-xl border text-left transition-all ${
              formData.weatherRisks.includes(risk.id)
                ? 'border-green-500 bg-green-500/10 text-green-400'
                : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-600'
            }`}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">{risk.label}</div>
                <div className="text-sm text-gray-500">{risk.description}</div>
              </div>
              <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                formData.weatherRisks.includes(risk.id)
                  ? 'border-green-500 bg-green-500'
                  : 'border-gray-600'
              }`}>
                {formData.weatherRisks.includes(risk.id) && <Check className="w-3 h-3 text-white" />}
              </div>
            </div>
          </button>
        ))}
      </div>

      {errors.weatherRisks && <p className="text-sm text-red-400">{errors.weatherRisks}</p>}
    </div>
  );

  // Render step 4: Schedule
  const renderStep4 = () => (
    <div className="space-y-6">
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-orange-500 to-red-600 rounded-2xl">
          <Clock className="w-8 h-8 text-white" />
        </div>
        <h2 className="mb-2 text-2xl font-bold text-white">Daily Schedule</h2>
        <p className="text-gray-400">When do you usually go out or work?</p>
      </div>

      <div className="space-y-4">
        <div className="flex items-center gap-3 mb-4">
          <input
            type="checkbox"
            id="enableSchedule"
            checked={formData.activeHours.enabled}
            onChange={(e) => updateFormData('activeHours', { 
              ...formData.activeHours, 
              enabled: e.target.checked 
            })}
            className="w-5 h-5 text-blue-500 bg-gray-800 border-gray-600 rounded focus:ring-blue-500"
          />
          <label htmlFor="enableSchedule" className="text-gray-300">
            Send alerts only during active hours
          </label>
        </div>

        {formData.activeHours.enabled && (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block mb-2 text-sm font-medium text-gray-300">Start Time</label>
              <input
                type="time"
                value={formData.activeHours.startTime}
                onChange={(e) => updateFormData('activeHours', { 
                  ...formData.activeHours, 
                  startTime: e.target.value 
                })}
                className="w-full px-4 py-3 text-white bg-gray-800 border border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block mb-2 text-sm font-medium text-gray-300">End Time</label>
              <input
                type="time"
                value={formData.activeHours.endTime}
                onChange={(e) => updateFormData('activeHours', { 
                  ...formData.activeHours, 
                  endTime: e.target.value 
                })}
                className="w-full px-4 py-3 text-white bg-gray-800 border border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        )}

        <div className="p-4 border bg-blue-500/10 border-blue-500/30 rounded-xl">
          <p className="text-sm text-blue-300">
            <strong>Smart Scheduling:</strong> We'll send important alerts only during your active hours. 
            Emergency weather warnings will still reach you anytime.
          </p>
        </div>
      </div>
    </div>
  );

  // Render step 5: Location
  const renderStep5 = () => (
    <div className="space-y-6">
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-teal-500 to-cyan-600 rounded-2xl">
          <MapPin className="w-8 h-8 text-white" />
        </div>
        <h2 className="mb-2 text-2xl font-bold text-white">Location Settings</h2>
        <p className="text-gray-400">How should we track your location for weather updates?</p>
      </div>

      <div className="space-y-4">
        <div className="grid grid-cols-1 gap-3">
          <button
            onClick={() => updateFormData('useCurrentLocation', true)}
            className={`p-4 rounded-xl border text-left transition-all ${
              formData.useCurrentLocation
                ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-600'
            }`}
          >
            <div className="flex items-center gap-3">
              <Smartphone className="w-5 h-5" />
              <div>
                <div className="font-medium">Use current location (Recommended)</div>
                <div className="text-sm text-gray-500">Automatically detect your location for accurate weather</div>
              </div>
            </div>
          </button>

          <button
            onClick={() => updateFormData('useCurrentLocation', false)}
            className={`p-4 rounded-xl border text-left transition-all ${
              !formData.useCurrentLocation
                ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-600'
            }`}
          >
            <div className="flex items-center gap-3">
              <MapPin className="w-5 h-5" />
              <div>
                <div className="font-medium">Set fixed locations</div>
                <div className="text-sm text-gray-500">Add home, work, and other fixed locations</div>
              </div>
            </div>
          </button>
        </div>

        {!formData.useCurrentLocation && (
          <div className="p-4 mt-4 border bg-yellow-500/10 border-yellow-500/30 rounded-xl">
            <p className="text-sm text-yellow-300">
              <strong>Note:</strong> Manual location setup will be available after signup. 
              You can always change this setting later in your profile.
            </p>
          </div>
        )}

        {errors.locations && <p className="text-sm text-red-400">{errors.locations}</p>}
      </div>
    </div>
  );

  // Render step 6: Notifications
  const renderStep6 = () => (
    <div className="space-y-6">
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl">
          <Bell className="w-8 h-8 text-white" />
        </div>
        <h2 className="mb-2 text-2xl font-bold text-white">Notification Preferences</h2>
        <p className="text-gray-400">How often would you like to receive weather updates?</p>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {notificationOptions.map(option => (
          <button
            key={option.id}
            onClick={() => updateFormData('notificationPreferences', option.id)}
            className={`p-4 rounded-xl border text-left transition-all ${
              formData.notificationPreferences === option.id
                ? 'border-purple-500 bg-purple-500/10 text-purple-400'
                : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-600'
            }`}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">{option.label}</div>
                <div className="text-sm text-gray-500">{option.description}</div>
              </div>
              <div className={`w-5 h-5 rounded-full border-2 ${
                formData.notificationPreferences === option.id
                  ? 'border-purple-500 bg-purple-500'
                  : 'border-gray-600'
              }`}>
                {formData.notificationPreferences === option.id && (
                  <div className="w-full h-full transform scale-50 bg-white rounded-full"></div>
                )}
              </div>
            </div>
          </button>
        ))}
      </div>

      <div className="p-4 border bg-green-500/10 border-green-500/30 rounded-xl">
        <p className="text-sm text-green-300">
          <strong>Almost done!</strong> You can always adjust these settings later in your profile. 
          Click "Complete Setup" to finish creating your account.
        </p>
      </div>
    </div>
  );

  // Get current step content
  const getCurrentStepContent = () => {
    switch (currentStep) {
      case 1: return renderStep1();
      case 2: return renderStep2();
      case 3: return renderStep3();
      case 4: return renderStep4();
      case 5: return renderStep5();
      case 6: return renderStep6();
      default: return renderStep1();
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4 bg-gradient-to-br from-black via-gray-950 to-black">
      <div className="w-full max-w-md">
        {/* Progress indicator */}
        {renderStepIndicator()}

        {/* Form content */}
        <div className="p-6 border border-gray-800 bg-gradient-to-br from-gray-900/80 to-black/80 backdrop-blur-sm rounded-2xl">
          {getCurrentStepContent()}

          {/* Error display */}
          {errors.general && (
            <div className="p-3 mt-4 border rounded-lg bg-red-500/10 border-red-500/30">
              <p className="text-sm text-red-400">{errors.general}</p>
            </div>
          )}

          {/* Navigation buttons */}
          <div className="flex gap-3 mt-8">
            {currentStep > 1 && (
              <button
                onClick={handlePrevious}
                disabled={isLoading}
                className="flex items-center justify-center flex-1 gap-2 px-4 py-3 text-gray-300 transition-colors bg-gray-800 border border-gray-700 rounded-xl hover:bg-gray-700 disabled:opacity-50"
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </button>
            )}
            
            <button
              onClick={handleNext}
              disabled={isLoading}
              className="flex items-center justify-center flex-1 gap-2 px-4 py-3 text-white transition-all bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl hover:from-blue-600 hover:to-purple-700 disabled:opacity-50"
            >
              {isLoading ? (
                <div className="w-4 h-4 border-2 border-white rounded-full border-t-transparent animate-spin"></div>
              ) : (
                <>
                  {currentStep === totalSteps ? 'Complete Setup' : 'Next'}
                  {currentStep < totalSteps && <ChevronRight className="w-4 h-4" />}
                </>
              )}
            </button>
          </div>

          {/* Skip option for optional steps */}
          {[3, 4, 6].includes(currentStep) && (
            <button
              onClick={() => setCurrentStep(prev => prev + 1)}
              className="w-full mt-2 text-sm text-gray-400 transition-colors hover:text-gray-300"
            >
              Skip for now
            </button>
          )}
        </div>

        {/* Login link */}
        <div className="mt-6 text-center">
          <p className="text-sm text-gray-400">
            Already have an account?{' '}
            <button 
              onClick={() => navigate('/login')}
              className="text-blue-400 transition-colors hover:text-blue-300"
            >
              Sign in
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Signup;
