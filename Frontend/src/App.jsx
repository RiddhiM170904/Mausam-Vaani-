import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import { UserProvider } from './context/UserContext'
import { WeatherProvider } from './context/WeatherContext'

// Pages
import Home from './pages/Home'
import Features from './pages/Features'
import About from './pages/About'
import Contact from './pages/Contact'
import Demo from './pages/Demo'
import Dashboard from './pages/Dashboard'
import Signup from './pages/Signup'
import Login from './pages/Login'
import Planner from './pages/Planner'
import Maps from './pages/Maps'
import Community from './pages/Community'
import Settings from './pages/Settings'

// Animated page wrapper
const PageWrapper = ({ children }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    transition={{ duration: 0.3 }}
  >
    {children}
  </motion.div>
)

// Layout component that handles footer visibility
const Layout = ({ children }) => {
  const location = useLocation()
  const hideFooterPaths = ['/maps', '/signup', '/login']
  const showFooter = !hideFooterPaths.includes(location.pathname)

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900">
      <Navbar />
      <main className="flex-grow pt-16">
        <AnimatePresence mode="wait">
          {children}
        </AnimatePresence>
      </main>
      {showFooter && <Footer />}
    </div>
  )
}

function App() {
  return (
    <UserProvider>
      <WeatherProvider>
        <Router>
          <Layout>
            <Routes>
              {/* Public Routes */}
              <Route path="/" element={<PageWrapper><Home /></PageWrapper>} />
              <Route path="/features" element={<PageWrapper><Features /></PageWrapper>} />
              <Route path="/about" element={<PageWrapper><About /></PageWrapper>} />
              <Route path="/contact" element={<PageWrapper><Contact /></PageWrapper>} />
              <Route path="/demo" element={<PageWrapper><Demo /></PageWrapper>} />
              
              {/* Auth Routes */}
              <Route path="/signup" element={<PageWrapper><Signup /></PageWrapper>} />
              <Route path="/login" element={<PageWrapper><Login /></PageWrapper>} />
              
              {/* Protected Routes (Dashboard) */}
              <Route path="/dashboard" element={<PageWrapper><Dashboard /></PageWrapper>} />
              <Route path="/planner" element={<PageWrapper><Planner /></PageWrapper>} />
              <Route path="/maps" element={<PageWrapper><Maps /></PageWrapper>} />
              <Route path="/community" element={<PageWrapper><Community /></PageWrapper>} />
              <Route path="/settings" element={<PageWrapper><Settings /></PageWrapper>} />
            </Routes>
          </Layout>
        </Router>
      </WeatherProvider>
    </UserProvider>
  )
}

export default App
