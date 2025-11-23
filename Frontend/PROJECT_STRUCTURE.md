# Frontend Project Structure

```
Frontend/
│
├── 📄 Configuration Files
│   ├── package.json              # Dependencies & scripts
│   ├── vite.config.js            # Vite build configuration
│   ├── tailwind.config.js        # Tailwind CSS theme & plugins
│   ├── postcss.config.js         # PostCSS with Tailwind
│   ├── eslint.config.js          # ESLint rules
│   └── .gitignore                # Git ignore patterns
│
├── 📄 Documentation
│   ├── README.md                 # Quick start guide
│   ├── DOCUMENTATION.md          # Comprehensive docs
│   └── SETUP_COMPLETE.md         # Setup summary
│
├── 🌐 public/
│   └── vite.svg                  # Vite logo (favicon)
│
├── 📦 src/
│   │
│   ├── 🧩 components/            # Reusable UI Components
│   │   ├── Navbar.jsx           # Navigation bar with mobile menu
│   │   ├── Footer.jsx           # Footer with links & social
│   │   ├── FeatureCard.jsx      # Feature display card
│   │   └── WeatherWidget.jsx    # Weather dashboard widget
│   │
│   ├── 📄 pages/                 # Route-based Pages
│   │   ├── Home.jsx             # Landing page
│   │   │   ├── Hero section
│   │   │   ├── Weather preview
│   │   │   ├── 6 feature cards
│   │   │   ├── Stats section
│   │   │   └── CTA sections
│   │   │
│   │   ├── Features.jsx         # Features detail page
│   │   │   ├── Detailed features
│   │   │   ├── Technology stack
│   │   │   ├── Use cases
│   │   │   └── CTA section
│   │   │
│   │   ├── About.jsx            # About page
│   │   │   ├── Mission & Vision
│   │   │   ├── Company story
│   │   │   ├── Values
│   │   │   └── Team expertise
│   │   │
│   │   └── Contact.jsx          # Contact page
│   │       ├── Contact form
│   │       ├── Contact info
│   │       ├── Business hours
│   │       └── FAQ section
│   │
│   ├── App.jsx                   # Main app with routing
│   ├── main.jsx                  # React entry point
│   └── index.css                 # Global styles + Tailwind
│
├── 📂 node_modules/              # Dependencies (329 packages)
└── index.html                    # HTML template

```

## 🎨 Component Hierarchy

```
App.jsx
├── Router
    ├── Navbar (on all pages)
    │   ├── Logo
    │   ├── Desktop Nav Links
    │   ├── Mobile Menu Button
    │   └── Mobile Menu (conditional)
    │
    ├── Routes
    │   ├── Home
    │   │   ├── Hero Section
    │   │   ├── Weather Dashboard Preview
    │   │   ├── Feature Cards Grid (6x FeatureCard)
    │   │   ├── Stats Section
    │   │   └── CTA Section
    │   │
    │   ├── Features
    │   │   ├── Header
    │   │   ├── Detailed Features Grid (6x FeatureCard)
    │   │   ├── Technology Stack
    │   │   ├── Use Cases Grid
    │   │   └── CTA Banner
    │   │
    │   ├── About
    │   │   ├── Header
    │   │   ├── Mission & Vision Cards
    │   │   ├── Story Section
    │   │   ├── Values Grid
    │   │   ├── Expertise Cards
    │   │   └── CTA Section
    │   │
    │   └── Contact
    │       ├── Header
    │       ├── Contact Info Sidebar
    │       ├── Contact Form
    │       └── FAQ Section
    │
    └── Footer (on all pages)
        ├── Brand Section
        ├── Quick Links
        ├── Services
        ├── Contact Info
        ├── Social Links
        └── Copyright
```

## 🎯 Page Routes

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | Home.jsx | Landing page with hero and features |
| `/features` | Features.jsx | Detailed features and use cases |
| `/about` | About.jsx | Company information and values |
| `/contact` | Contact.jsx | Contact form and information |

## 📦 Key Dependencies

### Production
- `react` (18.3.1) - UI library
- `react-dom` (18.3.1) - React DOM renderer
- `react-router-dom` (6.28.0) - Routing
- `lucide-react` (0.462.0) - Icons
- `clsx` (2.1.1) - Conditional classnames

### Development
- `vite` (6.0.1) - Build tool
- `@vitejs/plugin-react` (4.3.4) - React plugin
- `tailwindcss` (3.4.15) - CSS framework
- `autoprefixer` (10.4.20) - CSS prefixing
- `eslint` (9.15.0) - Linting

## 🎨 Styling Strategy

### Tailwind Utilities
- Responsive breakpoints: `sm:`, `md:`, `lg:`, `xl:`
- Custom color palette in `tailwind.config.js`
- Gradient backgrounds: `from-blue-50 via-white to-sky-50`
- Shadow utilities: `shadow-md`, `shadow-lg`, `shadow-xl`
- Transition classes for smooth animations

### Design Tokens
```javascript
// Primary Colors (tailwind.config.js)
primary: {
  50: '#e6f7ff',
  100: '#bae7ff',
  200: '#91d5ff',
  // ... up to 900
}
```

## 🔌 Backend Integration Points

### API Endpoints (To Be Connected)
```javascript
// Example integration structure
const API_BASE = 'http://localhost:5000/api'

// Weather data
GET  ${API_BASE}/weather/current?location=xyz
GET  ${API_BASE}/weather/forecast?location=xyz

// Advisory
GET  ${API_BASE}/advisory/personalized?userId=123

// Contact
POST ${API_BASE}/contact/submit

// User
POST ${API_BASE}/auth/login
POST ${API_BASE}/auth/register
```

## 📱 Responsive Features

- Mobile-first design approach
- Hamburger menu for mobile navigation
- Grid layouts that adapt: `grid-cols-1 md:grid-cols-2 lg:grid-cols-3`
- Touch-friendly button sizes (min 44px)
- Readable text sizes on all devices
- Optimized images for different screen sizes

## ⚡ Performance Features

- Vite's fast HMR (Hot Module Replacement)
- Code splitting by route
- Optimized production builds
- Lazy loading ready
- Minimal bundle size
- Tree-shaking enabled

## 🛠️ Development Workflow

1. **Start Dev Server**: `npm run dev`
2. **Edit Components**: Hot reload on save
3. **Check Errors**: ESLint feedback in editor
4. **Build**: `npm run build`
5. **Preview**: `npm run preview`

## 📊 Build Output

Production build creates:
```
dist/
├── assets/
│   ├── index-[hash].js    # Main bundle
│   ├── index-[hash].css   # Styles
│   └── vendor-[hash].js   # Dependencies
└── index.html             # Entry HTML
```

## 🎓 Learning Resources

- **React**: https://react.dev
- **Vite**: https://vitejs.dev
- **Tailwind CSS**: https://tailwindcss.com
- **React Router**: https://reactrouter.com
- **Lucide Icons**: https://lucide.dev

---

**Status**: ✅ Complete and Ready for Backend Integration
**Version**: 1.0.0
**Last Updated**: November 2025
