import { Target, Eye, Heart, Users } from 'lucide-react'

const About = () => {
  return (
    <div className="min-h-screen py-12">
      {/* Header */}
      <section className="px-4 sm:px-6 lg:px-8 mb-16">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 mb-6">
            About Mausam Vaani
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Transforming weather data into actionable intelligence for everyone, everywhere
          </p>
        </div>
      </section>

      {/* Mission & Vision */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12">
            <div className="bg-white rounded-2xl p-8 shadow-lg">
              <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-xl mb-6">
                <Target className="h-10 w-10 text-primary-600" />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 mb-4">Our Mission</h2>
              <p className="text-gray-600 text-lg leading-relaxed">
                To democratize access to hyperlocal weather intelligence by combining cutting-edge AI, 
                multi-source data fusion, and inclusive technology. We believe everyone deserves accurate, 
                actionable weather insights regardless of their location or device.
              </p>
            </div>

            <div className="bg-white rounded-2xl p-8 shadow-lg">
              <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-xl mb-6">
                <Eye className="h-10 w-10 text-primary-600" />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 mb-4">Our Vision</h2>
              <p className="text-gray-600 text-lg leading-relaxed">
                To become the most trusted source of hyperlocal weather intelligence in India and beyond, 
                empowering individuals, businesses, and governments to make informed decisions that save 
                lives, protect livelihoods, and optimize operations.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Story */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20 bg-gradient-to-br from-blue-50 to-sky-50 py-16">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">Our Story</h2>
          <div className="bg-white rounded-2xl p-8 shadow-lg">
            <p className="text-gray-700 text-lg leading-relaxed mb-6">
              Mausam Vaani was born from a simple observation: while weather forecasts are widely available, 
              they often fail to provide the granular, actionable insights that people actually need. A farmer 
              doesn't just need to know it might rain—they need to know if it will rain on their specific field, 
              at what time, and whether they should irrigate or wait.
            </p>
            <p className="text-gray-700 text-lg leading-relaxed mb-6">
              Our founders, a team of AI researchers, meteorologists, and social entrepreneurs, came together 
              with a vision to bridge this gap. By combining advanced AI models like Temporal Fusion Transformers 
              with multi-source data fusion and natural language processing, we created a platform that doesn't 
              just predict weather—it predicts impact.
            </p>
            <p className="text-gray-700 text-lg leading-relaxed">
              Today, Mausam Vaani serves millions of users across India, from smallholder farmers to large 
              enterprises, helping them make better decisions through hyperlocal weather intelligence.
            </p>
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center">Our Values</h2>
          <div className="grid md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mx-auto mb-4">
                <Heart className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Inclusivity</h3>
              <p className="text-gray-600">
                Weather intelligence for everyone, regardless of technology access or language
              </p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mx-auto mb-4">
                <Target className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Accuracy</h3>
              <p className="text-gray-600">
                Uncompromising commitment to precision through rigorous quality control
              </p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mx-auto mb-4">
                <Users className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">User-Centric</h3>
              <p className="text-gray-600">
                Designed with and for our users, delivering insights that truly matter
              </p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mx-auto mb-4">
                <Eye className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Innovation</h3>
              <p className="text-gray-600">
                Constantly pushing boundaries with AI and technology advancement
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">Our Expertise</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white rounded-xl p-6 shadow-md text-center">
              <div className="text-4xl mb-4">🤖</div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">AI & Machine Learning</h3>
              <p className="text-gray-600">
                Experts in deep learning, time-series forecasting, and natural language processing
              </p>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-md text-center">
              <div className="text-4xl mb-4">🌡️</div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Meteorology</h3>
              <p className="text-gray-600">
                Professional meteorologists with decades of combined experience
              </p>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-md text-center">
              <div className="text-4xl mb-4">💻</div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Engineering</h3>
              <p className="text-gray-600">
                World-class engineers building scalable, reliable infrastructure
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto bg-gradient-to-r from-primary-600 to-blue-600 rounded-2xl p-12 text-center text-white shadow-2xl">
          <h2 className="text-3xl font-bold mb-4">Join Us on Our Journey</h2>
          <p className="text-xl mb-8 text-primary-100">
            Be part of the weather intelligence revolution
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-white text-primary-600 px-8 py-3 rounded-lg hover:bg-gray-100 transition-colors font-semibold">
              Contact Us
            </button>
            <button className="bg-transparent border-2 border-white text-white px-8 py-3 rounded-lg hover:bg-white hover:text-primary-600 transition-colors font-semibold">
              View Careers
            </button>
          </div>
        </div>
      </section>
    </div>
  )
}

export default About
