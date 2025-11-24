"""
Gemini AI integration for personalized weather insights.

This module uses Google's Gemini API to generate contextual
and personalized weather advice based on user profession and context.
"""

import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)


def generate_personalized_insight(params, api_key):
    """
    Generate personalized weather insight using Gemini.
    
    Args:
        params: Dictionary with:
            - location: Location name
            - condition: Weather condition
            - temperature: Temperature in Celsius
            - humidity: Humidity percentage
            - rainfall: Rainfall in mm
            - wind_speed: Wind speed in km/h
            - user_profession: User's profession
            - user_context: Additional user context (dict)
        api_key: Gemini API key
    
    Returns:
        Personalized insight text
    """
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Use Gemini 1.5 Flash for fast responses
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Build prompt
    prompt = _build_prompt(params)
    
    try:
        # Generate response
        logger.info("Calling Gemini API for insight generation")
        response = model.generate_content(prompt)
        
        insight = response.text.strip()
        
        logger.info("Insight generated successfully")
        return insight
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        # Fallback to basic insight
        return _generate_fallback_insight(params)


def _build_prompt(params):
    """Build prompt for Gemini based on user context."""
    
    # Extract parameters
    location = params.get('location', 'your location')
    condition = params.get('condition', 'Normal')
    temp = params.get('temperature', 0)
    humidity = params.get('humidity', 0)
    rainfall = params.get('rainfall', 0)
    wind_speed = params.get('wind_speed', 0)
    profession = params.get('user_profession', 'General')
    context = params.get('user_context', {})
    
    # Build base prompt
    prompt = f"""You are a helpful weather assistant for India providing actionable weather advice.

Location: {location}
Current Weather:
- Condition: {condition}
- Temperature: {temp}°C
- Humidity: {humidity}%
- Rainfall: {rainfall}mm
- Wind Speed: {wind_speed} km/h

User Profile:
- Profession: {profession}
"""
    
    # Add context-specific information
    if context:
        prompt += "\nAdditional Context:\n"
        for key, value in context.items():
            prompt += f"- {key}: {value}\n"
    
    # Add profession-specific instructions
    profession_prompts = {
        'Farmer': """
Give SHORT actionable advice for farmers (2-3 sentences maximum):
- Best times for farming activities (sowing, harvesting, irrigation)
- Warnings about pest/disease risks
- Equipment/crop protection recommendations
Be specific and practical. Use emojis appropriately.""",
        
        'Commuter': """
Give SHORT travel advice for commuters (2-3 sentences maximum):
- Traffic/road condition warnings
- Best travel times
- Safety precautions
Be concise and practical. Use emojis appropriately.""",
        
        'Construction Worker': """
Give SHORT advice for construction workers (2-3 sentences maximum):
- Work schedule recommendations
- Safety warnings (heat, rain, wind)
- Material handling advice
Be specific about timing. Use emojis appropriately.""",
        
        'Outdoor Sports': """
Give SHORT advice for outdoor activities (2-3 sentences maximum):
- Best times for outdoor activities
- Safety precautions
- Hydration/protection recommendations
Be practical. Use emojis appropriately.""",
        
        'General': """
Give SHORT general weather advice (2-3 sentences maximum):
- What to expect today
- What to carry (umbrella, water, etc.)
- Safety precautions if needed
Be helpful and concise. Use emojis appropriately."""
    }
    
    # Add profession-specific prompt
    specific_prompt = profession_prompts.get(profession, profession_prompts['General'])
    prompt += f"\n{specific_prompt}"
    
    prompt += "\n\nProvide ONLY the actionable advice, nothing else. Maximum 3 sentences."
    
    return prompt


def _generate_fallback_insight(params):
    """Generate basic insight when Gemini API fails."""
    
    condition = params.get('condition', 'Normal')
    temp = params.get('temperature', 0)
    rainfall = params.get('rainfall', 0)
    profession = params.get('user_profession', 'General')
    
    # Basic insights based on conditions
    insights = []
    
    # Temperature-based
    if temp > 35:
        insights.append("🌡️ Very hot weather expected. Stay hydrated and avoid outdoor activities during peak hours (12-4 PM).")
    elif temp < 15:
        insights.append("🥶 Cold weather. Wear warm clothing.")
    
    # Rainfall-based
    if rainfall > 5:
        insights.append("🌧️ Heavy rain expected. Carry umbrella and avoid waterlogged areas.")
    elif rainfall > 0:
        insights.append("☔ Light rain expected. Carry an umbrella.")
    
    # Profession-specific
    if profession == 'Farmer' and rainfall > 5:
        insights.append("🚫 Postpone pesticide spraying and outdoor harvesting.")
    elif profession == 'Commuter' and rainfall > 0:
        insights.append("🚗 Expect traffic delays due to rain.")
    elif profession == 'Construction Worker':
        if rainfall > 0:
            insights.append("⚠️ Halt outdoor construction work.")
        elif temp > 35:
            insights.append("💧 Take frequent breaks and stay hydrated.")
    
    # Default message if no specific insights
    if not insights:
        insights.append("✅ Weather conditions are favorable for regular activities.")
    
    return " ".join(insights[:3])  # Return max 3 insights


# Test function
if __name__ == "__main__":
    print("Gemini integration module ready!")
    print("\nExample usage:")
    print("""
    from gemini_integration import generate_personalized_insight
    
    params = {
        'location': 'Delhi',
        'condition': 'Heavy Rain',
        'temperature': 28,
        'humidity': 85,
        'rainfall': 10,
        'wind_speed': 12,
        'user_profession': 'Farmer',
        'user_context': {'crop': 'Rice'}
    }
    
    insight = generate_personalized_insight(params, api_key='YOUR_API_KEY')
    print(insight)
    """)
