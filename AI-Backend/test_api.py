"""
Quick test script for Mausam-Vaani API
Run this to verify your API is working correctly
"""

import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*80)
    print("Testing Health Check...")
    print("="*80)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction():
    """Test weather prediction endpoint"""
    print("\n" + "="*80)
    print("Testing Weather Prediction...")
    print("="*80)
    
    # Test data
    test_cases = [
        {
            "name": "Farmer in Delhi",
            "data": {
                "weather_input": {
                    "latitude": 28.6139,
                    "longitude": 77.2090,
                    "location_name": "Delhi"
                },
                "user_context": {
                    "profession": "Farmer",
                    "additional_context": {"crop": "Rice"}
                },
                "forecast_hours": 24
            }
        },
        {
            "name": "Commuter in Mumbai",
            "data": {
                "weather_input": {
                    "latitude": 19.0760,
                    "longitude": 72.8777,
                    "location_name": "Mumbai"
                },
                "user_context": {
                    "profession": "Commuter"
                },
                "forecast_hours": 12
            }
        },
        {
            "name": "General User in Bangalore",
            "data": {
                "weather_input": {
                    "latitude": 12.9716,
                    "longitude": 77.5946,
                    "location_name": "Bangalore"
                },
                "user_context": {
                    "profession": "General"
                },
                "forecast_hours": 24
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📍 Test: {test_case['name']}")
        print("-" * 80)
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_case['data'],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n✅ SUCCESS!")
                print(f"Location: {data['location']}")
                print(f"Forecast Hours: {data['forecast_hours']}")
                print(f"\nSummary:")
                print(f"  Avg Temperature: {data['summary']['avg_temperature']:.1f}°C")
                print(f"  Total Rainfall: {data['summary']['total_rainfall']:.1f}mm")
                print(f"  Avg Humidity: {data['summary']['avg_humidity']:.1f}%")
                print(f"\nPersonalized Insight:")
                print(f"  {data['personalized_insight']}")
                print(f"\nFirst 3 Predictions:")
                for i, pred in enumerate(data['predictions'][:3]):
                    print(f"  {i+1}. {pred['timestamp']}: {pred['temperature']:.1f}°C, {pred['rainfall']:.1f}mm")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Run all tests"""
    print("="*80)
    print("🌤️  MAUSAM-VAANI API TEST SUITE")
    print("="*80)
    print(f"Testing API at: {BASE_URL}")
    
    # Test health
    health_ok = test_health()
    
    if not health_ok:
        print("\n❌ Health check failed! Make sure the server is running.")
        print("   Start server with: python app.py")
        return
    
    # Test predictions
    test_prediction()
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Check the API docs at: http://localhost:8000/docs")
    print("2. Connect your frontend to the API")
    print("3. Test with real user inputs")
    print("="*80)

if __name__ == "__main__":
    main()
