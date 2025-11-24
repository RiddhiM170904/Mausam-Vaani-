"""
Data validation script to check weather data quality.

Usage:
    python scripts/validate_data.py --input data/weather_time_series.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def validate_weather_data(csv_path):
    """
    Validate weather data CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Dictionary with validation results
    """
    print(f"Validating: {csv_path}")
    print("=" * 60)
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"✓ File loaded successfully")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Check required columns
        required_cols = [
            'timestamp', 'city', 'latitude', 'longitude',
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'pressure', 'cloud_cover'
        ]
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing columns: {missing_cols}")
            return results
        
        print(f"✓ All required columns present")
        
        # Check timestamp format
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"✓ Timestamps are valid")
            
            # Check date range
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            duration_days = (max_date - min_date).days
            
            print(f"  Date range: {min_date} to {max_date}")
            print(f"  Duration: {duration_days} days ({duration_days/30:.1f} months)")
            
            results['stats']['date_range'] = {
                'start': str(min_date),
                'end': str(max_date),
                'duration_days': duration_days
            }
            
            if duration_days < 30:
                results['warnings'].append(
                    f"Dataset is only {duration_days} days. "
                    "Recommend at least 180 days (6 months)"
                )
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Invalid timestamp format: {e}")
            return results
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            missing_info = missing_counts[missing_counts > 0]
            results['warnings'].append(f"Missing values found:\n{missing_info}")
            print(f"⚠ Missing values detected:")
            for col, count in missing_info.items():
                pct = (count / len(df)) * 100
                print(f"    {col}: {count} ({pct:.1f}%)")
        else:
            print(f"✓ No missing values")
        
        # Check value ranges
        numeric_cols = [
            'temperature', 'humidity', 'wind_speed', 
            'rainfall', 'pressure', 'cloud_cover'
        ]
        
        print("\nValue Range Checks:")
        
        # Temperature
        temp_min, temp_max = df['temperature'].min(), df['temperature'].max()
        print(f"  Temperature: {temp_min:.1f}°C to {temp_max:.1f}°C")
        if temp_min < -20 or temp_max > 55:
            results['warnings'].append(
                f"Temperature out of expected range: {temp_min:.1f} to {temp_max:.1f}"
            )
        
        # Humidity
        hum_min, hum_max = df['humidity'].min(), df['humidity'].max()
        print(f"  Humidity: {hum_min:.1f}% to {hum_max:.1f}%")
        if hum_min < 0 or hum_max > 100:
            results['errors'].append("Humidity must be between 0 and 100")
            results['valid'] = False
        
        # Wind speed
        ws_min, ws_max = df['wind_speed'].min(), df['wind_speed'].max()
        print(f"  Wind Speed: {ws_min:.1f} to {ws_max:.1f} km/h")
        if ws_min < 0:
            results['errors'].append("Wind speed cannot be negative")
            results['valid'] = False
        
        # Rainfall
        rf_min, rf_max = df['rainfall'].min(), df['rainfall'].max()
        print(f"  Rainfall: {rf_min:.1f} to {rf_max:.1f} mm")
        if rf_min < 0:
            results['errors'].append("Rainfall cannot be negative")
            results['valid'] = False
        
        # Pressure
        pr_min, pr_max = df['pressure'].min(), df['pressure'].max()
        print(f"  Pressure: {pr_min:.1f} to {pr_max:.1f} hPa")
        if pr_min < 900 or pr_max > 1100:
            results['warnings'].append(
                f"Pressure out of typical range: {pr_min:.1f} to {pr_max:.1f}"
            )
        
        # Cloud cover
        cc_min, cc_max = df['cloud_cover'].min(), df['cloud_cover'].max()
        print(f"  Cloud Cover: {cc_min:.1f}% to {cc_max:.1f}%")
        if cc_min < 0 or cc_max > 100:
            results['errors'].append("Cloud cover must be between 0 and 100")
            results['valid'] = False
        
        # Check latitude/longitude
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        print(f"\nLocation Range:")
        print(f"  Latitude: {lat_min:.4f} to {lat_max:.4f}")
        print(f"  Longitude: {lon_min:.4f} to {lon_max:.4f}")
        
        if lat_min < -90 or lat_max > 90:
            results['errors'].append("Latitude must be between -90 and 90")
            results['valid'] = False
        
        if lon_min < -180 or lon_max > 180:
            results['errors'].append("Longitude must be between -180 and 180")
            results['valid'] = False
        
        # Check number of cities
        num_cities = df['city'].nunique()
        print(f"\nCities: {num_cities}")
        print(f"  {', '.join(df['city'].unique()[:10])}")
        if num_cities > 10:
            print(f"  ... and {num_cities - 10} more")
        
        results['stats']['num_cities'] = num_cities
        results['stats']['total_records'] = len(df)
        
        if num_cities < 3:
            results['warnings'].append(
                f"Only {num_cities} cities. Recommend at least 5 for better generalization"
            )
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['city', 'timestamp']).sum()
        if duplicates > 0:
            results['warnings'].append(f"Found {duplicates} duplicate (city, timestamp) pairs")
            print(f"\n⚠ {duplicates} duplicate records found")
        else:
            print(f"\n✓ No duplicate records")
        
        # Records per city
        records_per_city = df.groupby('city').size()
        print(f"\nRecords per city:")
        for city, count in records_per_city.head(10).items():
            print(f"  {city}: {count}")
        
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Validation failed: {e}")
    
    return results


def print_results(results):
    """Print validation results summary."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if results['valid']:
        print("✓ Data is VALID")
    else:
        print("✗ Data is INVALID")
    
    if results['errors']:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  ✗ {error}")
    
    if results['warnings']:
        print(f"\nWarnings ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")
    
    if not results['errors'] and not results['warnings']:
        print("\n✓ No issues found!")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Validate weather data CSV')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to weather data CSV file')
    
    args = parser.parse_args()
    
    csv_path = Path(args.input)
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return
    
    results = validate_weather_data(csv_path)
    print_results(results)
    
    if results['valid']:
        print("\n✓ Data is ready for training!")
    else:
        print("\n✗ Please fix errors before training")


if __name__ == "__main__":
    main()
