"""
Comprehensive Dataset Combination and Cleaning Script

This script:
1. Loads all 4 Excel files (Weather, Location, Astronomical, Air Quality)
2. Merges them on 'last_updated_epoch'
3. Cleans and validates data
4. Converts to required format for TFT model training
5. Saves combined dataset as CSV
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DatasetCombiner:
    """Combine and clean multiple weather-related datasets."""
    
    def __init__(self, data_dir):
        """
        Initialize the combiner.
        
        Args:
            data_dir: Path to directory containing Excel files
        """
        self.data_dir = data_dir
        self.combined_df = None
        
    def load_datasets(self):
        """Load all Excel datasets."""
        print("=" * 80)
        print("LOADING DATASETS")
        print("=" * 80)
        
        # 1. Weather Data
        print("\n1. Loading Weather data...")
        weather_path = os.path.join(self.data_dir, "Weather data.xlsx")
        self.weather_df = pd.read_excel(weather_path)
        print(f"   ✓ Loaded {len(self.weather_df)} weather records")
        print(f"   Columns: {list(self.weather_df.columns)}")
        
        # 2. Location Information
        print("\n2. Loading Location data...")
        location_path = os.path.join(self.data_dir, "Location information.xlsx")
        self.location_df = pd.read_excel(location_path)
        print(f"   ✓ Loaded {len(self.location_df)} location records")
        print(f"   Columns: {list(self.location_df.columns)}")
        
        # 3. Astronomical Data
        print("\n3. Loading Astronomical data...")
        astro_path = os.path.join(self.data_dir, "Astronomical.xlsx")
        self.astro_df = pd.read_excel(astro_path)
        print(f"   ✓ Loaded {len(self.astro_df)} astronomical records")
        print(f"   Columns: {list(self.astro_df.columns)}")
        
        # 4. Air Quality Data
        print("\n4. Loading Air Quality data...")
        aqi_path = os.path.join(self.data_dir, "Air quality information.xlsx")
        self.aqi_df = pd.read_excel(aqi_path)
        print(f"   ✓ Loaded {len(self.aqi_df)} air quality records")
        print(f"   Columns: {list(self.aqi_df.columns)}")
        
        return self
    
    def merge_datasets(self):
        """Merge all datasets on last_updated_epoch."""
        print("\n" + "=" * 80)
        print("MERGING DATASETS")
        print("=" * 80)
        
        # Start with weather data as base
        print("\nMerging on 'last_updated_epoch'...")
        
        # Merge weather + location
        merged = pd.merge(
            self.weather_df,
            self.location_df,
            on='last_updated_epoch',
            how='inner',
            suffixes=('', '_loc')
        )
        print(f"✓ Weather + Location: {len(merged)} records")
        
        # Merge + astronomical
        merged = pd.merge(
            merged,
            self.astro_df,
            on='last_updated_epoch',
            how='inner',
            suffixes=('', '_astro')
        )
        print(f"✓ + Astronomical: {len(merged)} records")
        
        # Merge + air quality
        merged = pd.merge(
            merged,
            self.aqi_df,
            on='last_updated_epoch',
            how='inner',
            suffixes=('', '_aqi')
        )
        print(f"✓ + Air Quality: {len(merged)} records")
        
        print(f"\n✓ Final merged dataset: {len(merged)} records × {len(merged.columns)} columns")
        
        self.combined_df = merged
        return self
    
    def clean_data(self):
        """Clean and validate the combined dataset."""
        print("\n" + "=" * 80)
        print("CLEANING DATA")
        print("=" * 80)
        
        df = self.combined_df.copy()
        initial_rows = len(df)
        
        # 1. Handle duplicate columns (from merge suffixes)
        print("\n1. Removing duplicate columns...")
        # Keep the first occurrence of duplicate columns
        cols_to_drop = [col for col in df.columns if col.endswith('_loc') or col.endswith('_astro') or col.endswith('_aqi')]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        print(f"   ✓ Dropped {len(cols_to_drop)} duplicate columns")
        
        # 2. Convert epoch to datetime
        print("\n2. Converting timestamps...")
        df['timestamp'] = pd.to_datetime(df['last_updated_epoch'], unit='s')
        df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
        print(f"   ✓ Created timestamp column")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # 3. Remove duplicates
        print("\n3. Removing duplicate records...")
        duplicates = df.duplicated(subset=['last_updated_epoch', 'location_name'], keep='first')
        df = df[~duplicates]
        print(f"   ✓ Removed {duplicates.sum()} duplicate records")
        
        # 4. Sort by location and timestamp
        print("\n4. Sorting data...")
        df = df.sort_values(['location_name', 'timestamp']).reset_index(drop=True)
        print(f"   ✓ Sorted by location and timestamp")
        
        # 5. Handle missing values
        print("\n5. Handling missing values...")
        missing_before = df.isnull().sum().sum()
        
        # Forward fill within each location
        df = df.groupby('location_name').apply(
            lambda group: group.fillna(method='ffill').fillna(method='bfill')
        ).reset_index(drop=True)
        
        missing_after = df.isnull().sum().sum()
        print(f"   ✓ Reduced missing values: {missing_before} → {missing_after}")
        
        if missing_after > 0:
            print(f"\n   Remaining missing values by column:")
            missing_cols = df.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            for col, count in missing_cols.items():
                print(f"     - {col}: {count} ({count/len(df)*100:.2f}%)")
        
        # 6. Remove outliers
        print("\n6. Removing outliers...")
        outliers_removed = 0
        
        # Define valid ranges
        valid_ranges = {
            'temperature_celsius': (-20, 55),
            'humidity': (0, 100),
            'wind_kph': (0, 200),
            'pressure_mb': (900, 1100),
            'precip_mm': (0, 500),
            'cloud': (0, 100),
            'uv_index': (0, 15),
            'air_quality_PM2.5': (0, 1000),
            'air_quality_PM10': (0, 2000),
        }
        
        for col, (min_val, max_val) in valid_ranges.items():
            if col in df.columns:
                before = len(df)
                df = df[(df[col] >= min_val) & (df[col] <= max_val)]
                removed = before - len(df)
                outliers_removed += removed
                if removed > 0:
                    print(f"   - {col}: removed {removed} outliers")
        
        print(f"   ✓ Total outliers removed: {outliers_removed}")
        
        # 7. Data type conversions
        print("\n7. Converting data types...")
        # Ensure numeric columns are numeric
        numeric_cols = [
            'temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb',
            'precip_mm', 'cloud', 'uv_index', 'latitude', 'longitude',
            'air_quality_PM2.5', 'air_quality_PM10', 'air_quality_Carbon_Monoxide',
            'air_quality_Ozone', 'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide',
            'moon_illumination'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"   ✓ Converted {len(numeric_cols)} columns to numeric")
        
        self.combined_df = df
        print(f"\n✓ Cleaning complete: {initial_rows} → {len(df)} records ({initial_rows - len(df)} removed)")
        
        return self
    
    def transform_to_model_format(self):
        """Transform to the required format for TFT model."""
        print("\n" + "=" * 80)
        print("TRANSFORMING TO MODEL FORMAT")
        print("=" * 80)
        
        df = self.combined_df.copy()
        
        # Create the required column mapping
        print("\n1. Mapping columns to model format...")
        
        model_df = pd.DataFrame()
        
        # Basic columns
        model_df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        model_df['city'] = df['location_name']
        model_df['latitude'] = df['latitude']
        model_df['longitude'] = df['longitude']
        
        # Weather columns
        model_df['temperature'] = df['temperature_celsius']
        model_df['humidity'] = df['humidity']
        model_df['wind_speed'] = df['wind_kph']
        model_df['rainfall'] = df['precip_mm']
        model_df['pressure'] = df['pressure_mb']
        model_df['cloud_cover'] = df['cloud']
        
        # Air Quality columns
        model_df['aqi'] = df['air_quality_us-epa-index'] * 50  # Convert EPA index to AQI scale
        model_df['pm25'] = df['air_quality_PM2.5']
        model_df['pm10'] = df['air_quality_PM10']
        model_df['co'] = df['air_quality_Carbon_Monoxide']
        model_df['no2'] = df['air_quality_Nitrogen_dioxide']
        model_df['o3'] = df['air_quality_Ozone']
        model_df['so2'] = df['air_quality_Sulphur_dioxide']
        
        # Astronomical columns
        model_df['sunrise'] = df['sunrise']
        model_df['sunset'] = df['sunset']
        model_df['moonrise'] = df['moonrise']
        model_df['moonset'] = df['moonset']
        model_df['moon_phase'] = df['moon_illumination'] / 100  # Convert to 0-1 scale
        
        # Calculate day length
        def calculate_day_length(row):
            try:
                if pd.notna(row['sunrise']) and pd.notna(row['sunset']):
                    sunrise = pd.to_datetime(row['sunrise'], format='%I:%M %p', errors='coerce')
                    sunset = pd.to_datetime(row['sunset'], format='%I:%M %p', errors='coerce')
                    if pd.notna(sunrise) and pd.notna(sunset):
                        delta = sunset - sunrise
                        return delta.total_seconds() / 3600  # Convert to hours
            except:
                pass
            return np.nan
        
        print("   - Calculating day length...")
        model_df['day_length'] = df.apply(calculate_day_length, axis=1)
        
        # Fill missing day_length with average (around 12 hours)
        if model_df['day_length'].isnull().any():
            model_df['day_length'].fillna(12.0, inplace=True)
        
        print(f"\n✓ Transformed to model format")
        print(f"   Final columns: {list(model_df.columns)}")
        print(f"   Total: {len(model_df.columns)} columns")
        
        self.model_df = model_df
        return self
    
    def validate_final_data(self):
        """Validate the final dataset."""
        print("\n" + "=" * 80)
        print("VALIDATION")
        print("=" * 80)
        
        df = self.model_df
        
        print(f"\n✓ Dataset Shape: {df.shape}")
        print(f"✓ Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"✓ Cities: {df['city'].nunique()} unique cities")
        print(f"  {', '.join(df['city'].unique())}")
        
        # Check missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"\n⚠ Missing values found:")
            for col, count in missing[missing > 0].items():
                print(f"  - {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            print(f"\n✓ No missing values!")
        
        # Data statistics
        print(f"\n✓ Data Statistics:")
        print(df[['temperature', 'humidity', 'rainfall', 'aqi', 'pm25']].describe())
        
        return self
    
    def save_dataset(self, output_path=None):
        """Save the combined and cleaned dataset."""
        if output_path is None:
            output_path = os.path.join(self.data_dir, "combined_weather_dataset.csv")
        
        print("\n" + "=" * 80)
        print("SAVING DATASET")
        print("=" * 80)
        
        self.model_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Dataset saved to: {output_path}")
        print(f"✓ Total records: {len(self.model_df)}")
        print(f"✓ Total columns: {len(self.model_df.columns)}")
        print(f"✓ File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return output_path


def main():
    """Main execution function."""
    # Set data directory
    data_dir = r"C:\personal dg\github_repo\Mausam-Vaani-\AI-Backend\data"
    
    print("\n" + "=" * 80)
    print("WEATHER DATASET COMBINATION AND CLEANING")
    print("=" * 80)
    print(f"Data Directory: {data_dir}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize combiner
        combiner = DatasetCombiner(data_dir)
        
        # Execute pipeline
        combiner.load_datasets()
        combiner.merge_datasets()
        combiner.clean_data()
        combiner.transform_to_model_format()
        combiner.validate_final_data()
        output_file = combiner.save_dataset()
        
        print("\n" + "=" * 80)
        print("✓ SUCCESS!")
        print("=" * 80)
        print(f"\nYour combined dataset is ready for training:")
        print(f"📁 {output_file}")
        print(f"\nNext steps:")
        print(f"1. Review the data: Open combined_weather_dataset.csv")
        print(f"2. Validate the data: python scripts/validate_data.py --input {output_file}")
        print(f"3. Train the model: python models/train.py")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR!")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
