"""
Data Preprocessing Pipeline for Weather Time-Series Data

This module handles:
- Loading weather data from CSV
- Data cleaning and validation
- Feature engineering (time features, cyclical encoding)
- Train/validation/test split
- Sequence creation for TFT model
- Data normalization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class WeatherDataProcessor:
    """Process raw weather data for TFT model training."""
    
    def __init__(self, csv_path, target_features=None):
        """
        Initialize data processor.
        
        Args:
            csv_path: Path to weather CSV file
            target_features: List of features to predict (default: all weather params)
        """
        self.csv_path = csv_path
        self.target_features = target_features or [
            'temperature', 'humidity', 'wind_speed', 
            'rainfall', 'pressure', 'cloud_cover'
        ]
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.feature_columns = None
        self.data = None
    
    def load_data(self):
        """Load and validate weather data from CSV."""
        print(f"Loading data from {self.csv_path}...")
        
        # Read CSV
        df = pd.read_csv(self.csv_path)
        
        # Required columns
        required_cols = [
            'timestamp', 'city', 'latitude', 'longitude',
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'pressure', 'cloud_cover'
        ]
        
        # Validate columns
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by city and timestamp
        df = df.sort_values(['city', 'timestamp']).reset_index(drop=True)
        
        print(f"Loaded {len(df)} records for {df['city'].nunique()} cities")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        self.data = df
        return df
    
    def clean_data(self):
        """Clean and handle missing values."""
        if self.data is None:
            raise ValueError("Load data first using load_data()")
        
        print("\nCleaning data...")
        initial_rows = len(self.data)
        
        # Check for missing values
        missing_counts = self.data.isnull().sum()
        if missing_counts.any():
            print("Missing values found:")
            print(missing_counts[missing_counts > 0])
            
            # Forward fill within each city group
            self.data = self.data.groupby('city').apply(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            ).reset_index(drop=True)
        
        # Remove outliers (optional - using IQR method)
        for col in ['temperature', 'humidity', 'wind_speed', 'rainfall', 'pressure']:
            Q1 = self.data[col].quantile(0.01)
            Q3 = self.data[col].quantile(0.99)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Clip outliers instead of removing
            self.data[col] = self.data[col].clip(lower_bound, upper_bound)
        
        # Ensure non-negative values for certain features
        self.data['humidity'] = self.data['humidity'].clip(0, 100)
        self.data['rainfall'] = self.data['rainfall'].clip(0, None)
        self.data['wind_speed'] = self.data['wind_speed'].clip(0, None)
        self.data['cloud_cover'] = self.data['cloud_cover'].clip(0, 100)
        
        final_rows = len(self.data)
        print(f"Data cleaning complete. Rows: {initial_rows} → {final_rows}")
    
    def engineer_features(self):
        """Create time-based and cyclical features."""
        if self.data is None:
            raise ValueError("Load and clean data first")
        
        print("\nEngineering features...")
        
        # Extract time features
        self.data['year'] = self.data['timestamp'].dt.year
        self.data['month'] = self.data['timestamp'].dt.month
        self.data['day'] = self.data['timestamp'].dt.day
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        self.data['day_of_year'] = self.data['timestamp'].dt.dayofyear
        
        # Cyclical encoding for periodic features
        # Hour (24-hour cycle)
        self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        
        # Month (12-month cycle)
        self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)
        
        # Day of week (7-day cycle)
        self.data['dow_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['dow_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        
        # Season indicator
        self.data['season'] = self.data['month'].apply(self._get_season)
        
        print(f"Added {len(['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'season'])} time features")
    
    @staticmethod
    def _get_season(month):
        """Get season from month (India-specific)."""
        if month in [3, 4, 5]:
            return 0  # Summer
        elif month in [6, 7, 8, 9]:
            return 1  # Monsoon
        elif month in [10, 11]:
            return 2  # Post-monsoon
        else:
            return 3  # Winter
    
    def prepare_sequences(self, encoder_steps=168, forecast_steps=24):
        """
        Create sequences for time-series modeling.
        
        Args:
            encoder_steps: Number of historical time steps (default: 168 = 1 week)
            forecast_steps: Number of future time steps to predict
        
        Returns:
            Dictionary with 'features', 'targets', 'locations', 'timestamps'
        """
        if self.data is None:
            raise ValueError("Process data first")
        
        print(f"\nCreating sequences (encoder: {encoder_steps}, forecast: {forecast_steps})...")
        
        # Define feature columns (inputs to model)
        self.feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'pressure', 'cloud_cover', 'latitude', 'longitude',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'dow_sin', 'dow_cos', 'season'
        ]
        
        sequences = []
        targets = []
        locations = []
        timestamps = []
        
        # Group by city to maintain continuity
        for city, group in self.data.groupby('city'):
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            total_steps = encoder_steps + forecast_steps
            
            # Create sliding window sequences
            for i in range(len(group) - total_steps + 1):
                # Extract sequence window
                window = group.iloc[i:i + total_steps]
                
                # Features (entire window)
                seq_features = window[self.feature_columns].values
                
                # Targets (only forecast window)
                seq_targets = window.iloc[encoder_steps:][self.target_features].values
                
                # Metadata
                seq_location = {
                    'city': city,
                    'latitude': window.iloc[0]['latitude'],
                    'longitude': window.iloc[0]['longitude']
                }
                seq_timestamp = window.iloc[encoder_steps]['timestamp']
                
                sequences.append(seq_features)
                targets.append(seq_targets)
                locations.append(seq_location)
                timestamps.append(seq_timestamp)
        
        print(f"Created {len(sequences)} sequences")
        
        return {
            'features': np.array(sequences),
            'targets': np.array(targets),
            'locations': locations,
            'timestamps': timestamps
        }
    
    def normalize_data(self, features, targets, fit=True):
        """
        Normalize features and targets.
        
        Args:
            features: Input features array
            targets: Target values array
            fit: Whether to fit scaler (True for train, False for val/test)
        
        Returns:
            Normalized features and targets
        """
        batch_size, seq_len, num_features = features.shape
        _, target_len, num_targets = targets.shape
        
        # Reshape for scaling
        features_reshape = features.reshape(-1, num_features)
        targets_reshape = targets.reshape(-1, num_targets)
        
        if fit:
            features_norm = self.feature_scaler.fit_transform(features_reshape)
            targets_norm = self.target_scaler.fit_transform(targets_reshape)
        else:
            features_norm = self.feature_scaler.transform(features_reshape)
            targets_norm = self.target_scaler.transform(targets_reshape)
        
        # Reshape back
        features_norm = features_norm.reshape(batch_size, seq_len, num_features)
        targets_norm = targets_norm.reshape(batch_size, target_len, num_targets)
        
        return features_norm, targets_norm
    
    def split_data(self, features, targets, locations, timestamps, 
                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split data into train, validation, and test sets.
        
        Args:
            features, targets, locations, timestamps: Data arrays
            train_ratio, val_ratio, test_ratio: Split ratios
        
        Returns:
            Dictionary with train, val, test splits
        """
        total_samples = len(features)
        
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        # Time-based split (no shuffling to prevent data leakage)
        train_data = {
            'features': features[:train_end],
            'targets': targets[:train_end],
            'locations': locations[:train_end],
            'timestamps': timestamps[:train_end]
        }
        
        val_data = {
            'features': features[train_end:val_end],
            'targets': targets[train_end:val_end],
            'locations': locations[train_end:val_end],
            'timestamps': timestamps[train_end:val_end]
        }
        
        test_data = {
            'features': features[val_end:],
            'targets': targets[val_end:],
            'locations': locations[val_end:],
            'timestamps': timestamps[val_end:]
        }
        
        print(f"\nData split:")
        print(f"  Train: {len(train_data['features'])} samples")
        print(f"  Val:   {len(val_data['features'])} samples")
        print(f"  Test:  {len(test_data['features'])} samples")
        
        return train_data, val_data, test_data


class WeatherDataset(Dataset):
    """PyTorch Dataset for weather time-series."""
    
    def __init__(self, features, targets):
        """
        Initialize dataset.
        
        Args:
            features: Numpy array of shape (samples, seq_len, num_features)
            targets: Numpy array of shape (samples, forecast_len, num_targets)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def create_dataloaders(train_data, val_data, test_data, batch_size=32, num_workers=0):
    """
    Create PyTorch DataLoaders.
    
    Args:
        train_data, val_data, test_data: Data dictionaries
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = WeatherDataset(train_data['features'], train_data['targets'])
    val_dataset = WeatherDataset(val_data['features'], val_data['targets'])
    test_dataset = WeatherDataset(test_data['features'], test_data['targets'])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing Data Preprocessing Pipeline...")
    print("=" * 60)
    
    # Note: This requires actual data file
    # For demonstration, we'll show the expected usage
    
    print("\nExpected Usage:")
    print("""
    # Initialize processor
    processor = WeatherDataProcessor('data/weather_time_series.csv')
    
    # Load and process data
    processor.load_data()
    processor.clean_data()
    processor.engineer_features()
    
    # Create sequences
    data = processor.prepare_sequences(encoder_steps=168, forecast_steps=24)
    
    # Split data
    train_data, val_data, test_data = processor.split_data(
        data['features'], data['targets'], 
        data['locations'], data['timestamps']
    )
    
    # Normalize
    train_data['features'], train_data['targets'] = processor.normalize_data(
        train_data['features'], train_data['targets'], fit=True
    )
    val_data['features'], val_data['targets'] = processor.normalize_data(
        val_data['features'], val_data['targets'], fit=False
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size=32
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    """)
    
    print("\nPreprocessing module ready!")
