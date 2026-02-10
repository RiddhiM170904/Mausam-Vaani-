"""
🌤️ MAUSAM-VAANI - ADVANCED COLAB TRAINING SCRIPT
Complete optimized script for training Weather TFT model on Google Colab GPU

✨ ADVANCED FEATURES:
- Works with combined_weather_dataset.csv (23 features)
- Cyclical encoding for time & astronomical features
- Multi-scale temporal attention
- Learning rate scheduling with warm restarts
- Mixed precision training (FP16)
- Quantile loss for uncertainty estimation
- Advanced data augmentation
- Comprehensive evaluation metrics

📊 UPLOAD: combined_weather_dataset.csv to /content/ folder
🚀 RUN: Runtime > Run all
"""

import os
import sys
import time
import pickle
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration - OPTIMIZED FOR BEST ACCURACY
CONFIG = {
    'model': {
        'hidden_dim': 256,  # Increased for more capacity
        'num_heads': 8,  # More attention heads
        'num_layers': 3,  # Deeper network
        'forecast_horizon': 24,
        'dropout': 0.15,  # Slightly higher dropout
    },
    'data': {
        'csv_file': 'combined_weather_dataset.csv',  # Your merged dataset
        'encoder_steps': 168,  # 7 days history
        'forecast_steps': 24,   # 24 hours prediction
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
    },
    'training': {
        'epochs': 100,  # More epochs with early stopping
        'batch_size': 64,  # Larger batch for stability
        'learning_rate': 0.0005,  # Lower initial LR
        'weight_decay': 0.00005,
        'grad_clip': 1.0,
        'early_stopping_patience': 20,
        'use_mixed_precision': True,  # FP16 training
        'warmup_epochs': 5,
    },
    'advanced': {
        'use_cyclical_encoding': True,
        'use_quantile_loss': True,
        'augment_data': True,
        'use_lr_scheduler': True,
    },
    'paths': {
        'data_dir': '/content',
        'checkpoint_dir': '/content/checkpoints',
        'best_model': '/content/best_weather_model.pth',
        'results_dir': '/content/results',
    }
}

print("="*80)
print("🌤️  MAUSAM-VAANI - ADVANCED WEATHER PREDICTION TRAINING")
print("="*80)
print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🖥️  Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("="*80)

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("\n📊 STEP 1: LOADING COMBINED DATASET...")

csv_path = os.path.join(CONFIG['paths']['data_dir'], CONFIG['data']['csv_file'])

if not os.path.exists(csv_path):
    print(f"❌ ERROR: {CONFIG['data']['csv_file']} not found!")
    print(f"\n📝 INSTRUCTIONS:")
    print(f"1. Run combine_and_clean_datasets.py locally")
    print(f"2. Upload combined_weather_dataset.csv to Colab")
    print(f"3. Make sure file is in /content/ folder")
    sys.exit(1)

print(f"📁 Loading: {csv_path}")
df = pd.read_csv(csv_path)

print(f"✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"✓ Columns: {list(df.columns)}")
print(f"\n📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"🏙️  Cities: {df['city'].nunique()} ({', '.join(df['city'].unique()[:5])}...)")

# ============================================================================
# STEP 2: ADVANCED DATA PREPROCESSING
# ============================================================================
print("\n🔧 STEP 2: ADVANCED DATA PREPROCESSING...")

class AdvancedWeatherProcessor:
    """Advanced preprocessing with cyclical encoding and multi-scale features"""
    
    def __init__(self, df, config):
        self.df = df.copy()
        self.config = config
        self.feature_scalers = {}
        self.target_scaler = RobustScaler()  # Robust to outliers
        
    def process(self):
        """Complete preprocessing pipeline"""
        print("  🧹 Cleaning data...")
        self._clean_data()
        
        print("  🎨 Engineering features...")
        self._engineer_features()
        
        print("  🔄 Encoding cyclical features...")
        self._encode_cyclical_features()
        
        print("  📏 Scaling features...")
        self._scale_features()
        
        print("  📦 Creating sequences...")
        sequences = self._create_sequences()
        
        print("  ✂️  Splitting data...")
        splits = self._split_data(sequences)
        
        return splits
    
    def _clean_data(self):
        """Clean and validate data"""
        # Convert timestamp with error handling
        print(f"    - Converting timestamps...")
        
        # Try multiple timestamp formats
        if 'timestamp' in self.df.columns:
            # First, try standard pandas parsing
            self.df['timestamp'] = pd.to_datetime(
                self.df['timestamp'], 
                errors='coerce',  # Convert errors to NaT
                format='mixed'  # Infer format for each element
            )
            
            # Remove rows with invalid timestamps
            invalid_timestamps = self.df['timestamp'].isnull().sum()
            if invalid_timestamps > 0:
                print(f"    ⚠️  Removing {invalid_timestamps} rows with invalid timestamps")
                self.df = self.df[self.df['timestamp'].notna()]
        
        # Sort by city and timestamp
        self.df = self.df.sort_values(['city', 'timestamp']).reset_index(drop=True)
        
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Group by city and forward fill
        print(f"    - Handling missing values...")
        self.df[numeric_cols] = self.df.groupby('city')[numeric_cols].transform(
            lambda x: x.ffill().bfill()
        )
        
        # Fill remaining with median
        for col in numeric_cols:
            if self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Remove any remaining rows with NaN
        rows_before = len(self.df)
        self.df = self.df.dropna()
        rows_removed = rows_before - len(self.df)
        
        if rows_removed > 0:
            print(f"    ⚠️  Removed {rows_removed} rows with remaining NaN values")
        
        print(f"    ✓ Cleaned {len(self.df)} records")
    
    def _engineer_features(self):
        """Create advanced time-based and interaction features"""
        # Time features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['day_of_year'] = self.df['timestamp'].dt.dayofyear
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # Season (for India: Winter, Summer, Monsoon, Post-monsoon)
        def get_season(month):
            if month in [12, 1, 2]: return 0  # Winter
            elif month in [3, 4, 5]: return 1  # Summer
            elif month in [6, 7, 8, 9]: return 2  # Monsoon
            else: return 3  # Post-monsoon
        
        self.df['season'] = self.df['month'].apply(get_season)
        
        # Interaction features
        if 'temperature' in self.df.columns and 'humidity' in self.df.columns:
            # Heat index approximation
            self.df['heat_index'] = self.df['temperature'] + (0.5 * self.df['humidity'] / 100)
        
        # Lag features (previous hour values)
        lag_features = ['temperature', 'humidity', 'aqi', 'pm25']
        for col in lag_features:
            if col in self.df.columns:
                self.df[f'{col}_lag1h'] = self.df.groupby('city')[col].shift(1)
                self.df[f'{col}_lag24h'] = self.df.groupby('city')[col].shift(24)
        
        # Rolling statistics (24-hour window)
        for col in ['temperature', 'aqi']:
            if col in self.df.columns:
                self.df[f'{col}_rolling_mean'] = self.df.groupby('city')[col].transform(
                    lambda x: x.rolling(window=24, min_periods=1).mean()
                )
                self.df[f'{col}_rolling_std'] = self.df.groupby('city')[col].transform(
                    lambda x: x.rolling(window=24, min_periods=1).std()
                ).fillna(0)
        
        print(f"    ✓ Created {len(self.df.columns)} total features")
    
    def _encode_cyclical_features(self):
        """Encode cyclical features using sin/cos transformation"""
        if not self.config['advanced']['use_cyclical_encoding']:
            return
        
        # Hour of day (0-23)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        
        # Day of week (0-6)
        self.df['dow_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['dow_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        
        # Day of year (0-365)
        self.df['doy_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['doy_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)
        
        # Month (1-12)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        print(f"    ✓ Encoded cyclical features")
    
    def _scale_features(self):
        """Scale features using appropriate scalers"""
        # Define feature groups
        weather_features = ['temperature', 'humidity', 'wind_speed', 'rainfall', 
                           'pressure', 'cloud_cover']
        aqi_features = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'o3', 'so2']
        location_features = ['latitude', 'longitude']
        
        # Use RobustScaler for weather and AQI (handles outliers better)
        for col in weather_features + aqi_features:
            if col in self.df.columns:
                scaler = RobustScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                self.feature_scalers[col] = scaler
        
        # StandardScaler for location
        for col in location_features:
            if col in self.df.columns:
                scaler = StandardScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                self.feature_scalers[col] = scaler
        
        print(f"    ✓ Scaled {len(self.feature_scalers)} feature groups")
    
    def _create_sequences(self):
        """Create time series sequences with all features"""
        # Define all feature columns
        base_features = [
            'temperature', 'humidity', 'wind_speed', 'rainfall', 
            'pressure', 'cloud_cover'
        ]
        
        aqi_features = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'o3', 'so2']
        
        location_features = ['latitude', 'longitude']
        
        time_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                        'doy_sin', 'doy_cos', 'month_sin', 'month_cos',
                        'is_weekend', 'season']
        
        # Additional engineered features
        extra_features = ['heat_index', 'day_length']
        
        # Lag features
        lag_cols = [col for col in self.df.columns if 'lag' in col or 'rolling' in col]
        
        # Combine all input features
        input_features = []
        for feat_list in [base_features, aqi_features, location_features, 
                         time_features, extra_features, lag_cols]:
            input_features.extend([f for f in feat_list if f in self.df.columns])
        
        # Target features (what we want to predict)
        target_features = [f for f in base_features + aqi_features if f in self.df.columns]
        
        print(f"    📊 Input features ({len(input_features)}): {input_features[:10]}...")
        print(f"    🎯 Target features ({len(target_features)}): {target_features}")
        
        # Extract data
        data = self.df[input_features].values
        targets = self.df[target_features].values
        
        # Create sequences
        encoder_steps = self.config['data']['encoder_steps']
        forecast_steps = self.config['data']['forecast_steps']
        total_steps = encoder_steps + forecast_steps
        
        X, y = [], []
        
        for i in range(len(data) - total_steps):
            # Input: past encoder_steps + future forecast_steps
            X.append(data[i:i+total_steps])
            # Output: only target features for forecast_steps
            y.append(targets[i+encoder_steps:i+total_steps])
        
        return {
            'X': np.array(X, dtype=np.float32),
            'y': np.array(y, dtype=np.float32),
            'input_features': input_features,
            'target_features': target_features,
        }
    
    def _split_data(self, sequences):
        """Split into train/val/test sets"""
        X, y = sequences['X'], sequences['y']
        n = len(X)
        
        train_end = int(n * self.config['data']['train_ratio'])
        val_end = int(n * (self.config['data']['train_ratio'] + self.config['data']['val_ratio']))
        
        return {
            'train': {'X': X[:train_end], 'y': y[:train_end]},
            'val': {'X': X[train_end:val_end], 'y': y[train_end:val_end]},
            'test': {'X': X[val_end:], 'y': y[val_end:]},
            'input_features': sequences['input_features'],
            'target_features': sequences['target_features'],
            'scalers': self.feature_scalers,
        }

# Process data
processor = AdvancedWeatherProcessor(df, CONFIG)
data_splits = processor.process()

print(f"\n✓ Train: {len(data_splits['train']['X']):,} samples")
print(f"✓ Val: {len(data_splits['val']['X']):,} samples")
print(f"✓ Test: {len(data_splits['test']['X']):,} samples")
print(f"✓ Input dims: {data_splits['train']['X'].shape}")
print(f"✓ Output dims: {data_splits['train']['y'].shape}")

# ============================================================================
# STEP 3: ENHANCED TFT MODEL WITH QUANTILE REGRESSION
# ============================================================================
print("\n🧠 STEP 3: BUILDING ENHANCED TFT MODEL...")

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network with LayerNorm"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
    
    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        gate = torch.sigmoid(self.gate(F.elu(self.fc1(x))))
        h = h * gate
        if self.skip is not None:
            x = self.skip(x)
        return self.layer_norm(x + h)

class VariableSelectionNetwork(nn.Module):
    """Enhanced variable selection with attention"""
    def __init__(self, input_dim, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])
        self.grn_combine = GatedResidualNetwork(
            num_features * hidden_dim, hidden_dim, hidden_dim, dropout
        )
        # Add feature importance weights
        self.feature_weights = nn.Parameter(torch.ones(num_features))
    
    def forward(self, x):
        batch_size = x.size(0)
        processed = [grn(x[:, i, :]) for i, grn in enumerate(self.grns)]
        processed = torch.stack(processed, dim=1)
        
        # Apply learnable feature weights
        weights = F.softmax(self.feature_weights, dim=0)
        processed = processed * weights.view(1, -1, 1)
        
        flattened = processed.view(batch_size, -1)
        combined = self.grn_combine(flattened)
        return combined, processed, weights

class EnhancedTFT(nn.Module):
    """Enhanced Temporal Fusion Transformer with quantile predictions"""
    def __init__(self, num_features=23, hidden_dim=256, num_heads=8, num_layers=3,
                 forecast_horizon=24, output_dim=13, dropout=0.15, use_quantiles=True):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.use_quantiles = use_quantiles
        
        # Variable selection
        self.vsn = VariableSelectionNetwork(1, num_features, hidden_dim, dropout)
        
        # Encoder LSTM (bidirectional for better context)
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0, 
            bidirectional=True
        )
        
        # Project bidirectional output
        self.encoder_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention (decoder attends to encoder)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Post-attention processing
        self.grn_post_attention = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # Output heads
        if use_quantiles:
            # Predict 3 quantiles (10%, 50%, 90%) for uncertainty
            self.fc_out = nn.ModuleList([
                nn.Linear(hidden_dim, 3) for _ in range(output_dim)
            ])
        else:
            self.fc_out = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(output_dim)
            ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_steps=168, forecast_steps=24):
        batch_size = x.size(0)
        
        # Variable selection for encoder
        encoder_outputs = []
        for t in range(encoder_steps):
            step_input = x[:, t, :].unsqueeze(-1)
            selected, _, _ = self.vsn(step_input)
            encoder_outputs.append(selected)
        
        encoder_outputs = torch.stack(encoder_outputs, dim=1)
        
        # Encode with bidirectional LSTM
        encoder_out, (h_n, c_n) = self.encoder_lstm(encoder_outputs)
        encoder_out = self.encoder_proj(encoder_out)
        
        # Convert bidirectional hidden states to unidirectional for decoder
        # h_n/c_n shape: (num_layers * 2, batch, hidden_dim)
        # We need to combine forward and backward directions
        num_layers = self.encoder_lstm.num_layers
        
        # Reshape to separate layers and directions
        h_n = h_n.view(num_layers, 2, batch_size, self.hidden_dim)
        c_n = c_n.view(num_layers, 2, batch_size, self.hidden_dim)
        
        # Combine forward and backward by summing
        h_n = (h_n[:, 0, :, :] + h_n[:, 1, :, :]) / 2  # (num_layers, batch, hidden_dim)
        c_n = (c_n[:, 0, :, :] + c_n[:, 1, :, :]) / 2  # (num_layers, batch, hidden_dim)
        
        # Decoder
        decoder_input = encoder_out[:, -1:, :].repeat(1, forecast_steps, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input, (h_n, c_n))
        
        # Self-attention on decoder
        attn_out, _ = self.self_attention(decoder_out, decoder_out, decoder_out)
        decoder_out = decoder_out + attn_out
        
        # Cross-attention (decoder to encoder)
        cross_attn_out, _ = self.cross_attention(decoder_out, encoder_out, encoder_out)
        decoder_out = decoder_out + cross_attn_out
        
        # Post-attention processing
        decoder_out = self.grn_post_attention(decoder_out.reshape(-1, self.hidden_dim))
        decoder_out = decoder_out.reshape(batch_size, forecast_steps, self.hidden_dim)
        
        # Predictions
        if self.use_quantiles:
            # Predict 3 quantiles per feature
            predictions = [fc(decoder_out) for fc in self.fc_out]
            predictions = torch.stack(predictions, dim=-1)  # (B, T, 3, Features)
            predictions = predictions.permute(0, 1, 3, 2)  # (B, T, Features, 3)
        else:
            predictions = [fc(decoder_out) for fc in self.fc_out]
            predictions = torch.cat(predictions, dim=-1)
        
        return predictions

# Create enhanced model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_input_features = len(data_splits['input_features'])
num_output_features = len(data_splits['target_features'])

print(f"  📊 Input features: {num_input_features}")
print(f"  🎯 Output features: {num_output_features}")

model = EnhancedTFT(
    num_features=num_input_features,
    hidden_dim=CONFIG['model']['hidden_dim'],
    num_heads=CONFIG['model']['num_heads'],
    num_layers=CONFIG['model']['num_layers'],
    forecast_horizon=CONFIG['data']['forecast_steps'],
    output_dim=num_output_features,
    dropout=CONFIG['model']['dropout'],
    use_quantiles=CONFIG['advanced']['use_quantile_loss']
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n✓ Model created!")
print(f"  📦 Total parameters: {total_params:,}")
print(f"  🎓 Trainable parameters: {trainable_params:,}")
print(f"  💾 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
print(f"  🖥️  Device: {device}")

# ============================================================================
# STEP 4: ADVANCED TRAINING WITH MIXED PRECISION & LR SCHEDULING
# ============================================================================
print("\n🚀 STEP 4: TRAINING MODEL WITH ADVANCED TECHNIQUES...")

class WeatherDataset(Dataset):
    """Dataset with optional augmentation"""
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        
        # Simple augmentation: add small Gaussian noise
        if self.augment and np.random.rand() < 0.3:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        return x, y

# Create dataloaders
print("  📦 Creating dataloaders...")
train_dataset = WeatherDataset(
    data_splits['train']['X'], 
    data_splits['train']['y'],
    augment=CONFIG['advanced']['augment_data']
)
val_dataset = WeatherDataset(data_splits['val']['X'], data_splits['val']['y'])
test_dataset = WeatherDataset(data_splits['test']['X'], data_splits['test']['y'])

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['training']['batch_size'], 
    shuffle=True,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['training']['batch_size'],
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"    ✓ Train batches: {len(train_loader)}")
print(f"    ✓ Val batches: {len(val_loader)}")

# Advanced loss function
def quantile_loss(y_pred, y_true, quantiles=[0.1, 0.5, 0.9]):
    """Pinball loss for quantile regression"""
    if len(y_pred.shape) == 4:  # (B, T, F, Q)
        losses = []
        for i, q in enumerate(quantiles):
            error = y_true - y_pred[:, :, :, i]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        return sum(losses) / len(losses)
    else:
        return F.mse_loss(y_pred, y_true)

def combined_loss(y_pred, y_true, use_quantile=True):
    """Combined MSE + MAE loss"""
    if use_quantile and len(y_pred.shape) == 4:
        return quantile_loss(y_pred, y_true)
    else:
        mse = F.mse_loss(y_pred, y_true)
        mae = F.l1_loss(y_pred, y_true)
        return 0.7 * mse + 0.3 * mae  # Weighted combination

# Optimizer with warmup
print("  ⚙️  Setting up optimizer and scheduler...")
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=CONFIG['training']['learning_rate'], 
    weight_decay=CONFIG['training']['weight_decay'],
    betas=(0.9, 0.999)
)

# Learning rate scheduler with warm restarts
if CONFIG['advanced']['use_lr_scheduler']:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart period each time
        eta_min=1e-7
    )
else:
    scheduler = None

# Mixed precision training
if CONFIG['training']['use_mixed_precision'] and torch.cuda.is_available():
    scaler = torch.cuda.amp.GradScaler()
    use_amp = True
    print("    ✓ Using mixed precision (FP16)")
else:
    scaler = None
    use_amp = False
    print("    ✓ Using full precision (FP32)")

# Training metrics
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses = [], []
learning_rates = []

os.makedirs(CONFIG['paths']['checkpoint_dir'], exist_ok=True)
os.makedirs(CONFIG['paths']['results_dir'], exist_ok=True)

print("\n🏃 Starting training loop...")
print("="*80)

for epoch in range(CONFIG['training']['epochs']):
    epoch_start = time.time()
    
    # ========== TRAINING ==========
    model.train()
    train_loss = 0
    train_mae = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['training']['epochs']}")
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                predictions = model(
                    X_batch, 
                    encoder_steps=CONFIG['data']['encoder_steps'], 
                    forecast_steps=CONFIG['data']['forecast_steps']
                )
                loss = combined_loss(predictions, y_batch, CONFIG['advanced']['use_quantile_loss'])
            
            # Backward with scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(
                X_batch, 
                encoder_steps=CONFIG['data']['encoder_steps'], 
                forecast_steps=CONFIG['data']['forecast_steps']
            )
            loss = combined_loss(predictions, y_batch, CONFIG['advanced']['use_quantile_loss'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['grad_clip'])
            optimizer.step()
        
        train_loss += loss.item()
        
        # Calculate MAE for monitoring
        if CONFIG['advanced']['use_quantile_loss']:
            pred_median = predictions[:, :, :, 1]  # Use median prediction
        else:
            pred_median = predictions
        train_mae += F.l1_loss(pred_median, y_batch).item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss /= len(train_loader)
    train_mae /= len(train_loader)
    train_losses.append(train_loss)
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0
    val_mae = 0
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    predictions = model(
                        X_batch, 
                        encoder_steps=CONFIG['data']['encoder_steps'],
                        forecast_steps=CONFIG['data']['forecast_steps']
                    )
                    loss = combined_loss(predictions, y_batch, CONFIG['advanced']['use_quantile_loss'])
            else:
                predictions = model(
                    X_batch, 
                    encoder_steps=CONFIG['data']['encoder_steps'],
                    forecast_steps=CONFIG['data']['forecast_steps']
                )
                loss = combined_loss(predictions, y_batch, CONFIG['advanced']['use_quantile_loss'])
            
            val_loss += loss.item()
            
            if CONFIG['advanced']['use_quantile_loss']:
                pred_median = predictions[:, :, :, 1]
            else:
                pred_median = predictions
            
            val_mae += F.l1_loss(pred_median, y_batch).item()
            
            val_predictions.append(pred_median.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_mae /= len(val_loader)
    val_losses.append(val_loss)
    
    # Learning rate step
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    if scheduler:
        scheduler.step()
    
    epoch_time = time.time() - epoch_start
    
    # Print metrics
    print(f"\n📊 Epoch {epoch+1}/{CONFIG['training']['epochs']} ({epoch_time:.1f}s)")
    print(f"   Train Loss: {train_loss:.6f} | Train MAE: {train_mae:.6f}")
    print(f"   Val Loss:   {val_loss:.6f} | Val MAE:   {val_mae:.6f}")
    print(f"   LR: {current_lr:.2e}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': CONFIG,
            'input_features': data_splits['input_features'],
            'target_features': data_splits['target_features'],
        }, CONFIG['paths']['best_model'])
        print(f"   ✅ New best model! (Val Loss: {val_loss:.6f})")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"   ⏳ Patience: {patience_counter}/{CONFIG['training']['early_stopping_patience']}")
    
    # Early stopping
    if patience_counter >= CONFIG['training']['early_stopping_patience']:
        print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
        break
    
    print("="*80)

print(f"\n🎉 Training completed!")
print(f"✓ Best validation loss: {best_val_loss:.6f}")
print(f"✓ Total epochs: {epoch+1}")
print(f"✓ Model saved to: {CONFIG['paths']['best_model']}")

# Save metadata
metadata = {
    'input_features': data_splits['input_features'],
    'target_features': data_splits['target_features'],
    'num_inputs': len(data_splits['input_features']),
    'num_outputs': len(data_splits['target_features']),
    'model_config': CONFIG['model'],
    'data_config': CONFIG['data'],
    'scalers': data_splits.get('scalers', {}),
    'best_val_loss': best_val_loss,
}

metadata_path = '/content/model_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Metadata saved to: {metadata_path}")

# ============================================================================
# STEP 5: COMPREHENSIVE EVALUATION
# ============================================================================
print("\n📈 STEP 5: EVALUATING MODEL PERFORMANCE...")

# Load best model
checkpoint = torch.load(CONFIG['paths']['best_model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate on test set
print("  🧪 Running test evaluation...")
test_loader = DataLoader(test_dataset, batch_size=CONFIG['training']['batch_size'])

test_predictions = []
test_targets = []

with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
        X_batch = X_batch.to(device)
        predictions = model(
            X_batch,
            encoder_steps=CONFIG['data']['encoder_steps'],
            forecast_steps=CONFIG['data']['forecast_steps']
        )
        
        if CONFIG['advanced']['use_quantile_loss']:
            predictions = predictions[:, :, :, 1]  # Use median
        
        test_predictions.append(predictions.cpu().numpy())
        test_targets.append(y_batch.numpy())

test_predictions = np.concatenate(test_predictions, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

# Calculate metrics per feature
print("\n📊 PERFORMANCE METRICS BY FEATURE:")
print("="*80)

target_features = data_splits['target_features']
metrics_summary = []

for i, feature in enumerate(target_features):
    y_true = test_targets[:, :, i].flatten()
    y_pred = test_predictions[:, :, i].flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    metrics_summary.append({
        'feature': feature,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    })
    
    print(f"{feature:15s} | MAE: {mae:8.4f} | RMSE: {rmse:8.4f} | R²: {r2:7.4f} | MAPE: {mape:6.2f}%")

print("="*80)

# Overall metrics
overall_mae = mean_absolute_error(test_targets.flatten(), test_predictions.flatten())
overall_rmse = np.sqrt(mean_squared_error(test_targets.flatten(), test_predictions.flatten()))
overall_r2 = r2_score(test_targets.flatten(), test_predictions.flatten())

print(f"\n🎯 OVERALL PERFORMANCE:")
print(f"   MAE:  {overall_mae:.6f}")
print(f"   RMSE: {overall_rmse:.6f}")
print(f"   R²:   {overall_r2:.6f}")

# ============================================================================
# STEP 6: VISUALIZATION
# ============================================================================
print("\n🎨 STEP 6: CREATING VISUALIZATIONS...")

# 1. Training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Progress', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(learning_rates, linewidth=2, color='orange')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.bar(range(len(target_features)), [m['R²'] for m in metrics_summary])
plt.xticks(range(len(target_features)), target_features, rotation=45, ha='right')
plt.ylabel('R² Score', fontsize=12)
plt.title('R² Score by Feature', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/content/training_metrics.png', dpi=200, bbox_inches='tight')
print("  ✓ Saved: training_metrics.png")

# 2. Prediction samples
plt.figure(figsize=(20, 10))

num_samples = min(4, len(target_features))
sample_idx = np.random.randint(0, len(test_predictions))

for i in range(num_samples):
    plt.subplot(2, 2, i+1)
    
    feature_idx = i % len(target_features)
    feature_name = target_features[feature_idx]
    
    true_values = test_targets[sample_idx, :, feature_idx]
    pred_values = test_predictions[sample_idx, :, feature_idx]
    
    hours = np.arange(CONFIG['data']['forecast_steps'])
    
    plt.plot(hours, true_values, 'o-', label='True', linewidth=2, markersize=4)
    plt.plot(hours, pred_values, 's-', label='Predicted', linewidth=2, markersize=4)
    plt.fill_between(hours, true_values, pred_values, alpha=0.2)
    
    plt.xlabel('Forecast Hour', fontsize=11)
    plt.ylabel(feature_name.replace('_', ' ').title(), fontsize=11)
    plt.title(f'{feature_name.title()} - 24h Forecast', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/prediction_samples.png', dpi=200, bbox_inches='tight')
print("  ✓ Saved: prediction_samples.png")

# 3. Error distribution
plt.figure(figsize=(15, 8))

for i in range(min(6, len(target_features))):
    plt.subplot(2, 3, i+1)
    
    feature_idx = i
    feature_name = target_features[feature_idx]
    
    errors = (test_targets[:, :, feature_idx] - test_predictions[:, :, feature_idx]).flatten()
    
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.title(f'{feature_name.title()} Error Distribution', fontsize=11, fontweight='bold')
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.text(0.02, 0.98, f'μ={mean_error:.3f}\nσ={std_error:.3f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/content/error_distribution.png', dpi=200, bbox_inches='tight')
print("  ✓ Saved: error_distribution.png")

# 4. Scatter plots (Predicted vs True)
plt.figure(figsize=(15, 10))

for i in range(min(6, len(target_features))):
    plt.subplot(2, 3, i+1)
    
    feature_idx = i
    feature_name = target_features[feature_idx]
    
    y_true = test_targets[:, :, feature_idx].flatten()
    y_pred = test_predictions[:, :, feature_idx].flatten()
    
    # Sample for faster plotting
    sample_size = min(5000, len(y_true))
    indices = np.random.choice(len(y_true), sample_size, replace=False)
    
    plt.scatter(y_true[indices], y_pred[indices], alpha=0.3, s=1)
    
    # Perfect prediction line
    min_val, max_val = y_true.min(), y_true.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    plt.xlabel('True Value', fontsize=10)
    plt.ylabel('Predicted Value', fontsize=10)
    plt.title(f'{feature_name.title()}', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/scatter_plots.png', dpi=200, bbox_inches='tight')
print("  ✓ Saved: scatter_plots.png")

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv('/content/performance_metrics.csv', index=False)
print("  ✓ Saved: performance_metrics.csv")

print("\n" + "="*80)
print("🎉 ALL DONE! TRAINING & EVALUATION COMPLETE!")
print("="*80)
print(f"\n⏱️  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n📥 DOWNLOAD THESE FILES FROM COLAB:")
print(f"   1. {CONFIG['paths']['best_model']}")
print(f"   2. {metadata_path}")
print(f"   3. /content/training_metrics.png")
print(f"   4. /content/prediction_samples.png")
print(f"   5. /content/error_distribution.png")
print(f"   6. /content/scatter_plots.png")
print(f"   7. /content/performance_metrics.csv")
print(f"\n📂 Place model files in: AI-Backend/checkpoints/")
print(f"📊 Review visualizations to understand model performance")
print(f"\n🚀 Next: Use the model for predictions in your Flask API!")
print("="*80)