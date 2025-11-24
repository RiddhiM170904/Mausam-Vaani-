"""
🌤️ MAUSAM-VAANI - SIMPLE COLAB TRAINING SCRIPT
Complete standalone script for training Weather TFT model on Google Colab GPU

This script includes EVERYTHING:
- Data merging from 4 Excel files
- Data preprocessing
- TFT model definition
- Training loop
- Evaluation
- Model saving

Just run: python colab_simple_train.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Configuration (inline - no external files needed!)
CONFIG = {
    'model': {
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'forecast_horizon': 24,
        'dropout': 0.1,
    },
    'data': {
        'encoder_steps': 168,
        'forecast_steps': 24,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
    },
    'training': {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'grad_clip': 1.0,
        'early_stopping_patience': 15,
    },
    'paths': {
        'data_dir': '/content',
        'checkpoint_dir': '/content/checkpoints',
        'best_model': '/content/best_weather_model.pth',
    }
}

print("="*80)
print("🌤️  MAUSAM-VAANI - WEATHER PREDICTION MODEL TRAINING")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("="*80)

# ============================================================================
# STEP 1: DATA MERGING
# ============================================================================
print("\n📊 STEP 1: MERGING EXCEL FILES...")

def merge_excel_files(data_dir='/content'):
    """Merge 4 Excel files into one dataset"""
    files = {
        'location': 'Location information.xlsx',
        'weather': 'Weather data.xlsx',
        'astronomical': 'Astronomical.xlsx',
        'air_quality': 'Air quality information.xlsx',
    }
    
    print("Loading Excel files...")
    dfs = {}
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✓ Loading {filename}...")
            dfs[name] = pd.read_excel(filepath, engine='openpyxl')
        else:
            print(f"  ✗ {filename} not found!")
    
    if 'weather' not in dfs:
        print("ERROR: Weather data.xlsx is required!")
        return None
    
    # Start with weather data
    merged = dfs['weather'].copy()
    print(f"  Base dataset: {merged.shape}")
    
    # Merge other datasets
    for name, df in dfs.items():
        if name != 'weather':
            common_cols = list(set(merged.columns) & set(df.columns))
            if common_cols:
                print(f"  Merging {name} on: {common_cols}")
                merged = pd.merge(merged, df, on=common_cols, how='left', suffixes=('', f'_{name}'))
    
    print(f"  Final shape: {merged.shape}")
    
    # Save merged dataset
    output_path = os.path.join(data_dir, 'merged_weather_dataset.csv')
    merged.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    
    return merged

# Check if merged file already exists
merged_path = os.path.join(CONFIG['paths']['data_dir'], 'merged_weather_dataset.csv')
if os.path.exists(merged_path):
    print(f"  ✓ Using existing merged dataset: {merged_path}")
    df = pd.read_csv(merged_path)
else:
    df = merge_excel_files(CONFIG['paths']['data_dir'])
    if df is None:
        print("ERROR: Could not merge data files!")
        sys.exit(1)

print(f"✓ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n🔧 STEP 2: DATA PREPROCESSING...")

class WeatherDataProcessor:
    """Process weather data for TFT model"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def process(self, encoder_steps=168, forecast_steps=24):
        """Complete preprocessing pipeline"""
        print("  Cleaning data...")
        self._clean_data()
        
        print("  Engineering features...")
        self._engineer_features()
        
        print("  Creating sequences...")
        sequences = self._create_sequences(encoder_steps, forecast_steps)
        
        print("  Splitting data...")
        splits = self._split_data(sequences)
        
        return splits
    
    def _clean_data(self):
        """Clean and handle missing values"""
        # Convert datetime
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.df = self.df.sort_values(col)
                    break
                except:
                    continue
        
        # Fill numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Fill remaining with median
        for col in numeric_cols:
            if self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].median(), inplace=True)
    
    def _engineer_features(self):
        """Create time-based features"""
        # Find datetime column
        datetime_col = None
        for col in self.df.columns:
            if self.df[col].dtype == 'datetime64[ns]':
                datetime_col = col
                break
        
        if datetime_col:
            self.df['hour'] = self.df[datetime_col].dt.hour
            self.df['day_of_week'] = self.df[datetime_col].dt.dayofweek
            self.df['month'] = self.df[datetime_col].dt.month
    
    def _create_sequences(self, encoder_steps, forecast_steps):
        """Create time series sequences"""
        # Define features (adjust based on your data)
        feature_cols = []
        target_cols = []  # Features we want to predict
        
        # Target features (weather variables we want to predict)
        for col in ['temperature', 'humidity', 'wind_speed', 'rainfall', 'pressure', 'cloud_cover']:
            if col in self.df.columns:
                feature_cols.append(col)
                target_cols.append(col)
        
        # Add location and time features (input only, not predicted)
        for col in ['latitude', 'longitude', 'hour']:
            if col in self.df.columns and col not in feature_cols:
                feature_cols.append(col)
        
        print(f"  Found {len(feature_cols)} input features: {feature_cols}")
        print(f"  Predicting {len(target_cols)} target features: {target_cols}")
        
        if len(feature_cols) < 3:
            print(f"  ERROR: Not enough features found!")
            print(f"  Available columns: {list(self.df.columns)}")
            raise ValueError("Insufficient features for training")
        
        data = self.df[feature_cols].values
        
        # Create sequences
        total_steps = encoder_steps + forecast_steps
        X, y = [], []
        
        # Number of target features to predict
        num_targets = len(target_cols)
        
        for i in range(len(data) - total_steps):
            X.append(data[i:i+total_steps])
            # Only predict the target features (first num_targets columns)
            y.append(data[i+encoder_steps:i+total_steps, :num_targets])
        
        return {
            'X': np.array(X),
            'y': np.array(y),
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'num_targets': num_targets
        }
    
    def _split_data(self, sequences):
        """Split into train/val/test"""
        X, y = sequences['X'], sequences['y']
        n = len(X)
        
        train_end = int(n * CONFIG['data']['train_ratio'])
        val_end = int(n * (CONFIG['data']['train_ratio'] + CONFIG['data']['val_ratio']))
        
        return {
            'train': {'X': X[:train_end], 'y': y[:train_end]},
            'val': {'X': X[train_end:val_end], 'y': y[train_end:val_end]},
            'test': {'X': X[val_end:], 'y': y[val_end:]},
            'feature_cols': sequences['feature_cols'],
            'target_cols': sequences['target_cols'],
            'num_targets': sequences['num_targets']
        }

processor = WeatherDataProcessor(df)
data_splits = processor.process(
    encoder_steps=CONFIG['data']['encoder_steps'],
    forecast_steps=CONFIG['data']['forecast_steps']
)

print(f"✓ Train: {len(data_splits['train']['X']):,} samples")
print(f"✓ Val: {len(data_splits['val']['X']):,} samples")
print(f"✓ Test: {len(data_splits['test']['X']):,} samples")
print(f"✓ Input features: {len(data_splits['feature_cols'])}")
print(f"✓ Output features: {data_splits['num_targets']}")

# ============================================================================
# STEP 3: MODEL DEFINITION
# ============================================================================
print("\n🧠 STEP 3: DEFINING TFT MODEL...")

# Copy the TFT model classes from tft_model.py
class GatedResidualNetwork(nn.Module):
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
    
    def forward(self, x):
        batch_size = x.size(0)
        processed = [grn(x[:, i, :]) for i, grn in enumerate(self.grns)]
        processed = torch.stack(processed, dim=1)
        flattened = processed.view(batch_size, -1)
        combined = self.grn_combine(flattened)
        return combined, processed

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_features=9, hidden_dim=128, num_heads=4, num_layers=2, 
                 forecast_horizon=24, output_dim=6, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        
        self.vsn = VariableSelectionNetwork(1, num_features, hidden_dim, dropout)
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, 
                                     dropout=dropout if num_layers > 1 else 0)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                                     dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.grn_post_attention = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.fc_out = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(output_dim)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_steps=168, forecast_steps=24):
        batch_size = x.size(0)
        encoder_outputs = []
        
        for t in range(encoder_steps):
            step_input = x[:, t, :].unsqueeze(-1)
            selected, _ = self.vsn(step_input)
            encoder_outputs.append(selected)
        
        encoder_outputs = torch.stack(encoder_outputs, dim=1)
        encoder_out, (h_n, c_n) = self.encoder_lstm(encoder_outputs)
        decoder_input = encoder_out[:, -1:, :].repeat(1, forecast_steps, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input, (h_n, c_n))
        attn_out, _ = self.attention(decoder_out, encoder_out, encoder_out)
        attn_out = self.grn_post_attention(attn_out.reshape(-1, self.hidden_dim))
        attn_out = attn_out.reshape(batch_size, forecast_steps, self.hidden_dim)
        
        predictions = [fc(attn_out) for fc in self.fc_out]
        predictions = torch.cat(predictions, dim=-1)
        
        return predictions

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use actual number of target features from data
num_output_features = data_splits['num_targets']
print(f"  Model will predict {num_output_features} features: {data_splits['target_cols']}")

model = TemporalFusionTransformer(
    num_features=len(data_splits['feature_cols']),
    hidden_dim=CONFIG['model']['hidden_dim'],
    num_heads=CONFIG['model']['num_heads'],
    num_layers=CONFIG['model']['num_layers'],
    forecast_horizon=CONFIG['data']['forecast_steps'],
    output_dim=num_output_features,  # Dynamic based on actual data
    dropout=CONFIG['model']['dropout']
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created with {total_params:,} parameters")
print(f"✓ Device: {device}")

# ============================================================================
# STEP 4: TRAINING
# ============================================================================
print("\n🚀 STEP 4: TRAINING MODEL...")

class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataloaders
train_dataset = WeatherDataset(data_splits['train']['X'], data_splits['train']['y'])
val_dataset = WeatherDataset(data_splits['val']['X'], data_splits['val']['y'])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['training']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['training']['batch_size'])

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['training']['learning_rate'], 
                            weight_decay=CONFIG['training']['weight_decay'])

# Training loop
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses = [], []

os.makedirs(CONFIG['paths']['checkpoint_dir'], exist_ok=True)

for epoch in range(CONFIG['training']['epochs']):
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['training']['epochs']}"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch, encoder_steps=CONFIG['data']['encoder_steps'], 
                          forecast_steps=CONFIG['data']['forecast_steps'])
        loss = criterion(predictions, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['grad_clip'])
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch, encoder_steps=CONFIG['data']['encoder_steps'],
                              forecast_steps=CONFIG['data']['forecast_steps'])
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CONFIG['paths']['best_model'])
        print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= CONFIG['training']['early_stopping_patience']:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\n✓ Training completed!")
print(f"✓ Best validation loss: {best_val_loss:.4f}")
print(f"✓ Model saved to: {CONFIG['paths']['best_model']}")

# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================
print("\n📈 STEP 5: CREATING VISUALIZATIONS...")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('/content/training_curve.png', dpi=150, bbox_inches='tight')
print("✓ Training curve saved to: /content/training_curve.png")

print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nDownload your trained model:")
print(f"  {CONFIG['paths']['best_model']}")
print("="*80)
