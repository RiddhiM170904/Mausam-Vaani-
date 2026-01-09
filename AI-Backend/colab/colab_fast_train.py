"""
🌤️ MAUSAM-VAANI - FAST 1-HOUR TRAINING SCRIPT
Optimized lightweight script for quick training on Google Colab GPU

⚡ OPTIMIZATIONS FOR SPEED:
- Smaller model architecture (128 hidden dim, 2 layers)
- Shorter sequences (72h history → 12h forecast)
- Simplified preprocessing (core features only)
- Fewer epochs (40 max with early stopping)
- Streamlined architecture (no bidirectional LSTM)
- Mixed precision training (FP16)

📊 UPLOAD: combined_weather_dataset.csv to /content/ folder
🚀 RUN: Runtime > Run all
⏱️ EXPECTED TIME: ~45-60 minutes on T4 GPU
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Core imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

# FAST CONFIGURATION - OPTIMIZED FOR 1 HOUR TRAINING
CONFIG = {
    'model': {
        'hidden_dim': 128,  # Reduced from 256
        'num_heads': 4,     # Reduced from 8
        'num_layers': 2,    # Reduced from 3
        'forecast_horizon': 12,  # Reduced from 24
        'dropout': 0.1,
    },
    'data': {
        'csv_file': 'combined_weather_dataset.csv',
        'encoder_steps': 72,   # 3 days history (reduced from 7 days)
        'forecast_steps': 12,  # 12 hours forecast (reduced from 24)
        'train_ratio': 0.75,   # More training data
        'val_ratio': 0.15,
        'test_ratio': 0.10,
    },
    'training': {
        'epochs': 40,          # Reduced from 100
        'batch_size': 128,     # Increased for faster epochs
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'grad_clip': 1.0,
        'early_stopping_patience': 10,  # Reduced from 20
        'use_mixed_precision': True,
    },
    'paths': {
        'data_dir': '/content',
        'checkpoint_dir': '/content/checkpoints',
        'best_model': '/content/fast_weather_model.pth',
        'results_dir': '/content/results',
    }
}

print("="*80)
print("⚡ MAUSAM-VAANI - FAST 1-HOUR TRAINING")
print("="*80)
print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🖥️  Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"⏱️  Expected Time: ~45-60 minutes")
print("="*80)

# ============================================================================
# STEP 1: FAST DATA LOADING
# ============================================================================
print("\n📊 STEP 1: LOADING DATASET...")

csv_path = os.path.join(CONFIG['paths']['data_dir'], CONFIG['data']['csv_file'])

if not os.path.exists(csv_path):
    print(f"❌ ERROR: {CONFIG['data']['csv_file']} not found!")
    print(f"\n📝 INSTRUCTIONS:")
    print(f"1. Run combine_and_clean_datasets.py locally")
    print(f"2. Upload combined_weather_dataset.csv to Colab")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# STEP 2: FAST PREPROCESSING
# ============================================================================
print("\n🔧 STEP 2: FAST PREPROCESSING...")

class FastWeatherProcessor:
    """Lightweight preprocessing - core features only"""
    
    def __init__(self, df, config):
        self.df = df.copy()
        self.config = config
        self.scalers = {}
        
    def process(self):
        """Quick preprocessing pipeline"""
        print("  🧹 Cleaning data...")
        self._clean_data()
        
        print("  🎨 Creating basic features...")
        self._create_features()
        
        print("  📏 Scaling...")
        self._scale_features()
        
        print("  📦 Creating sequences...")
        sequences = self._create_sequences()
        
        print("  ✂️  Splitting...")
        splits = self._split_data(sequences)
        
        return splits
    
    def _clean_data(self):
        """Fast data cleaning"""
        # Parse timestamps
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(
                self.df['timestamp'], 
                errors='coerce',
                format='mixed'
            )
            
            # Remove invalid timestamps
            invalid_mask = self.df['timestamp'].isna()
            if invalid_mask.any():
                print(f"    ⚠️  Removed {invalid_mask.sum()} invalid timestamps")
                self.df = self.df[~invalid_mask].reset_index(drop=True)
        
        # Sort by time
        self.df = self.df.sort_values(['city', 'timestamp']).reset_index(drop=True)
        
        # Fill missing values (simple forward fill)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df.groupby('city')[numeric_cols].transform(
            lambda x: x.ffill().bfill()
        )
        
        # Fill any remaining with median
        for col in numeric_cols:
            if self.df[col].isna().any():
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Drop remaining NaN
        self.df = self.df.dropna()
        print(f"    ✓ Cleaned {len(self.df):,} records")
    
    def _create_features(self):
        """Create only essential time features"""
        # Basic time features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # Simple cyclical encoding (hour only)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        
        print(f"    ✓ Created {len(self.df.columns)} features")
    
    def _scale_features(self):
        """Scale key features"""
        # Core features to scale
        scale_features = [
            'temperature', 'humidity', 'wind_speed', 'rainfall', 
            'pressure', 'cloud_cover', 'aqi', 'pm25', 'pm10',
            'co', 'no2', 'o3', 'so2', 'latitude', 'longitude'
        ]
        
        for col in scale_features:
            if col in self.df.columns:
                scaler = RobustScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                self.scalers[col] = scaler
        
        print(f"    ✓ Scaled {len(self.scalers)} features")
    
    def _create_sequences(self):
        """Create simple sequences - core features only"""
        # Input features
        input_features = [
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'pressure', 'cloud_cover', 'aqi', 'pm25', 'pm10',
            'co', 'no2', 'o3', 'so2', 'latitude', 'longitude',
            'hour_sin', 'hour_cos', 'day_of_week', 'month', 'is_weekend'
        ]
        
        # Filter available features
        input_features = [f for f in input_features if f in self.df.columns]
        
        # Target features (what to predict)
        target_features = [
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'pressure', 'cloud_cover', 'aqi', 'pm25', 'pm10',
            'co', 'no2', 'o3', 'so2'
        ]
        target_features = [f for f in target_features if f in self.df.columns]
        
        print(f"    📊 Input: {len(input_features)} features")
        print(f"    🎯 Output: {len(target_features)} features")
        
        # Extract arrays
        data = self.df[input_features].values
        targets = self.df[target_features].values
        
        # Create sequences
        encoder_steps = self.config['data']['encoder_steps']
        forecast_steps = self.config['data']['forecast_steps']
        total_steps = encoder_steps + forecast_steps
        
        X, y = [], []
        for i in range(len(data) - total_steps):
            X.append(data[i:i+encoder_steps])
            y.append(targets[i+encoder_steps:i+total_steps])
        
        return {
            'X': np.array(X, dtype=np.float32),
            'y': np.array(y, dtype=np.float32),
            'input_features': input_features,
            'target_features': target_features,
        }
    
    def _split_data(self, sequences):
        """Split into train/val/test"""
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
            'scalers': self.scalers,
        }

# Process data
processor = FastWeatherProcessor(df, CONFIG)
data_splits = processor.process()

print(f"\n✓ Train: {len(data_splits['train']['X']):,} samples")
print(f"✓ Val: {len(data_splits['val']['X']):,} samples")
print(f"✓ Test: {len(data_splits['test']['X']):,} samples")

# ============================================================================
# STEP 3: LIGHTWEIGHT TFT MODEL
# ============================================================================
print("\n🧠 STEP 3: BUILDING LIGHTWEIGHT MODEL...")

class SimpleTFT(nn.Module):
    """Fast, simplified Temporal Fusion Transformer"""
    
    def __init__(self, num_features, hidden_dim, num_heads, num_layers,
                 forecast_horizon, output_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # Encoder LSTM (unidirectional for speed)
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Output layers
        self.output_gate = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, encoder_steps=72, forecast_steps=12):
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Encode historical data
        encoder_input = x[:, :encoder_steps, :]
        encoded, (h, c) = self.encoder_lstm(encoder_input)
        
        # Self-attention on encoder output
        attn_out, _ = self.self_attention(encoded, encoded, encoded)
        encoded = self.layer_norm(encoded + attn_out)
        
        # Decode future
        decoder_input = torch.zeros(batch_size, forecast_steps, self.hidden_dim).to(x.device)
        decoded, _ = self.decoder_lstm(decoder_input, (h, c))
        
        # Gated output
        gate = torch.sigmoid(self.output_gate(decoded))
        output = decoded * gate
        output = self.dropout(output)
        
        # Final projection
        predictions = self.output_proj(output)
        
        return predictions

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_input_features = len(data_splits['input_features'])
num_output_features = len(data_splits['target_features'])

model = SimpleTFT(
    num_features=num_input_features,
    hidden_dim=CONFIG['model']['hidden_dim'],
    num_heads=CONFIG['model']['num_heads'],
    num_layers=CONFIG['model']['num_layers'],
    forecast_horizon=CONFIG['data']['forecast_steps'],
    output_dim=num_output_features,
    dropout=CONFIG['model']['dropout']
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n✓ Model created!")
print(f"  📦 Parameters: {total_params:,}")
print(f"  💾 Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

# ============================================================================
# STEP 4: FAST TRAINING
# ============================================================================
print("\n🚀 STEP 4: FAST TRAINING...")

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

print(f"  ✓ Train batches: {len(train_loader)}")
print(f"  ✓ Val batches: {len(val_loader)}")

# Optimizer & Loss
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['training']['learning_rate'],
    weight_decay=CONFIG['training']['weight_decay']
)

criterion = nn.MSELoss()

# Mixed precision
if CONFIG['training']['use_mixed_precision'] and torch.cuda.is_available():
    scaler = torch.cuda.amp.GradScaler()
    use_amp = True
    print("  ✓ Using mixed precision (FP16)")
else:
    scaler = None
    use_amp = False

# Training loop
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses = [], []

os.makedirs(CONFIG['paths']['checkpoint_dir'], exist_ok=True)

print("\n🏃 Starting training...")
print("="*80)

training_start = time.time()

for epoch in range(CONFIG['training']['epochs']):
    epoch_start = time.time()
    
    # ========== TRAINING ==========
    model.train()
    train_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['training']['epochs']}")
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                predictions = model(
                    X_batch,
                    encoder_steps=CONFIG['data']['encoder_steps'],
                    forecast_steps=CONFIG['data']['forecast_steps']
                )
                loss = criterion(predictions, y_batch)
            
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
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['grad_clip'])
            optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0
    
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
                    loss = criterion(predictions, y_batch)
            else:
                predictions = model(
                    X_batch,
                    encoder_steps=CONFIG['data']['encoder_steps'],
                    forecast_steps=CONFIG['data']['forecast_steps']
                )
                loss = criterion(predictions, y_batch)
            
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    epoch_time = time.time() - epoch_start
    total_time = time.time() - training_start
    
    # Print metrics
    print(f"\n📊 Epoch {epoch+1}/{CONFIG['training']['epochs']} ({epoch_time:.1f}s | Total: {total_time/60:.1f}min)")
    print(f"   Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': CONFIG,
        }, CONFIG['paths']['best_model'])
        
        print(f"   ✅ Best model saved! (Val Loss: {val_loss:.6f})")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= CONFIG['training']['early_stopping_patience']:
        print(f"\n⏹️  Early stopping at epoch {epoch+1}")
        break
    
    print("="*80)

total_training_time = time.time() - training_start

print(f"\n🎉 Training completed!")
print(f"⏱️  Total time: {total_training_time/60:.1f} minutes")
print(f"✓ Best validation loss: {best_val_loss:.6f}")
print(f"✓ Model saved to: {CONFIG['paths']['best_model']}")

# Save metadata
metadata = {
    'input_features': data_splits['input_features'],
    'target_features': data_splits['target_features'],
    'num_inputs': num_input_features,
    'num_outputs': num_output_features,
    'model_config': CONFIG['model'],
    'data_config': CONFIG['data'],
    'scalers': data_splits['scalers'],
    'best_val_loss': best_val_loss,
    'training_time_minutes': total_training_time / 60,
}

metadata_path = '/content/fast_model_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Metadata saved to: {metadata_path}")

# ============================================================================
# STEP 5: EVALUATION
# ============================================================================
print("\n📈 STEP 5: EVALUATING MODEL...")

# Load best model
checkpoint = torch.load(CONFIG['paths']['best_model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test evaluation
test_dataset = WeatherDataset(data_splits['test']['X'], data_splits['test']['y'])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['training']['batch_size'])

test_predictions = []
test_targets = []

with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
        X_batch = X_batch.to(device)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                predictions = model(
                    X_batch,
                    encoder_steps=CONFIG['data']['encoder_steps'],
                    forecast_steps=CONFIG['data']['forecast_steps']
                )
        else:
            predictions = model(
                X_batch,
                encoder_steps=CONFIG['data']['encoder_steps'],
                forecast_steps=CONFIG['data']['forecast_steps']
            )
        
        test_predictions.append(predictions.cpu().numpy())
        test_targets.append(y_batch.numpy())

test_predictions = np.concatenate(test_predictions, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

# Calculate metrics
print("\n📊 PERFORMANCE METRICS:")
print("="*80)

target_features = data_splits['target_features']
metrics_summary = []

for i, feature in enumerate(target_features):
    y_true = test_targets[:, :, i].flatten()
    y_pred = test_predictions[:, :, i].flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics_summary.append({
        'feature': feature,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    })
    
    print(f"{feature:15s} | MAE: {mae:8.4f} | RMSE: {rmse:8.4f} | R²: {r2:7.4f}")

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

# Training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Fast Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(len(target_features)), [m['R²'] for m in metrics_summary])
plt.xticks(range(len(target_features)), target_features, rotation=45, ha='right')
plt.ylabel('R² Score')
plt.title('R² Score by Feature')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/content/fast_training_results.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: fast_training_results.png")

# Prediction samples
plt.figure(figsize=(15, 8))

sample_idx = np.random.randint(0, len(test_predictions))

for i in range(min(4, len(target_features))):
    plt.subplot(2, 2, i+1)
    
    feature_name = target_features[i]
    true_values = test_targets[sample_idx, :, i]
    pred_values = test_predictions[sample_idx, :, i]
    
    hours = np.arange(CONFIG['data']['forecast_steps'])
    
    plt.plot(hours, true_values, 'o-', label='True', linewidth=2)
    plt.plot(hours, pred_values, 's-', label='Predicted', linewidth=2)
    plt.fill_between(hours, true_values, pred_values, alpha=0.2)
    
    plt.xlabel('Forecast Hour')
    plt.ylabel(feature_name.replace('_', ' ').title())
    plt.title(f'{feature_name.title()} - 12h Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/fast_predictions.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: fast_predictions.png")

# Save metrics
metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv('/content/fast_performance_metrics.csv', index=False)
print("  ✓ Saved: fast_performance_metrics.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎉 FAST TRAINING COMPLETE!")
print("="*80)
print(f"\n⏱️  Training Time: {total_training_time/60:.1f} minutes")
print(f"📊 Best Val Loss: {best_val_loss:.6f}")
print(f"🎯 Test R² Score: {overall_r2:.6f}")
print(f"\n📥 DOWNLOAD THESE FILES:")
print(f"   1. {CONFIG['paths']['best_model']}")
print(f"   2. {metadata_path}")
print(f"   3. /content/fast_training_results.png")
print(f"   4. /content/fast_predictions.png")
print(f"   5. /content/fast_performance_metrics.csv")
print(f"\n💡 MODEL OPTIMIZATIONS:")
print(f"   • Hidden dim: 128 (vs 256 in full model)")
print(f"   • Layers: 2 (vs 3 in full model)")
print(f"   • Attention heads: 4 (vs 8 in full model)")
print(f"   • History: 72h (vs 168h in full model)")
print(f"   • Forecast: 12h (vs 24h in full model)")
print(f"   • No bidirectional LSTM (faster)")
print(f"   • Simplified features (faster preprocessing)")
print("="*80)
