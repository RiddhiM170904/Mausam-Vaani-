"""
Simple Configuration for Google Colab Training
No YAML dependency - just pure Python!
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_CONFIG = {
    'hidden_dim': 128,          # Hidden dimension size
    'num_heads': 4,             # Number of attention heads
    'num_layers': 2,            # Number of transformer layers
    'forecast_horizon': 24,     # Predict next 24 hours
    'dropout': 0.1,             # Dropout rate
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_CONFIG = {
    # Data paths (modify these in Colab)
    'data_dir': '/content/drive/MyDrive/Mausam-Vaani/data',  # Google Drive path
    'merged_csv': 'merged_weather_dataset.csv',
    
    # Or use local Colab storage
    # 'data_dir': '/content/data',
    
    # Sequence parameters
    'encoder_steps': 168,       # 1 week of hourly data (7 * 24)
    'forecast_steps': 24,       # Predict next 24 hours
    
    # Data split ratios
    'train_ratio': 0.7,         # 70% for training
    'val_ratio': 0.15,          # 15% for validation
    'test_ratio': 0.15,         # 15% for testing
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    'epochs': 100,              # Maximum number of epochs
    'batch_size': 32,           # Batch size (increase if you have more GPU memory)
    'learning_rate': 0.001,     # Initial learning rate
    'weight_decay': 0.0001,     # L2 regularization
    'grad_clip': 1.0,           # Gradient clipping value
    'early_stopping_patience': 15,  # Stop if no improvement for 15 epochs
    'save_every': 5,            # Save checkpoint every N epochs
    'num_workers': 2,           # DataLoader workers (use 2 for Colab)
}

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================
PATHS_CONFIG = {
    'checkpoint_dir': '/content/checkpoints',
    'log_dir': '/content/logs',
    'results_dir': '/content/results',
    'best_model_path': '/content/best_weather_model.pth',
}

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
FEATURE_CONFIG = {
    # Input features (9 features)
    'input_features': [
        'temperature', 'humidity', 'wind_speed', 'rainfall', 
        'pressure', 'cloud_cover', 'latitude', 'longitude', 'hour'
    ],
    
    # Target features to predict (6 features)
    'target_features': [
        'temperature', 'humidity', 'wind_speed', 
        'rainfall', 'pressure', 'cloud_cover'
    ],
    
    # Number of features
    'num_input_features': 9,
    'num_target_features': 6,
}

# ============================================================================
# EXCEL FILES CONFIGURATION (for data merging)
# ============================================================================
EXCEL_FILES = {
    'location': 'Location information.xlsx',
    'weather': 'Weather data.xlsx',
    'astronomical': 'Astronomical.xlsx',
    'air_quality': 'Air quality information.xlsx',
}

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
GPU_CONFIG = {
    'use_gpu': True,            # Use GPU if available
    'gpu_id': 0,                # GPU device ID
    'mixed_precision': True,    # Use mixed precision training (faster on modern GPUs)
}

# ============================================================================
# HELPER FUNCTION
# ============================================================================
def get_full_config():
    """Get complete configuration dictionary"""
    return {
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
        'training': TRAINING_CONFIG,
        'paths': PATHS_CONFIG,
        'features': FEATURE_CONFIG,
        'excel_files': EXCEL_FILES,
        'gpu': GPU_CONFIG,
    }

def print_config():
    """Print configuration in a readable format"""
    config = get_full_config()
    print("="*80)
    print("MAUSAM-VAANI TRAINING CONFIGURATION")
    print("="*80)
    for section, params in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in params.items():
            print(f"  {key:25s}: {value}")
    print("="*80)

if __name__ == "__main__":
    print_config()
