# 🚀 Google Colab Setup Guide - Mausam-Vaani Weather Model Training

## 📌 Overview

This guide will help you train the **Temporal Fusion Transformer (TFT)** weather prediction model on **Google Colab's free GPU**. Everything is simplified for direct execution!

---

## 🎯 Quick Start (3 Simple Steps)

### Step 1: Upload Files to Google Colab

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Upload the notebook**: `colab_train_weather_model.ipynb`
3. **Upload your 4 Excel files** to Colab or Google Drive:
   - `Location information.xlsx`
   - `Weather data.xlsx`
   - `Astronomical.xlsx`
   - `Air quality information.xlsx`

### Step 2: Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Select **T4 GPU** (free tier)
3. Click **Save**

### Step 3: Run All Cells

1. Click **Runtime** → **Run all**
2. Wait for training to complete (~30-60 minutes)
3. Download your trained model!

---

## 📂 File Structure for Colab

```
/content/
├── Location information.xlsx
├── Weather data.xlsx
├── Astronomical.xlsx
├── Air quality information.xlsx
├── colab_config.py (optional - for standalone script)
├── colab_simple_train.py (optional - standalone training)
└── colab_train_weather_model.ipynb (main notebook)
```

---

## 📓 Using the Jupyter Notebook

### Option A: All-in-One Notebook (Recommended)

Upload `colab_train_weather_model.ipynb` and run all cells sequentially.

**The notebook includes:**
1. ✅ GPU setup and verification
2. ✅ Install dependencies
3. ✅ Merge 4 Excel files into one dataset
4. ✅ Data preprocessing and feature engineering
5. ✅ TFT model definition
6. ✅ Training with progress bars
7. ✅ Validation and checkpointing
8. ✅ Model evaluation and visualization
9. ✅ Download trained model

---

## 🐍 Using Standalone Python Script

### Option B: Simple Python Script

If you prefer running a single Python file:

```bash
# 1. Upload files to Colab
# Upload: colab_simple_train.py, colab_config.py, and 4 Excel files

# 2. Install dependencies
!pip install torch pandas numpy scikit-learn openpyxl tqdm matplotlib

# 3. Run training
!python colab_simple_train.py
```

---

## 💻 Step-by-Step Commands

### 1. Check GPU Availability

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Expected Output:**
```
GPU Available: True
GPU Name: Tesla T4
```

### 2. Install Required Packages

```bash
!pip install torch pandas numpy scikit-learn openpyxl tqdm matplotlib seaborn tensorboard
```

### 3. Upload Excel Files

```python
from google.colab import files

# Upload your 4 Excel files
uploaded = files.upload()
```

Or mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Set data directory
DATA_DIR = '/content/drive/MyDrive/Mausam-Vaani/data'
```

### 4. Merge Excel Files

```python
# This is included in the notebook
# It will create: merged_weather_dataset.csv
```

### 5. Train the Model

```python
# Run training cells in the notebook
# Or execute: !python colab_simple_train.py
```

### 6. Download Trained Model

```python
from google.colab import files

# Download the best model
files.download('/content/best_weather_model.pth')

# Download checkpoints
files.download('/content/checkpoints/checkpoint_best.pth')
```

---

## 🔧 Configuration

### Modify Training Parameters

Edit these in the notebook or `colab_config.py`:

```python
# Model size
MODEL_CONFIG = {
    'hidden_dim': 128,      # Increase for more capacity (128, 256, 512)
    'num_heads': 4,         # Attention heads (4, 8)
    'num_layers': 2,        # Transformer layers (2, 3, 4)
    'forecast_horizon': 24, # Hours to predict (24, 48, 72)
    'dropout': 0.1,
}

# Training settings
TRAINING_CONFIG = {
    'epochs': 100,          # Max epochs
    'batch_size': 32,       # Increase if you have more GPU memory (32, 64, 128)
    'learning_rate': 0.001, # Learning rate
}
```

---

## 📊 Expected Training Output

```
================================================================================
MAUSAM-VAANI WEATHER MODEL TRAINING
================================================================================
GPU: Tesla T4
Device: cuda:0

Loading data...
✓ Merged dataset: 50,000 samples
✓ Train: 35,000 | Val: 7,500 | Test: 7,500

Model Parameters: 1,234,567
Trainable Parameters: 1,234,567

Epoch 1/100
Train Loss: 0.4523 | Val Loss: 0.3891 | Time: 45s
✓ New best model saved!

Epoch 2/100
Train Loss: 0.3214 | Val Loss: 0.2987 | Time: 43s
✓ New best model saved!

...

Training completed in 42m 15s
Best validation loss: 0.0234 (Epoch 67)
```

---

## 🎨 Visualization

The notebook includes:
- **Training curves**: Loss over epochs
- **Prediction plots**: Actual vs predicted weather
- **Feature importance**: Which features matter most
- **Error analysis**: Where the model struggles

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size

```python
TRAINING_CONFIG['batch_size'] = 16  # or 8
```

### Issue: "No module named 'openpyxl'"

**Solution:** Install missing package

```bash
!pip install openpyxl
```

### Issue: "GPU not detected"

**Solution:** Enable GPU runtime

1. Runtime → Change runtime type
2. Select T4 GPU
3. Save and reconnect

### Issue: "Excel file not found"

**Solution:** Check file paths

```python
import os
print(os.listdir('/content'))  # List all files
```

### Issue: Training is too slow

**Solutions:**
- Reduce `encoder_steps` from 168 to 72 (3 days instead of 1 week)
- Reduce `batch_size`
- Use mixed precision training (already enabled)

---

## 📦 What You'll Get

After training completes, you'll have:

1. **Trained model file**: `best_weather_model.pth` (~5-10 MB)
2. **Checkpoints**: Saved every 5 epochs
3. **Training logs**: TensorBoard logs for visualization
4. **Evaluation metrics**: MAE, RMSE, R² scores
5. **Prediction plots**: Visual comparison of predictions

---

## 🚀 Advanced: Using Google Drive

### Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Save Everything to Drive

```python
import shutil

# Create directory
!mkdir -p /content/drive/MyDrive/Mausam-Vaani/models

# Copy trained model
shutil.copy('/content/best_weather_model.pth', 
            '/content/drive/MyDrive/Mausam-Vaani/models/')

# Copy checkpoints
shutil.copytree('/content/checkpoints', 
                '/content/drive/MyDrive/Mausam-Vaani/checkpoints')
```

---

## ⚡ Performance Tips

### 1. Use Mixed Precision Training (Already Enabled)
Speeds up training by 2-3x on modern GPUs

### 2. Increase Batch Size
If you have GPU memory available:
```python
TRAINING_CONFIG['batch_size'] = 64  # or 128
```

### 3. Use Gradient Accumulation
For larger effective batch sizes:
```python
# Add to training loop
accumulation_steps = 4
```

### 4. Enable TensorBoard
Monitor training in real-time:
```python
%load_ext tensorboard
%tensorboard --logdir /content/logs
```

---

## 📈 Next Steps After Training

1. **Evaluate on test set**: Check model performance
2. **Download model**: Save to your local machine
3. **Deploy model**: Use with Flask API (see main README)
4. **Fine-tune**: Adjust hyperparameters and retrain
5. **Experiment**: Try different forecast horizons

---

## 🎓 Understanding the Model

### Input Features (9)
- Temperature, Humidity, Wind Speed, Rainfall
- Pressure, Cloud Cover
- Latitude, Longitude, Hour of day

### Output Predictions (6)
- Temperature, Humidity, Wind Speed, Rainfall
- Pressure, Cloud Cover

### Architecture
- **Encoder**: 168 hours (1 week) of historical data
- **Decoder**: 24 hours of future predictions
- **Attention**: Multi-head self-attention mechanism
- **Components**: Variable selection, LSTM, GRN

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the notebook cell outputs for error messages
3. Ensure all 4 Excel files are uploaded correctly
4. Verify GPU is enabled and detected

---

## ✅ Checklist

Before running:
- [ ] Google Colab account created
- [ ] GPU runtime enabled (T4)
- [ ] 4 Excel files uploaded
- [ ] Notebook uploaded
- [ ] All cells run successfully

After training:
- [ ] Model trained without errors
- [ ] Validation loss decreased
- [ ] Best model downloaded
- [ ] Checkpoints saved
- [ ] Ready for deployment!

---

## 🎉 Success!

Once training completes, you'll have a fully trained weather prediction model ready to deploy with the Mausam-Vaani Flask API!

**Estimated Training Time:**
- Small dataset (10K samples): ~15 minutes
- Medium dataset (50K samples): ~45 minutes
- Large dataset (100K+ samples): ~90 minutes

**GPU Usage:**
- Memory: ~4-6 GB (out of 15 GB on T4)
- Utilization: ~80-90% during training

Happy training! 🌤️🌧️⛈️
