# 📋 Colab Training - Command Cheat Sheet

## 🚀 Complete Setup (Copy All)

```python
# ============================================================================
# STEP 1: CHECK GPU
# ============================================================================
import torch
print(f"✓ GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================================
# STEP 2: INSTALL DEPENDENCIES
# ============================================================================
!pip install -q torch pandas numpy scikit-learn openpyxl tqdm matplotlib seaborn

# ============================================================================
# STEP 3: UPLOAD FILES
# ============================================================================
from google.colab import files
print("📤 Upload your 5 files:")
print("  1. colab_simple_train.py")
print("  2. Location information.xlsx")
print("  3. Weather data.xlsx")
print("  4. Astronomical.xlsx")
print("  5. Air quality information.xlsx")
uploaded = files.upload()

# ============================================================================
# STEP 4: VERIFY FILES
# ============================================================================
import os
print("\n📁 Uploaded files:")
for file in os.listdir('/content'):
    if file.endswith(('.py', '.xlsx')):
        size = os.path.getsize(f'/content/{file}') / 1024**2
        print(f"  ✓ {file} ({size:.1f} MB)")

# ============================================================================
# STEP 5: RUN TRAINING
# ============================================================================
!python colab_simple_train.py

# ============================================================================
# STEP 6: DOWNLOAD RESULTS
# ============================================================================
print("\n📥 Downloading trained model...")
files.download('/content/best_weather_model.pth')
files.download('/content/training_curve.png')

print("\n✅ DONE! Your model is ready for deployment!")
```

---

## 🎯 Individual Commands

### Check GPU
```python
import torch
print(torch.cuda.is_available())
```

### Install Packages
```bash
!pip install torch pandas numpy scikit-learn openpyxl tqdm matplotlib seaborn
```

### Upload Files
```python
from google.colab import files
files.upload()
```

### List Files
```bash
!ls -lh /content/*.xlsx
!ls -lh /content/*.py
```

### Run Training
```bash
!python colab_simple_train.py
```

### Download Model
```python
from google.colab import files
files.download('/content/best_weather_model.pth')
```

### Save to Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/best_weather_model.pth /content/drive/MyDrive/
```

---

## 🔧 Modify Training Settings

### Quick Edit (In Colab)
```python
# Open the script
!nano colab_simple_train.py

# Or edit directly in Colab
# Click on colab_simple_train.py in Files panel
# Modify CONFIG dictionary
```

### Common Modifications
```python
# Larger model
'hidden_dim': 256  # was 128

# More epochs
'epochs': 150  # was 100

# Bigger batches (if GPU allows)
'batch_size': 64  # was 32

# Longer forecast
'forecast_steps': 48  # was 24 (predict 2 days instead of 1)
```

---

## 📊 Monitor Training

### View Progress
Training shows progress bars automatically with tqdm.

### Check GPU Usage
```bash
!nvidia-smi
```

### View Training Curve
```python
from IPython.display import Image, display
display(Image('/content/training_curve.png'))
```

---

## 🐛 Debug Commands

### Check Python Version
```bash
!python --version
```

### Check Installed Packages
```bash
!pip list | grep -E "torch|pandas|numpy"
```

### View File Contents
```bash
!head -20 colab_simple_train.py
```

### Check Disk Space
```bash
!df -h /content
```

### View Logs
```bash
!tail -50 /content/data_merge.log
```

---

## 💾 Backup Commands

### Zip Everything
```bash
!zip -r mausam_vaani_results.zip /content/*.pth /content/*.png /content/checkpoints
```

### Download Zip
```python
files.download('/content/mausam_vaani_results.zip')
```

### Copy to Drive
```bash
!cp -r /content/checkpoints /content/drive/MyDrive/
!cp /content/*.pth /content/drive/MyDrive/
!cp /content/*.png /content/drive/MyDrive/
```

---

## ⚡ Performance Commands

### Enable Mixed Precision (Already in script)
```python
# Already enabled in colab_simple_train.py
# Uses automatic mixed precision for faster training
```

### Check GPU Memory
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")
```

### Clear GPU Cache
```python
import torch
torch.cuda.empty_cache()
```

---

## 📈 Post-Training Analysis

### Load and Test Model
```python
import torch

# Load model
model = torch.load('/content/best_weather_model.pth')
print("✓ Model loaded successfully!")

# Check model size
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

### View Training Curve
```python
import matplotlib.pyplot as plt
from IPython.display import Image
Image('/content/training_curve.png')
```

---

## 🎯 Quick Troubleshooting

```bash
# GPU not detected
# → Runtime → Change runtime type → T4 GPU → Save

# Out of memory
# → Edit CONFIG['training']['batch_size'] = 16

# File not found
!ls /content  # Check uploaded files

# Package missing
!pip install <package_name>

# Training stuck
# → Check GPU is enabled
# → Reduce batch_size
# → Check data files are correct
```

---

**Save this as a reference!** 📌
