# 🚀 QUICK START - Google Colab Commands

## ⚡ Super Quick (Copy-Paste These Commands)

### 1. Check GPU
```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 2. Install Dependencies
```bash
!pip install torch pandas numpy scikit-learn openpyxl tqdm matplotlib seaborn
```

### 3. Upload Files
```python
from google.colab import files
uploaded = files.upload()  # Upload your 4 Excel files
```

### 4. Run Training
```bash
!python colab_simple_train.py
```

### 5. Download Model
```python
from google.colab import files
files.download('/content/best_weather_model.pth')
```

---

## 📋 Files Needed

Upload these to Colab:
- ✅ `Location information.xlsx`
- ✅ `Weather data.xlsx`
- ✅ `Astronomical.xlsx`
- ✅ `Air quality information.xlsx`
- ✅ `colab_simple_train.py`

---

## 🎯 Complete Command Sequence

```python
# 1. Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")

# 2. Install packages
!pip install -q torch pandas numpy scikit-learn openpyxl tqdm matplotlib seaborn

# 3. Upload Excel files
from google.colab import files
print("Upload your 4 Excel files:")
uploaded = files.upload()

# 4. Upload training script
print("Upload colab_simple_train.py:")
uploaded = files.upload()

# 5. Run training
!python colab_simple_train.py

# 6. Download trained model
files.download('/content/best_weather_model.pth')
files.download('/content/training_curve.png')
```

---

## 🔧 Modify Training Parameters

Edit `colab_simple_train.py` and change these lines:

```python
CONFIG = {
    'model': {
        'hidden_dim': 128,      # ← Increase for larger model (256, 512)
        'num_heads': 4,         # ← More attention heads (4, 8)
        'num_layers': 2,        # ← Deeper model (2, 3, 4)
    },
    'training': {
        'epochs': 100,          # ← More epochs
        'batch_size': 32,       # ← Larger batches if GPU allows (64, 128)
        'learning_rate': 0.001, # ← Adjust learning rate
    },
}
```

---

## 📊 Expected Output

```
================================================================================
🌤️  MAUSAM-VAANI - WEATHER PREDICTION MODEL TRAINING
================================================================================
Start Time: 2025-11-24 15:10:00
Device: GPU
GPU Name: Tesla T4
================================================================================

📊 STEP 1: MERGING EXCEL FILES...
  ✓ Loading Weather data.xlsx...
  ✓ Loading Location information.xlsx...
  ✓ Loading Astronomical.xlsx...
  ✓ Loading Air quality information.xlsx...
  Final shape: (50000, 35)
  ✓ Saved to: /content/merged_weather_dataset.csv
✓ Dataset loaded: 50,000 rows, 35 columns

🔧 STEP 2: DATA PREPROCESSING...
  Cleaning data...
  Engineering features...
  Creating sequences...
  Splitting data...
✓ Train: 35,000 samples
✓ Val: 7,500 samples
✓ Test: 7,500 samples

🧠 STEP 3: DEFINING TFT MODEL...
✓ Model created with 1,234,567 parameters
✓ Device: cuda

🚀 STEP 4: TRAINING MODEL...
Epoch 1/100: 100%|██████████| 1094/1094 [00:45<00:00]
Epoch 1: Train Loss = 0.4523, Val Loss = 0.3891
  ✓ New best model saved! (Val Loss: 0.3891)

Epoch 2/100: 100%|██████████| 1094/1094 [00:43<00:00]
Epoch 2: Train Loss = 0.3214, Val Loss = 0.2987
  ✓ New best model saved! (Val Loss: 0.2987)

...

✓ Training completed!
✓ Best validation loss: 0.0234
✓ Model saved to: /content/best_weather_model.pth

📈 STEP 5: CREATING VISUALIZATIONS...
✓ Training curve saved to: /content/training_curve.png

================================================================================
🎉 TRAINING COMPLETE!
================================================================================
End Time: 2025-11-24 15:52:15

Download your trained model:
  /content/best_weather_model.pth
================================================================================
```

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| No GPU | Runtime → Change runtime type → T4 GPU |
| Out of memory | Reduce `batch_size` to 16 or 8 |
| File not found | Check `!ls` to see uploaded files |
| Import error | Run `!pip install <package>` |
| Slow training | Ensure GPU is enabled |

---

## ⏱️ Training Time Estimates

| Dataset Size | Time (GPU) | Time (CPU) |
|--------------|------------|------------|
| 10K samples  | ~15 min    | ~2 hours   |
| 50K samples  | ~45 min    | ~8 hours   |
| 100K samples | ~90 min    | ~16 hours  |

---

## 💾 Save to Google Drive

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy model to Drive
!cp /content/best_weather_model.pth /content/drive/MyDrive/
!cp /content/training_curve.png /content/drive/MyDrive/
!cp -r /content/checkpoints /content/drive/MyDrive/
```

---

## ✅ Success Checklist

- [ ] GPU enabled (T4)
- [ ] All 4 Excel files uploaded
- [ ] Training script uploaded
- [ ] Dependencies installed
- [ ] Training completed without errors
- [ ] Model file downloaded
- [ ] Training curve looks good (loss decreasing)

---

**That's it! You're ready to train! 🚀**
