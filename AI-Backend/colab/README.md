# 🌤️ Mausam-Vaani - Google Colab Training

**Train your Weather TFT model on FREE Google Colab GPU in 3 simple steps!**

---

## 🚀 Quick Start

### 1. Upload Files to Colab
- `colab_simple_train.py`
- `Location information.xlsx`
- `Weather data.xlsx`
- `Astronomical.xlsx`
- `Air quality information.xlsx`

### 2. Enable GPU
Runtime → Change runtime type → **T4 GPU** → Save

### 3. Run Training
```bash
!pip install torch pandas numpy scikit-learn openpyxl tqdm matplotlib seaborn
!python colab_simple_train.py
```

**Done!** Download your trained model in ~45 minutes.

---

## 📁 Files in This Directory

### 🎯 Core Training
| File | Purpose | Size |
|------|---------|------|
| **colab_simple_train.py** | Complete standalone training script | 500+ lines |
| **colab_config.py** | Python configuration (optional) | 150 lines |

### 📚 Documentation
| File | Purpose |
|------|---------|
| **QUICK_START.md** | Copy-paste commands for immediate use |
| **COLAB_SETUP_GUIDE.md** | Comprehensive step-by-step guide |
| **FILES_TO_EXCLUDE.md** | What NOT to upload to Colab |
| **README.md** | This file |

---

## 🎯 What You Need

**Minimum (5 files)**:
1. `colab_simple_train.py` ← Training script
2. `Location information.xlsx` ← Data
3. `Weather data.xlsx` ← Data
4. `Astronomical.xlsx` ← Data
5. `Air quality information.xlsx` ← Data

**Optional**:
- `colab_config.py` - If you want to modify settings separately
- Documentation files - For reference

---

## 📖 Documentation Guide

### For First-Time Users
👉 **Start here**: `QUICK_START.md`
- Copy-paste commands
- 5-minute setup
- No explanations, just commands

### For Detailed Setup
👉 **Read this**: `COLAB_SETUP_GUIDE.md`
- Complete walkthrough
- Troubleshooting
- Performance tips
- Customization options

### For Understanding What to Upload
👉 **Check this**: `FILES_TO_EXCLUDE.md`
- What NOT to upload
- Why certain files aren't needed
- Minimal package guide

---

## ⚡ Super Quick Commands

```python
# 1. Check GPU
import torch
print(f"GPU: {torch.cuda.is_available()}")

# 2. Install
!pip install -q torch pandas numpy scikit-learn openpyxl tqdm matplotlib seaborn

# 3. Upload files
from google.colab import files
uploaded = files.upload()  # Upload 5 files

# 4. Train
!python colab_simple_train.py

# 5. Download model
files.download('/content/best_weather_model.pth')
```

---

## 🎨 What the Script Does

**colab_simple_train.py** is a complete, self-contained script that:

1. ✅ **Merges Data** - Combines 4 Excel files into one dataset
2. ✅ **Preprocesses** - Cleans, engineers features, creates sequences
3. ✅ **Builds Model** - Complete TFT architecture (~1.2M parameters)
4. ✅ **Trains** - With progress bars, validation, early stopping
5. ✅ **Saves** - Best model, checkpoints, training curve
6. ✅ **Visualizes** - Training progress plot

**No external files needed!** Everything is in one script.

---

## 📊 Expected Results

### Training Output
```
================================================================================
🌤️  MAUSAM-VAANI - WEATHER PREDICTION MODEL TRAINING
================================================================================
Device: GPU
GPU Name: Tesla T4

📊 STEP 1: MERGING EXCEL FILES...
✓ Dataset loaded: 50,000 rows, 35 columns

🔧 STEP 2: DATA PREPROCESSING...
✓ Train: 35,000 samples
✓ Val: 7,500 samples

🧠 STEP 3: DEFINING TFT MODEL...
✓ Model created with 1,234,567 parameters

🚀 STEP 4: TRAINING MODEL...
Epoch 1/100: Train Loss = 0.4523, Val Loss = 0.3891
  ✓ New best model saved!
...

✓ Training completed!
✓ Best validation loss: 0.0234

📈 STEP 5: CREATING VISUALIZATIONS...
✓ Training curve saved

🎉 TRAINING COMPLETE!
Download your trained model: /content/best_weather_model.pth
================================================================================
```

### Generated Files
- `best_weather_model.pth` - Your trained model (download this!)
- `merged_weather_dataset.csv` - Combined dataset
- `training_curve.png` - Loss visualization
- `checkpoints/` - Model checkpoints

---

## ⏱️ Training Time

| Dataset Size | Time (T4 GPU) |
|--------------|---------------|
| 10K samples  | ~15 minutes   |
| 50K samples  | ~45 minutes   |
| 100K samples | ~90 minutes   |

---

## 🔧 Customization

Edit these in `colab_simple_train.py`:

```python
CONFIG = {
    'model': {
        'hidden_dim': 128,      # Model size (128, 256, 512)
        'num_heads': 4,         # Attention heads (4, 8)
        'num_layers': 2,        # Depth (2, 3, 4)
    },
    'training': {
        'epochs': 100,          # Max epochs
        'batch_size': 32,       # Batch size (16, 32, 64, 128)
        'learning_rate': 0.001, # Learning rate
    },
    'data': {
        'encoder_steps': 168,   # History (hours)
        'forecast_steps': 24,   # Prediction (hours)
    },
}
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| No GPU | Runtime → Change runtime type → T4 GPU |
| Out of memory | Reduce `batch_size` to 16 |
| File not found | Check `!ls /content` |
| Slow training | Ensure GPU is enabled |

---

## ✅ Success Checklist

- [ ] Google Colab account
- [ ] GPU enabled (T4)
- [ ] 5 files uploaded
- [ ] Dependencies installed
- [ ] Training completed
- [ ] Model downloaded
- [ ] Ready for deployment!

---

## 🎯 Next Steps

After training:

1. **Download model** - `best_weather_model.pth`
2. **Review training curve** - Check for learning
3. **Deploy** - Use with Flask API (see main README)
4. **Integrate** - Connect to frontend
5. **Fine-tune** - Adjust hyperparameters if needed

---

## 💡 Why This Setup?

### Original Project
- 13+ Python files
- Multiple directories
- YAML configs
- Complex dependencies
- 30+ minute setup

### Colab Version
- ✅ 1 Python file
- ✅ Simple structure
- ✅ Python config
- ✅ Minimal dependencies
- ✅ 5-minute setup

**Result**: 6x faster setup, same powerful model!

---

## 📞 Need Help?

1. Check `QUICK_START.md` for commands
2. Read `COLAB_SETUP_GUIDE.md` for details
3. Review `FILES_TO_EXCLUDE.md` for cleanup
4. Check troubleshooting sections

---

## 🎉 You're Ready!

Everything you need is in this directory. Just:
1. Upload 5 files to Colab
2. Run one command
3. Download trained model

**Happy Training!** 🌤️🌧️⛈️

---

**Estimated Total Time**: < 60 minutes (5 min setup + 45 min training)

**Cost**: FREE (Google Colab T4 GPU)

**Result**: Production-ready weather prediction model!
