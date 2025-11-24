# Files to Exclude from Google Colab

When uploading to Google Colab, you **DO NOT** need these files/directories:

## ❌ Unnecessary for Training

### API & Serving
- `api/` - Flask API (only for deployment)
  - `api/__init__.py`
  - `api/app.py`
  - `api/routes.py`
  - `api/gemini_integration.py`
  - `api/utils.py`

### Model Serving
- `models/model_serving.py` - For deployment only

### Evaluation Scripts
- `models/evaluate.py` - Integrated into training script

### Validation Scripts
- `scripts/validate_data.py` - Optional validation

### Environment & Config
- `venv/` - Virtual environment (Colab has its own)
- `.gitignore` - Git-specific
- `README.md` - Documentation (optional)

### Config Files (if using standalone script)
- `config/model_config.yaml` - Replaced by `colab_config.py`

## ✅ Files Needed for Colab

### Essential Data
- `Location information.xlsx` ✓
- `Weather data.xlsx` ✓
- `Astronomical.xlsx` ✓
- `Air quality information.xlsx` ✓

### Training Scripts (Choose One)
**Option A: Standalone Script (Recommended)**
- `colab/colab_simple_train.py` ✓

**Option B: Jupyter Notebook**
- `colab/colab_train_weather_model.ipynb` ✓

### Optional
- `colab/colab_config.py` - Configuration (only if modifying)
- `colab/COLAB_SETUP_GUIDE.md` - Instructions
- `colab/QUICK_START.md` - Quick reference

## 📦 Minimal Upload Package

For the simplest setup, upload only:
1. `colab_simple_train.py`
2. `Location information.xlsx`
3. `Weather data.xlsx`
4. `Astronomical.xlsx`
5. `Air quality information.xlsx`

**Total: 5 files** (~6 MB)

## 🗂️ Simplified Directory Structure

```
/content/
├── colab_simple_train.py          ← Training script
├── Location information.xlsx       ← Data
├── Weather data.xlsx               ← Data
├── Astronomical.xlsx               ← Data
└── Air quality information.xlsx    ← Data
```

After training, you'll have:
```
/content/
├── ... (above files)
├── merged_weather_dataset.csv      ← Generated
├── best_weather_model.pth          ← Your trained model!
├── training_curve.png              ← Visualization
└── checkpoints/                    ← Model checkpoints
    ├── checkpoint_epoch_5.pth
    ├── checkpoint_epoch_10.pth
    └── ...
```

## 💡 Why Exclude These?

- **API files**: Only needed for serving predictions, not training
- **Virtual environment**: Colab provides its own Python environment
- **Config YAML**: Replaced with simple Python config
- **Git files**: Not needed in Colab
- **Evaluation scripts**: Integrated into main training script

## 🎯 Result

By excluding unnecessary files, you:
- ✅ Reduce upload time
- ✅ Simplify setup
- ✅ Focus only on training
- ✅ Avoid confusion

Everything you need is in the `colab/` directory!
