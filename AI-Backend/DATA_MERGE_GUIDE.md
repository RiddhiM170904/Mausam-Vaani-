# Data Merging Pipeline - Step-by-Step Guide

This guide will help you merge the 4 Kaggle Excel files into a unified training dataset for the Mausam-Vaani weather prediction model.

## 📋 Prerequisites

You have 4 Excel files in the `AI-Backend/data/` directory:
- ✓ `Location information.xlsx` (1.04 MB)
- ✓ `Weather data.xlsx` (2.47 MB)
- ✓ `Astronomical.xlsx` (985 KB)
- ✓ `Air quality information.xlsx` (1.23 MB)

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd AI-Backend
pip install -r scripts/requirements_data.txt
```

This installs:
- `pandas` - Data manipulation
- `openpyxl` - Excel file reading
- `numpy` - Numerical operations

### Step 2: Run the Merge Script

```bash
python scripts/merge_datasets.py
```

### Step 3: Check the Output

The script will create:
- `data/merged_weather_dataset.csv` - Full merged dataset
- `data/sample_merged_weather_dataset.csv` - Sample (1000 rows) for quick inspection
- `data/merged_dataset_info.txt` - Column information and statistics
- `data_merge.log` - Detailed execution log

## 📊 What the Script Does

### 1. **Examines All Files**
   - Loads each Excel file
   - Displays columns, data types, and missing values
   - Shows memory usage and row counts

### 2. **Identifies Merge Keys**
   - Automatically detects datetime columns (date, time, datetime, timestamp)
   - Finds location identifiers (city, station, lat, lon, location)
   - Identifies common columns across files

### 3. **Intelligent Merging**
   - Starts with Weather data as the base (most comprehensive)
   - Merges Location, Astronomical, and Air Quality data
   - Uses LEFT JOIN to preserve all weather records
   - Handles duplicate column names with suffixes

### 4. **Data Preparation for TFT**
   - Sorts by datetime
   - Handles missing values:
     - Numeric: Forward fill → Backward fill → Median
     - Categorical: Mode or 'Unknown'
   - Ensures no null values remain

### 5. **Saves Output**
   - CSV format (efficient for large datasets)
   - Includes sample file for quick review
   - Generates metadata file with column info

## 🔍 Verify the Output

### Check the merged dataset:
```bash
# View first few rows
python -c "import pandas as pd; df = pd.read_csv('data/merged_weather_dataset.csv'); print(df.head())"

# Check shape
python -c "import pandas as pd; df = pd.read_csv('data/merged_weather_dataset.csv'); print(f'Shape: {df.shape}')"

# View column names
python -c "import pandas as pd; df = pd.read_csv('data/merged_weather_dataset.csv'); print(df.columns.tolist())"
```

### Review the info file:
```bash
# Windows
type data\merged_dataset_info.txt

# Linux/Mac
cat data/merged_dataset_info.txt
```

## 🎯 Next Steps

After successful merging:

1. **Review the merged dataset** - Check `merged_dataset_info.txt` for column details

2. **Update model configuration** - Modify `config/model_config.yaml` to use the new dataset:
   ```yaml
   data:
     train_data_path: "data/merged_weather_dataset.csv"
   ```

3. **Run data preprocessing** - Prepare data for TFT model:
   ```bash
   python models/data_preprocessing.py
   ```

4. **Train the model** - Start training with the merged dataset:
   ```bash
   python models/train.py
   ```

## 🐛 Troubleshooting

### Issue: "No module named 'openpyxl'"
**Solution:** Install dependencies
```bash
pip install -r scripts/requirements_data.txt
```

### Issue: "File not found"
**Solution:** Make sure you're in the `AI-Backend` directory
```bash
cd AI-Backend
python scripts/merge_datasets.py
```

### Issue: Memory error with large files
**Solution:** The script processes files efficiently, but if you still face issues:
- Close other applications
- Use the sample file for testing first
- Consider processing in chunks (modify script if needed)

### Issue: Merge produces unexpected results
**Solution:** Check the log file
```bash
type data_merge.log  # Windows
cat data_merge.log   # Linux/Mac
```

## 📝 Script Output Example

```
================================================================================
MAUSAM-VAANI DATA MERGING PIPELINE
================================================================================

Loading Location information.xlsx...
✓ Loaded Location information.xlsx: 50000 rows, 8 columns
  Columns: ['date', 'location', 'latitude', 'longitude', ...]

Loading Weather data.xlsx...
✓ Loaded Weather data.xlsx: 50000 rows, 15 columns
  Columns: ['date', 'location', 'temperature', 'humidity', ...]

[... more output ...]

✓ Saved to: data/merged_weather_dataset.csv
  Final shape: (50000, 35)
  File size: 15.23 MB

================================================================================
PIPELINE COMPLETED
================================================================================
Duration: 45.67 seconds
Output: data/merged_weather_dataset.csv
```

## 💡 Tips

- **Large files**: The script handles large files efficiently with pandas
- **Check logs**: Always review `data_merge.log` for detailed information
- **Sample first**: Use the sample file to quickly verify the merge worked correctly
- **Backup**: Keep original Excel files as backup
- **Version control**: Don't commit large CSV files to git (add to `.gitignore`)

## 🎓 Understanding the Merged Dataset

The merged dataset will contain:
- **Time series data**: Weather observations over time
- **Location data**: Geographic information for each observation
- **Astronomical data**: Sun/moon positions, day length, etc.
- **Air quality data**: Pollution levels, AQI, particulate matter
- **All features**: Ready for TFT model training

This comprehensive dataset enables the model to learn complex patterns and make accurate hyperlocal weather predictions!
