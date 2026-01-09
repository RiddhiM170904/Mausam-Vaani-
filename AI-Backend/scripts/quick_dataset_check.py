"""
Quick Dataset Analysis - Check your Excel files structure
Run this first to understand your data before combining

NOTE: Requires openpyxl. Install with: pip install openpyxl
"""

import pandas as pd
import os

data_dir = r"C:\personal dg\github_repo\Mausam-Vaani-\AI-Backend\data"

print("=" * 80)
print("QUICK DATASET CHECK")
print("=" * 80)

try:
    import openpyxl
    print("\n✓ openpyxl is installed")
except ImportError:
    print("\n❌ openpyxl not found!")
    print("Please install it with: pip install openpyxl")
    exit(1)

print("\n" + "=" * 80)
print("ANALYZING YOUR DATASETS")
print("=" * 80)

files = {
    "Weather Data": "Weather data.xlsx",
    "Location Info": "Location information.xlsx",
    "Astronomical": "Astronomical.xlsx",
    "Air Quality": "Air quality information.xlsx"
}

for name, filename in files.items():
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"\n📊 {name}")
        print("-" * 80)
        df = pd.read_excel(filepath)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        print(f"\nFirst 2 rows:")
        print(df.head(2).to_string())
        print(f"\nMissing values: {df.isnull().sum().sum()} total")
        if df.isnull().sum().sum() > 0:
            missing = df.isnull().sum()
            print(f"  {missing[missing > 0].to_dict()}")
    else:
        print(f"\n❌ {name}: File not found at {filepath}")

print("\n" + "=" * 80)
print("✓ Analysis Complete!")
print("=" * 80)
print("\nNext: Run the combination script:")
print("python scripts/combine_and_clean_datasets.py")
