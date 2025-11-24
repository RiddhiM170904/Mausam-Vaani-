"""
Data Merging Script for Mausam-Vaani Weather Prediction Model
This script merges 4 Excel files from Kaggle into a unified training dataset:
1. Location information.xlsx
2. Weather data.xlsx
3. Astronomical.xlsx
4. Air quality information.xlsx

The merged dataset will be optimized for Temporal Fusion Transformer (TFT) model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_merge.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DatasetMerger:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.location_file = self.data_dir / "Location information.xlsx"
        self.weather_file = self.data_dir / "Weather data.xlsx"
        self.astronomical_file = self.data_dir / "Astronomical.xlsx"
        self.air_quality_file = self.data_dir / "Air quality information.xlsx"
        
    def load_excel_file(self, filepath, sheet_name=0):
        """Load Excel file with error handling"""
        try:
            logger.info(f"Loading {filepath.name}...")
            df = pd.read_excel(filepath, sheet_name=sheet_name, engine='openpyxl')
            logger.info(f"✓ Loaded {filepath.name}: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"  Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"✗ Error loading {filepath.name}: {str(e)}")
            return None
    
    def examine_all_files(self):
        """Examine structure of all files"""
        logger.info("\n" + "="*80)
        logger.info("EXAMINING ALL DATASET FILES")
        logger.info("="*80 + "\n")
        
        files_info = {}
        
        # Load all files
        location_df = self.load_excel_file(self.location_file)
        weather_df = self.load_excel_file(self.weather_file)
        astronomical_df = self.load_excel_file(self.astronomical_file)
        air_quality_df = self.load_excel_file(self.air_quality_file)
        
        # Store info
        files_info['location'] = location_df
        files_info['weather'] = weather_df
        files_info['astronomical'] = astronomical_df
        files_info['air_quality'] = air_quality_df
        
        # Display detailed info for each
        for name, df in files_info.items():
            if df is not None:
                logger.info(f"\n{name.upper()} Dataset Info:")
                logger.info(f"  Shape: {df.shape}")
                logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                logger.info(f"  Data types:\n{df.dtypes.value_counts()}")
                logger.info(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
                
        return files_info
    
    def identify_merge_keys(self, files_info):
        """Identify common columns for merging"""
        logger.info("\n" + "="*80)
        logger.info("IDENTIFYING MERGE KEYS")
        logger.info("="*80 + "\n")
        
        all_columns = {}
        for name, df in files_info.items():
            if df is not None:
                all_columns[name] = set(df.columns)
        
        # Find common columns
        common_cols = set.intersection(*all_columns.values()) if all_columns else set()
        logger.info(f"Common columns across all files: {common_cols}")
        
        # Check for datetime/location columns
        datetime_patterns = ['date', 'time', 'datetime', 'timestamp', 'dt']
        location_patterns = ['location', 'city', 'station', 'lat', 'lon', 'latitude', 'longitude']
        
        for name, cols in all_columns.items():
            logger.info(f"\n{name.upper()} potential merge keys:")
            for col in cols:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in datetime_patterns):
                    logger.info(f"  - {col} (datetime)")
                elif any(pattern in col_lower for pattern in location_patterns):
                    logger.info(f"  - {col} (location)")
        
        return common_cols
    
    def merge_datasets(self, files_info):
        """Merge all datasets into one unified dataset"""
        logger.info("\n" + "="*80)
        logger.info("MERGING DATASETS")
        logger.info("="*80 + "\n")
        
        # Start with weather data as the base (usually has the most comprehensive time series)
        merged_df = files_info['weather'].copy()
        logger.info(f"Starting with Weather data: {merged_df.shape}")
        
        # Identify merge keys (common columns)
        # Common patterns: datetime, location identifiers
        merge_keys = []
        
        # Check for common column names
        for col in merged_df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ['date', 'time', 'datetime', 'dt']):
                merge_keys.append(col)
            elif any(pattern in col_lower for pattern in ['location', 'city', 'station', 'lat', 'lon']):
                if col not in merge_keys:
                    merge_keys.append(col)
        
        logger.info(f"Identified merge keys: {merge_keys}")
        
        # If no automatic merge keys found, we'll need to examine the data
        if not merge_keys:
            logger.warning("No automatic merge keys found. Will attempt intelligent merging...")
            # Try to find common columns
            for name, df in files_info.items():
                if name != 'weather' and df is not None:
                    common = set(merged_df.columns) & set(df.columns)
                    if common:
                        logger.info(f"Common columns with {name}: {common}")
        
        # Merge each dataset
        datasets_to_merge = [
            ('location', files_info['location']),
            ('astronomical', files_info['astronomical']),
            ('air_quality', files_info['air_quality'])
        ]
        
        for name, df in datasets_to_merge:
            if df is not None:
                try:
                    # Find common columns between merged_df and current df
                    common_cols = list(set(merged_df.columns) & set(df.columns))
                    
                    if common_cols:
                        logger.info(f"\nMerging {name} on columns: {common_cols}")
                        
                        # Perform merge
                        before_shape = merged_df.shape
                        merged_df = pd.merge(
                            merged_df, 
                            df, 
                            on=common_cols, 
                            how='left',
                            suffixes=('', f'_{name}')
                        )
                        after_shape = merged_df.shape
                        
                        logger.info(f"  Before: {before_shape} → After: {after_shape}")
                        logger.info(f"  Added {after_shape[1] - before_shape[1]} new columns")
                    else:
                        logger.warning(f"No common columns found with {name}. Skipping merge.")
                        logger.info(f"  {name} columns: {list(df.columns)[:10]}...")
                        
                except Exception as e:
                    logger.error(f"Error merging {name}: {str(e)}")
        
        return merged_df
    
    def prepare_for_tft(self, df):
        """Prepare merged dataset for TFT model training"""
        logger.info("\n" + "="*80)
        logger.info("PREPARING FOR TFT MODEL")
        logger.info("="*80 + "\n")
        
        # Identify datetime column
        datetime_col = None
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    datetime_col = col
                    logger.info(f"Found datetime column: {col}")
                    break
                except:
                    continue
        
        # Sort by datetime if found
        if datetime_col:
            df = df.sort_values(datetime_col)
            logger.info(f"Sorted by {datetime_col}")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        logger.info(f"\nHandling missing values in {len(numeric_cols)} numeric columns...")
        
        # Fill numeric columns with forward fill then backward fill
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Fill remaining with median
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns - fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        
        logger.info(f"✓ Missing values handled")
        logger.info(f"  Remaining nulls: {df.isnull().sum().sum()}")
        
        return df
    
    def save_merged_dataset(self, df, output_filename='merged_weather_dataset.csv'):
        """Save the merged dataset"""
        output_path = self.data_dir / output_filename
        
        logger.info("\n" + "="*80)
        logger.info("SAVING MERGED DATASET")
        logger.info("="*80 + "\n")
        
        try:
            # Save as CSV (more efficient for large datasets)
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Saved to: {output_path}")
            logger.info(f"  Final shape: {df.shape}")
            logger.info(f"  File size: {output_path.stat().st_size / 1024**2:.2f} MB")
            
            # Also save a sample for quick inspection
            sample_path = self.data_dir / f"sample_{output_filename}"
            df.head(1000).to_csv(sample_path, index=False)
            logger.info(f"✓ Saved sample (1000 rows) to: {sample_path}")
            
            # Save column information
            info_path = self.data_dir / "merged_dataset_info.txt"
            with open(info_path, 'w') as f:
                f.write("MERGED DATASET INFORMATION\n")
                f.write("="*80 + "\n\n")
                f.write(f"Created: {datetime.now()}\n")
                f.write(f"Total rows: {len(df)}\n")
                f.write(f"Total columns: {len(df.columns)}\n\n")
                f.write("COLUMNS:\n")
                f.write("-"*80 + "\n")
                for i, col in enumerate(df.columns, 1):
                    f.write(f"{i:3d}. {col:40s} ({df[col].dtype})\n")
                f.write("\n" + "="*80 + "\n")
                f.write("DATA TYPES SUMMARY:\n")
                f.write(str(df.dtypes.value_counts()))
                f.write("\n\n" + "="*80 + "\n")
                f.write("BASIC STATISTICS:\n")
                f.write(str(df.describe()))
            
            logger.info(f"✓ Saved dataset info to: {info_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"✗ Error saving dataset: {str(e)}")
            return None
    
    def run(self):
        """Main execution pipeline"""
        logger.info("\n" + "="*80)
        logger.info("MAUSAM-VAANI DATA MERGING PIPELINE")
        logger.info("="*80 + "\n")
        
        start_time = datetime.now()
        
        # Step 1: Examine all files
        files_info = self.examine_all_files()
        
        # Step 2: Identify merge keys
        common_cols = self.identify_merge_keys(files_info)
        
        # Step 3: Merge datasets
        merged_df = self.merge_datasets(files_info)
        
        # Step 4: Prepare for TFT
        prepared_df = self.prepare_for_tft(merged_df)
        
        # Step 5: Save
        output_path = self.save_merged_dataset(prepared_df)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Output: {output_path}")
        logger.info("="*80 + "\n")
        
        return output_path

if __name__ == "__main__":
    # Set data directory
    data_dir = Path(__file__).parent.parent / "data"
    
    # Create merger instance
    merger = DatasetMerger(data_dir)
    
    # Run the pipeline
    output_file = merger.run()
    
    if output_file:
        print(f"\n✓ SUCCESS! Merged dataset saved to: {output_file}")
        print(f"\nNext steps:")
        print(f"1. Review the merged dataset: {output_file}")
        print(f"2. Check the dataset info: {data_dir / 'merged_dataset_info.txt'}")
        print(f"3. Use this dataset for TFT model training")
    else:
        print("\n✗ FAILED! Check the logs for details.")
        sys.exit(1)
