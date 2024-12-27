# src/utils/verify_data.py
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_csv_file(file_path: Path) -> None:
    """Verify CSV file format"""
    try:
        df = pd.read_csv(file_path, sep=';' if 'raw' in str(file_path) else ',')
        logger.info(f"\nVerifying {file_path.name}:")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Number of rows: {len(df)}")
        logger.info(f"Sample row:")
        logger.info(df.iloc[0].to_dict())
        
        # Check null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning("Null values found:")
            logger.warning(null_counts[null_counts > 0])
    except Exception as e:
        logger.error(f"Error verifying {file_path}: {str(e)}")

def verify_json_file(file_path: Path) -> None:
    """Verify JSON file format"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"\nVerifying {file_path.name}:")
        logger.info(f"Number of records: {len(data)}")
        if isinstance(data, list) and len(data) > 0:
            logger.info("Sample record:")
            logger.info(data[0])
    except Exception as e:
        logger.error(f"Error verifying {file_path}: {str(e)}")

def verify_data_format() -> None:
    """Verify all data files"""
    data_dir = Path("data")
    
    # Verify processed data
    processed_files = [
        'train.csv',
        'val.csv',
        'test.csv'
    ]
    
    for file_name in processed_files:
        file_path = data_dir / 'processed' / file_name
        if file_path.exists():
            verify_csv_file(file_path)
    
    # Verify raw data
    raw_files = {
        'json': [
            'alpaca_data.json',
            'chatdoctor5k.json',
            'demodata_1.json',
            'demodata_2.json',
            'in_out_data.json'
        ],
        'csv': [
            'disease_database_mini.csv',
            'disease_symptom.csv',
            'format_dataset.csv',
            'format_dataset(1).csv',
            'In_Out_data.csv'
        ]
    }
    
    for json_file in raw_files['json']:
        file_path = data_dir / 'raw' / json_file
        if file_path.exists():
            verify_json_file(file_path)
    
    for csv_file in raw_files['csv']:
        file_path = data_dir / 'raw' / csv_file
        if file_path.exists():
            verify_csv_file(file_path)

if __name__ == "__main__":
    try:
        verify_data_format()
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")