import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List, Union
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.validation_results = {
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

    def validate_json_file(self, file_path: Path) -> bool:
        """Validate JSON file format and content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Validate each item in the list
                for idx, item in enumerate(data):
                    if not isinstance(item, dict):
                        self.validation_results['errors'].append(
                            f"Invalid item format at index {idx} in {file_path}"
                        )
                        return False
                    
                    # Check required fields based on file type
                    if 'alpaca_data.json' in str(file_path):
                        required_fields = ['instruction', 'input', 'output']
                    elif 'chatdoctor5k.json' in str(file_path):
                        required_fields = ['instruction', 'input', 'output']
                    else:
                        required_fields = []

                    for field in required_fields:
                        if field not in item:
                            self.validation_results['errors'].append(
                                f"Missing required field '{field}' at index {idx} in {file_path}"
                            )
                            return False

            self.validation_results['statistics'][str(file_path)] = {
                'total_records': len(data) if isinstance(data, list) else 1
            }
            return True
            
        except json.JSONDecodeError as e:
            self.validation_results['errors'].append(
                f"Invalid JSON format in {file_path}: {str(e)}"
            )
            return False
        except Exception as e:
            self.validation_results['errors'].append(
                f"Error processing {file_path}: {str(e)}"
            )
            return False

    def validate_csv_file(self, file_path: Path) -> bool:
        """Validate CSV file format and content"""
        try:
            df = pd.read_csv(file_path, sep=';')
            file_name = file_path.name
            
            # Check required columns based on file type
            if file_name == 'disease_database_mini.csv':
                required_cols = ['Medical Specialty', 'Disease Name', 'Symptom']
            elif file_name == 'disease_symptom.csv':
                required_cols = ['Disease Name', 'Medical Specialty', 'Symptom']
            else:
                required_cols = []

            # Validate required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.validation_results['errors'].append(
                    f"Missing required columns in {file_path}: {missing_cols}"
                )
                return False

            # Check for empty values in important columns
            for col in required_cols:
                if col in df.columns and df[col].isnull().any():
                    self.validation_results['warnings'].append(
                        f"Found null values in column '{col}' in {file_path}"
                    )

            # Collect statistics
            self.validation_results['statistics'][str(file_path)] = {
                'total_records': len(df),
                'columns': list(df.columns),
                'null_counts': df.isnull().sum().to_dict()
            }
            return True

        except Exception as e:
            self.validation_results['errors'].append(
                f"Error processing {file_path}: {str(e)}"
            )
            return False

    def validate_output_format(self, specialty_predictions: List[str], 
                             queue_numbers: Dict[str, int]) -> bool:
        """Validate output format consistency"""
        try:
            # Validate specialty predictions
            if not specialty_predictions or not isinstance(specialty_predictions, list):
                self.validation_results['errors'].append(
                    "Invalid specialty predictions format"
                )
                return False

            # Validate queue numbers
            if not isinstance(queue_numbers, dict):
                self.validation_results['errors'].append(
                    "Invalid queue numbers format"
                )
                return False

            # Check if each specialty has a queue number
            for specialty in specialty_predictions:
                if specialty not in queue_numbers:
                    self.validation_results['warnings'].append(
                        f"Missing queue number for specialty: {specialty}"
                    )

            return True

        except Exception as e:
            self.validation_results['errors'].append(
                f"Error validating output format: {str(e)}"
            )
            return False

    def validate_processed_data(self, data_path: Path) -> bool:
        """Validate processed data files"""
        try:
            required_files = ['train.csv', 'val.csv', 'test.csv']
            
            for file_name in required_files:
                file_path = data_path / file_name
                if not file_path.exists():
                    self.validation_results['errors'].append(
                        f"Missing required file: {file_name}"
                    )
                    return False
                
                df = pd.read_csv(file_path)
                
                # Validate required columns
                required_cols = ['input', 'specialty', 'output_type']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    self.validation_results['errors'].append(
                        f"Missing required columns in {file_name}: {missing_cols}"
                    )
                    return False
                
                # Check data distribution
                self.validation_results['statistics'][file_name] = {
                    'total_records': len(df),
                    'specialty_distribution': df['specialty'].value_counts().to_dict(),
                    'output_type_distribution': df['output_type'].value_counts().to_dict()
                }
            
            return True

        except Exception as e:
            self.validation_results['errors'].append(
                f"Error validating processed data: {str(e)}"
            )
            return False

    def save_validation_report(self, output_path: Path):
        """Save validation results to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Validation report saved to {output_path}")

    def run_validation(self, raw_data_path: Path, processed_data_path: Path) -> bool:
        """Run complete validation pipeline"""
        is_valid = True
        
        # Validate raw data files
        json_files = list(raw_data_path.glob('*.json'))
        csv_files = list(raw_data_path.glob('*.csv'))
        
        for file_path in json_files:
            if not self.validate_json_file(file_path):
                is_valid = False
                
        for file_path in csv_files:
            if not self.validate_csv_file(file_path):
                is_valid = False
        
        # Validate processed data
        if not self.validate_processed_data(processed_data_path):
            is_valid = False
        
        return is_valid