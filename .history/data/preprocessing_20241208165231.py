import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Tuple
from underthesea import word_tokenize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataPreprocessor:
    def __init__(self, raw_data_path: str = 'data/raw'):
        """
        Initialize the preprocessor with path to raw data
        
        Args:
            raw_data_path (str): Path to directory containing raw data files
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data = None

    def load_json_file(self, file_path: Path) -> Dict:
        """Load JSON file and return data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return {}

    def load_csv_file(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file and return DataFrame"""
        try:
            return pd.read_csv(file_path, encoding='utf-8', sep=';')
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return pd.DataFrame()

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess Vietnamese text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Vietnamese word tokenization
        text = word_tokenize(text, format="text")
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text

    def process_alpaca_data(self, data: List[Dict]) -> pd.DataFrame:
        """Process alpaca format data"""
        processed = []
        for item in data:
            processed.append({
                'instruction': self.preprocess_text(item.get('instruction', '')),
                'input': self.preprocess_text(item.get('input', '')),
                'output': self.preprocess_text(item.get('output', ''))
            })
        return pd.DataFrame(processed)

    def process_disease_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process disease database"""
        processed = []
        for _, row in data.iterrows():
            processed.append({
                'specialty': row.get('Medical Specialty', ''),
                'disease': row.get('Disease Name', ''),
                'symptoms': self.preprocess_text(str(row.get('Symptom', ''))),
                'tests': self.preprocess_text(str(row.get('Medical Tests', ''))),
                'medications': self.preprocess_text(str(row.get('Medications', '')))
            })
        return pd.DataFrame(processed)

    def load_and_preprocess_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess all data files
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                (conversation_data, disease_data, symptom_data)
        """
        # Load conversation data
        alpaca_path = self.raw_data_path / 'alpaca_data.json'
        chatdoctor_path = self.raw_data_path / 'chatdoctor5k.json'
        
        alpaca_data = self.process_alpaca_data(self.load_json_file(alpaca_path))
        chatdoctor_data = self.process_alpaca_data(self.load_json_file(chatdoctor_path))
        conversation_data = pd.concat([alpaca_data, chatdoctor_data], ignore_index=True)
        
        # Load disease data
        disease_db_path = self.raw_data_path / 'disease_database_mini.csv'
        disease_data = self.process_disease_data(self.load_csv_file(disease_db_path))
        
        # Load symptom data
        symptom_path = self.raw_data_path / 'disease_symptom.csv'
        symptom_data = self.load_csv_file(symptom_path)
        symptom_data['Symptom'] = symptom_data['Symptom'].apply(self.preprocess_text)
        
        return conversation_data, disease_data, symptom_data

    def create_training_data(self) -> pd.DataFrame:
        """
        Create training dataset by combining all processed data
        
        Returns:
            pd.DataFrame: Combined training dataset
        """
        conv_data, disease_data, symptom_data = self.load_and_preprocess_all_data()
        
        # Combine conversation and disease data
        training_data = []
        
        # Add disease-symptom pairs
        for _, row in disease_data.iterrows():
            training_data.append({
                'input': row['symptoms'],
                'specialty': row['specialty'],
                'disease': row['disease'],
                'output_type': 'diagnosis'
            })
            
        # Add conversation samples
        for _, row in conv_data.iterrows():
            if row['input']:  # Only add if there's input
                training_data.append({
                    'input': row['input'],
                    'output': row['output'],
                    'output_type': 'conversation'
                })
        
        return pd.DataFrame(training_data)

    def save_processed_data(self, output_path: str = 'data/processed'):
        """
        Save processed data to files
        
        Args:
            output_path (str): Directory to save processed data
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        training_data = self.create_training_data()
        
        # Split into train/val/test
        train_size = int(0.8 * len(training_data))
        val_size = int(0.1 * len(training_data))
        
        indices = np.random.permutation(len(training_data))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Save splits
        training_data.iloc[train_indices].to_csv(
            output_path / 'train.csv', index=False, encoding='utf-8'
        )
        training_data.iloc[val_indices].to_csv(
            output_path / 'val.csv', index=False, encoding='utf-8'
        )
        training_data.iloc[test_indices].to_csv(
            output_path / 'test.csv', index=False, encoding='utf-8'
        )
        
        logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    preprocessor = MedicalDataPreprocessor()
    preprocessor.save_processed_data()