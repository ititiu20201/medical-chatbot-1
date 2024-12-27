import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Tuple
from underthesea import word_tokenize
import logging
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataPreprocessor:
    def __init__(self, raw_data_path: str = 'data/raw'):
        """Initialize the preprocessor with path to raw data"""
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path('data/processed')
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    def preprocess_text(self, text: str) -> str:
        """Preprocess Vietnamese text"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Vietnamese word tokenization
        text = word_tokenize(text, format="text")
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text

    def load_disease_data(self) -> pd.DataFrame:
        """Load and preprocess the disease symptom data"""
        try:
            df = pd.read_csv(self.raw_data_path / 'disease_symptom.csv', sep=';')
            df.columns = ['Medical Specialty', 'Disease Name', 'Symptom']
            return df
        except Exception as e:
            logger.error(f"Error loading disease data: {str(e)}")
            return pd.DataFrame()

    def create_training_data(self) -> pd.DataFrame:
        """Create training dataset"""
        try:
            training_data = []
            specialties = set()

            # Process disease-symptom data
            disease_data = self.load_disease_data()
            
            for _, row in disease_data.iterrows():
                try:
                    specialty = row['Medical Specialty'].strip()
                    disease = row['Disease Name'].strip()
                    symptoms = ast.literal_eval(row['Symptom'])
                    
                    specialties.add(specialty)
                    training_data.append({
                        'input': ' '.join(symptoms),
                        'specialty': specialty,
                        'output_type': 'specialty',
                        'disease': disease
                    })
                except Exception as e:
                    logger.debug(f"Error processing row: {str(e)}")
                    continue

            # Load conversation data
            for filename in ['alpaca_data.json', 'chatdoctor5k.json']:
                file_path = self.raw_data_path / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            conv_data = json.load(f)
                            for item in conv_data:
                                if isinstance(item, dict) and 'input' in item and 'output' in item:
                                    training_data.append({
                                        'input': self.preprocess_text(item['input']),
                                        'output': item['output'],
                                        'output_type': 'conversation'
                                    })
                    except Exception as e:
                        logger.error(f"Error loading {filename}: {str(e)}")

            # Create DataFrame
            df = pd.DataFrame(training_data)
            
            # Save specialty mapping
            specialty_map = {spec: idx for idx, spec in enumerate(sorted(specialties))}
            with open(self.processed_data_path / 'specialty_map.json', 'w', encoding='utf-8') as f:
                json.dump(specialty_map, f, ensure_ascii=False, indent=2)

            # Augment data for specialties with few samples
            min_samples_per_specialty = 3
            augmented_data = []
            
            for specialty in specialties:
                specialty_samples = df[df['specialty'] == specialty]
                if len(specialty_samples) < min_samples_per_specialty and len(specialty_samples) > 0:
                    # Repeat existing samples
                    samples_needed = min_samples_per_specialty - len(specialty_samples)
                    augmented = specialty_samples.sample(n=samples_needed, replace=True).copy()
                    augmented_data.extend(augmented.to_dict('records'))
            
            # Add augmented data
            if augmented_data:
                df = pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)

            # Print statistics
            logger.info(f"Created dataset with {len(df)} samples")
            for specialty in specialties:
                count = len(df[df['specialty'] == specialty])
                logger.info(f"  {specialty}: {count} samples")
            
            return df

        except Exception as e:
            logger.error(f"Error creating training data: {str(e)}")
            raise

    def save_processed_data(self, output_path: str = 'data/processed'):
        """Save processed data to files with stratified split"""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get training data
            df = self.create_training_data()
            
            # Ensure minimum samples per specialty
            train_data = []
            val_data = []
            test_data = []
            
            # Split by specialty
            for specialty in df['specialty'].unique():
                specialty_data = df[df['specialty'] == specialty].copy()
                n_samples = len(specialty_data)
                
                if n_samples >= 3:  # If we have enough samples
                    n_train = max(2, int(0.6 * n_samples))
                    n_val = max(1, int(0.2 * n_samples))
                    
                    # Shuffle data
                    specialty_data = specialty_data.sample(frac=1, random_state=42)
                    
                    train_data.append(specialty_data.iloc[:n_train])
                    val_data.append(specialty_data.iloc[n_train:n_train + n_val])
                    test_data.append(specialty_data.iloc[n_train + n_val:])
                else:  # If we have less than 3 samples
                    # Put samples in train set
                    train_data.append(specialty_data)
                    
                    # Create synthetic samples for val and test
                    if len(specialty_data) > 0:
                        val_data.append(specialty_data.sample(n=1, replace=True))
                        test_data.append(specialty_data.sample(n=1, replace=True))
            
            # Add conversation data to training set
            conversation_data = df[df['output_type'] == 'conversation'].copy()
            if len(conversation_data) > 0:
                n_train = int(0.8 * len(conversation_data))
                n_val = int(0.1 * len(conversation_data))
                
                conversation_data = conversation_data.sample(frac=1, random_state=42)
                train_data.append(conversation_data.iloc[:n_train])
                val_data.append(conversation_data.iloc[n_train:n_train + n_val])
                test_data.append(conversation_data.iloc[n_train + n_val:])

            # Combine and save
            train_df = pd.concat(train_data, ignore_index=True).sample(frac=1, random_state=42)
            val_df = pd.concat(val_data, ignore_index=True).sample(frac=1, random_state=42)
            test_df = pd.concat(test_data, ignore_index=True).sample(frac=1, random_state=42)
            
            # Save splits
            train_df.to_csv(output_path / 'train.csv', index=False)
            val_df.to_csv(output_path / 'val.csv', index=False)
            test_df.to_csv(output_path / 'test.csv', index=False)
            
            logger.info(f"Saved processed data to {output_path}")
            logger.info(f"Train samples: {len(train_df)}")
            logger.info(f"Validation samples: {len(val_df)}")
            logger.info(f"Test samples: {len(test_df)}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        try:
            self.save_processed_data()
            return True
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return False