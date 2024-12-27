# import logging
# import sys
# from pathlib import Path
# import pandas as pd
# import json
# from typing import Dict, List, Tuple
# from sklearn.model_selection import train_test_split
# import numpy as np

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class DataProcessor:
#     def __init__(self, raw_data_path: str = 'data/raw', processed_data_path: str = 'data/processed'):
#         """Initialize data processor"""
#         self.raw_data_path = Path(raw_data_path)
#         self.processed_data_path = Path(processed_data_path)
#         self.processed_data_path.mkdir(parents=True, exist_ok=True)
#         self.specialty_map = {}
        
#     def load_conversation_data(self) -> List[Dict]:
#         """Load and combine conversation data from multiple sources"""
#         conversations = []
        
#         # Load alpaca data
#         try:
#             with open(self.raw_data_path / 'alpaca_data.json', 'r', encoding='utf-8') as f:
#                 alpaca_data = json.load(f)
#                 conversations.extend(alpaca_data)
#         except Exception as e:
#             logger.warning(f"Error loading alpaca data: {e}")
        
#         # Load chatdoctor data
#         try:
#             with open(self.raw_data_path / 'chatdoctor5k.json', 'r', encoding='utf-8') as f:
#                 chatdoctor_data = json.load(f)
#                 conversations.extend(chatdoctor_data)
#         except Exception as e:
#             logger.warning(f"Error loading chatdoctor data: {e}")
            
#         return conversations

#     def load_specialty_data(self) -> pd.DataFrame:
#         """Load and process specialty data"""
#         try:
#             # Load disease symptom data
#             symptom_df = pd.read_csv(
#                 self.raw_data_path / 'disease_symptom.csv',
#                 sep=';',
#                 encoding='utf-8'
#             )
            
#             # Extract specialties and create mapping
#             specialties = symptom_df['Medical Specialty'].unique()
#             self.specialty_map = {spec: idx for idx, spec in enumerate(specialties)}
            
#             return symptom_df
            
#         except Exception as e:
#             logger.error(f"Error loading specialty data: {e}")
#             return pd.DataFrame()

#     def prepare_training_data(self) -> Tuple[pd.DataFrame, Dict]:
#         """Prepare training data"""
#         # Load all data sources
#         conversations = self.load_conversation_data()
#         specialty_data = self.load_specialty_data()
        
#         training_data = []
        
#         # Process conversation data
#         for conv in conversations:
#             if 'input' in conv and 'output' in conv:
#                 training_data.append({
#                     'input': conv['input'],
#                     'output': conv['output'],
#                     'output_type': 'conversation'
#                 })
        
#         # Process specialty data
#         for _, row in specialty_data.iterrows():
#             specialty = row.get('Medical Specialty')
#             if specialty and specialty in self.specialty_map:
#                 training_data.append({
#                     'input': str(row.get('Symptom', '')),
#                     'specialty': specialty,
#                     'output_type': 'specialty'
#                 })
        
#         # Convert to DataFrame
#         df = pd.DataFrame(training_data)
        
#         # Save specialty mapping
#         with open(self.processed_data_path / 'specialty_map.json', 'w', encoding='utf-8') as f:
#             json.dump(self.specialty_map, f, ensure_ascii=False, indent=2)
        
#         return df

#     def split_and_save_data(self, df: pd.DataFrame, train_size: float = 0.8, val_size: float = 0.1):
#         """Split data into train/val/test sets and save"""
#         # First split into train and temp
#         train_df, temp_df = train_test_split(df, train_size=train_size, random_state=42)
        
#         # Split temp into val and test
#         val_size_adjusted = val_size / (1 - train_size)
#         val_df, test_df = train_test_split(temp_df, train_size=val_size_adjusted, random_state=42)
        
#         # Save datasets
#         train_df.to_csv(self.processed_data_path / 'train.csv', index=False, encoding='utf-8')
#         val_df.to_csv(self.processed_data_path / 'val.csv', index=False, encoding='utf-8')
#         test_df.to_csv(self.processed_data_path / 'test.csv', index=False, encoding='utf-8')
        
#         logger.info(f"Saved {len(train_df)} training samples")
#         logger.info(f"Saved {len(val_df)} validation samples")
#         logger.info(f"Saved {len(test_df)} test samples")

# def main():
#     """Run data processing"""
#     try:
#         processor = DataProcessor()
        
#         # Prepare data
#         logger.info("Preparing training data...")
#         df = processor.prepare_training_data()
        
#         # Split and save
#         logger.info("Splitting and saving data...")
#         processor.split_and_save_data(df)
        
#         logger.info("Data processing completed successfully!")
        
#     except Exception as e:
#         logger.error(f"Error in data processing: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()