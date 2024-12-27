import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
from transformers import AutoTokenizer
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "vinai/phobert-base",
        max_length: int = 256,
        specialty_map: Optional[Dict[str, int]] = None
    ):
        """Initialize Medical Dataset"""
        self.data = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Filter out rows with NaN inputs
        self.data = self.data.dropna(subset=['input'])
        
        # Create specialty mapping if not provided
        if specialty_map is None:
            unique_specialties = self.data[self.data['specialty'].notna()]['specialty'].unique()
            self.specialty_map = {spec: idx for idx, spec in enumerate(unique_specialties)}
        else:
            self.specialty_map = specialty_map
            
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        logger.info(f"Found {len(self.specialty_map)} unique specialties")

    def __len__(self) -> int:
        return len(self.data)

    def _process_input(self, input_text: str) -> str:
        """Process input text, handling lists and strings"""
        if isinstance(input_text, str):
            try:
                if input_text.strip().startswith('['):
                    symptoms = ast.literal_eval(input_text)
                    if isinstance(symptoms, list):
                        return ' '.join(symptoms)
            except:
                pass
            return input_text
        return ""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        row = self.data.iloc[idx]
        
        # Process input text
        input_text = self._process_input(row['input'])
        
        # Tokenize input
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(-100, dtype=torch.long)  # Default label for non-specialty samples
        }
        
        # Add proper label if this is a specialty classification sample
        if pd.notna(row.get('specialty')) and row['specialty'] in self.specialty_map:
            item['labels'] = torch.tensor(
                self.specialty_map[row['specialty']],
                dtype=torch.long
            )
        
        return item

    def get_specialty_map(self) -> Dict[str, int]:
        """Get mapping of specialties to indices"""
        return self.specialty_map

    def get_inverse_specialty_map(self) -> Dict[int, str]:
        """Get mapping of indices to specialties"""
        return {v: k for k, v in self.specialty_map.items()}

    @staticmethod
    def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
        """Custom collate function for DataLoader"""
        # Stack all tensors in the batch
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }