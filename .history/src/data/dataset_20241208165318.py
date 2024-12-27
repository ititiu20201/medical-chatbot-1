import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Optional
from transformers import AutoTokenizer
import logging

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
        """
        Initialize Medical Dataset
        
        Args:
            data_path (str): Path to processed data CSV file
            tokenizer_name (str): Name or path of tokenizer
            max_length (int): Maximum sequence length
            specialty_map (Dict[str, int], optional): Mapping of specialties to indices
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Create specialty mapping if not provided
        if specialty_map is None:
            unique_specialties = self.data['specialty'].dropna().unique()
            self.specialty_map = {spec: idx for idx, spec in enumerate(unique_specialties)}
        else:
            self.specialty_map = specialty_map
            
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        logger.info(f"Found {len(self.specialty_map)} unique specialties")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing input_ids, attention_mask,
                                   and labels (if available)
        """
        row = self.data.iloc[idx]
        
        # Prepare input text
        input_text = str(row['input'])
        
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
        }
        
        # Add labels if available
        if 'specialty' in row and pd.notna(row['specialty']):
            item['labels'] = torch.tensor(
                self.specialty_map.get(row['specialty'], -1),
                dtype=torch.long
            )
        
        # Add output type
        if 'output_type' in row:
            item['output_type'] = row['output_type']
        
        return item

    def get_specialty_map(self) -> Dict[str, int]:
        """Get mapping of specialties to indices"""
        return self.specialty_map

    def get_inverse_specialty_map(self) -> Dict[int, str]:
        """Get mapping of indices to specialties"""
        return {v: k for k, v in self.specialty_map.items()}

    @staticmethod
    def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader
        
        Args:
            batch (list): List of samples
            
        Returns:
            Dict[str, torch.Tensor]: Batched samples
        """
        # Stack all tensors in the batch
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        collated = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        # Add labels if they exist
        if 'labels' in batch[0]:
            labels = torch.stack([item['labels'] for item in batch])
            collated['labels'] = labels
            
        # Add output types if they exist
        if 'output_type' in batch[0]:
            output_types = [item['output_type'] for item in batch]
            collated['output_type'] = output_types
            
        return collated