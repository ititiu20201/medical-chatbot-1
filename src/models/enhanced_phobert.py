import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMedicalPhoBERT(nn.Module):
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_specialties: int = 0,
        num_symptoms: int = 0,
        num_treatments: int = 0,
        dropout_rate: float = 0.1
    ):
        """Enhanced Medical PhoBERT model"""
        super().__init__()
        
        # Load PhoBERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name)
        
        # Freeze initial layers
        self._freeze_layers(num_layers_to_freeze=8)
        
        # Task-specific heads
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = self.config.hidden_size  # Should be 768 for PhoBERT base
        
        # Specialty prediction
        self.specialty_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_specialties)
        )
        
        # Symptom recognition
        self.symptom_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_symptoms)
        )
        
        # Treatment recommendation
        combined_size = hidden_size + num_symptoms  # Concatenated features size
        self.treatment_recommender = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_treatments)
        )
        
        logger.info(
            f"Initialized Enhanced Medical PhoBERT with:"
            f"\n- {num_specialties} specialties"
            f"\n- {num_symptoms} symptoms"
            f"\n- {num_treatments} treatments"
        )

    def _freeze_layers(self, num_layers_to_freeze: int):
        """Freeze initial layers"""
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False
            
        for layer in self.phobert.encoder.layer[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Get PhoBERT outputs
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        sequence_output = self.dropout(sequence_output)
        
        result = {}
        
        # Specialty prediction
        specialty_logits = self.specialty_classifier(sequence_output)
        result["specialty_logits"] = specialty_logits
        
        # Calculate loss only for samples with valid labels
        if labels is not None:
            # Create mask for valid labels (not -100)
            valid_mask = labels != -100
            if valid_mask.any():
                valid_logits = specialty_logits[valid_mask]
                valid_labels = labels[valid_mask]
                
                loss_fct = nn.CrossEntropyLoss()
                specialty_loss = loss_fct(valid_logits, valid_labels)
                result["specialty_loss"] = specialty_loss
        
        # Symptom detection
        symptom_logits = self.symptom_detector(sequence_output)
        result["symptom_logits"] = symptom_logits
        
        # Treatment recommendation
        symptom_features = torch.sigmoid(symptom_logits)
        combined_features = torch.cat([sequence_output, symptom_features], dim=-1)
        treatment_logits = self.treatment_recommender(combined_features)
        result["treatment_logits"] = treatment_logits

        return result

    def save_pretrained(self, save_path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        num_specialties: int,
        num_symptoms: int,
        num_treatments: int
    ):
        """Load model"""
        checkpoint = torch.load(load_path)
        model = cls(
            num_specialties=num_specialties,
            num_symptoms=num_symptoms,
            num_treatments=num_treatments
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {load_path}")
        return model