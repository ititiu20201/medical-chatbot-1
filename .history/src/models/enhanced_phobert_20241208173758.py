import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple, List
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
        """
        Enhanced Medical PhoBERT model with multi-task capabilities
        
        Args:
            model_name: PhoBERT model name/path
            num_specialties: Number of medical specialties
            num_symptoms: Number of possible symptoms
            num_treatments: Number of possible treatments
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        # Load PhoBERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name)
        
        # Freeze initial layers
        self._freeze_layers(num_layers_to_freeze=8)
        
        # Task-specific heads
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = self.config.hidden_size
        
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
        self.treatment_recommender = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Concatenated with symptom features
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_treatments)
        )
        
        # Severity assessment
        self.severity_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        logger.info(
            f"Initialized Enhanced Medical PhoBERT with:"
            f"\n- {num_specialties} specialties"
            f"\n- {num_symptoms} symptoms"
            f"\n- {num_treatments} treatments"
        )

    def _freeze_layers(self, num_layers_to_freeze: int):
        """Freeze initial layers"""
        # Freeze embeddings
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False
            
        # Freeze encoder layers
        for layer in self.phobert.encoder.layer[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        specialty_labels: Optional[torch.Tensor] = None,
        symptom_labels: Optional[torch.Tensor] = None,
        treatment_labels: Optional[torch.Tensor] = None,
        task: str = "all"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            specialty_labels: Specialty labels (optional)
            symptom_labels: Symptom labels (optional)
            treatment_labels: Treatment labels (optional)
            task: Task to perform ("all", "specialty", "symptoms", "treatment")
            
        Returns:
            Dict containing outputs and losses
        """
        # Get PhoBERT outputs
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        sequence_output = self.dropout(sequence_output)
        
        result = {}
        total_loss = 0.0
        
        # Specialty prediction
        if task in ["all", "specialty"]:
            specialty_logits = self.specialty_classifier(sequence_output)
            result["specialty_logits"] = specialty_logits
            
            if specialty_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                specialty_loss = loss_fct(
                    specialty_logits.view(-1, self.specialty_classifier[-1].out_features),
                    specialty_labels.view(-1)
                )
                result["specialty_loss"] = specialty_loss
                total_loss += specialty_loss

        # Symptom recognition
        if task in ["all", "symptoms"]:
            symptom_logits = self.symptom_detector(sequence_output)
            result["symptom_logits"] = symptom_logits
            
            if symptom_labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                symptom_loss = loss_fct(symptom_logits, symptom_labels.float())
                result["symptom_loss"] = symptom_loss
                total_loss += symptom_loss

        # Treatment recommendation
        if task in ["all", "treatment"]:
            # Combine sequence output with symptom features
            symptom_features = torch.sigmoid(self.symptom_detector(sequence_output))
            combined_features = torch.cat([sequence_output, symptom_features], dim=-1)
            
            treatment_logits = self.treatment_recommender(combined_features)
            result["treatment_logits"] = treatment_logits
            
            if treatment_labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                treatment_loss = loss_fct(treatment_logits, treatment_labels.float())
                result["treatment_loss"] = treatment_loss
                total_loss += treatment_loss

        # Severity assessment
        severity_score = self.severity_assessor(sequence_output)
        result["severity_score"] = severity_score

        if total_loss > 0:
            result["total_loss"] = total_loss

        return result

    def predict_specialty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict top-k specialties"""
        self.eval()
        with torch.no_grad():
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task="specialty"
            )
            logits = outputs["specialty_logits"]
            probs = torch.softmax(logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
            
        return top_indices, top_probs

    def predict_treatments(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict treatments with confidence scores"""
        self.eval()
        with torch.no_grad():
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task="treatment"
            )
            logits = outputs["treatment_logits"]
            probs = torch.sigmoid(logits)
            
            predictions = (probs > threshold).float()
            
        return predictions, probs

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