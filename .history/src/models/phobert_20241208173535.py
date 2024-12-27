import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalPhoBERT(nn.Module):
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_specialties: int = 0,
        dropout_rate: float = 0.1
    ):
        """
        Initialize Medical PhoBERT model
        
        Args:
            model_name (str): Name or path of pre-trained PhoBERT model
            num_specialties (int): Number of medical specialties to predict
            dropout_rate (float): Dropout rate for classification head
        """
        super().__init__()
        
        # Load PhoBERT configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name)
        
        # Freeze some layers if needed
        self._freeze_layers(num_layers_to_freeze=8)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.specialty_classifier = nn.Linear(self.config.hidden_size, num_specialties)
        
        # Additional heads for other tasks
        self.disease_classifier = nn.Linear(self.config.hidden_size, 1)  # Binary disease detection
        self.severity_regressor = nn.Linear(self.config.hidden_size, 1)  # Severity score
        
        logger.info(f"Initialized Medical PhoBERT with {num_specialties} specialties")

    def _freeze_layers(self, num_layers_to_freeze: int):
        """
        Freeze initial layers of PhoBERT
        
        Args:
            num_layers_to_freeze (int): Number of layers to freeze
        """
        # Freeze embeddings
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False
            
        # Freeze initial transformer layers
        for layer in self.phobert.encoder.layer[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
                
        logger.info(f"Frozen {num_layers_to_freeze} initial layers")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_type: str = "specialty"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor, optional): Ground truth labels
            output_type (str): Type of output required ("specialty", "disease", "severity")
            
        Returns:
            Dict[str, torch.Tensor]: Model outputs and loss if training
        """
        # Get PhoBERT outputs
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence output for classification
        sequence_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        sequence_output = self.dropout(sequence_output)
        
        # Initialize output dictionary
        result = {}
        
        # Task-specific heads
        if output_type == "specialty":
            logits = self.specialty_classifier(sequence_output)
            result["logits"] = logits
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.specialty_classifier.out_features),
                              labels.view(-1))
                result["loss"] = loss
                
        elif output_type == "disease":
            disease_pred = self.disease_classifier(sequence_output)
            result["disease_pred"] = torch.sigmoid(disease_pred)
            
        elif output_type == "severity":
            severity_score = self.severity_regressor(sequence_output)
            result["severity_score"] = severity_score
            
        return result

    def predict_specialty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k specialties
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            top_k (int): Number of top specialties to return
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted specialty indices and probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self(input_ids, attention_mask, output_type="specialty")
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
            
        return top_indices, top_probs

    def save_pretrained(self, save_path: str):
        """
        Save model to path
        
        Args:
            save_path (str): Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_path: str, num_specialties: int):
        """
        Load model from path
        
        Args:
            load_path (str): Path to load model from
            num_specialties (int): Number of specialties
            
        Returns:
            MedicalPhoBERT: Loaded model
        """
        checkpoint = torch.load(load_path)
        model = cls(num_specialties=num_specialties)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {load_path}")
        return model