import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import set_seed
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import json
from sklearn.metrics import f1_score

from ..models.enhanced_phobert import EnhancedMedicalPhoBERT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTrainer:
    def __init__(
        self,
        model: EnhancedMedicalPhoBERT,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        warmup_steps: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "data/models",
        task_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Enhanced Trainer
        
        Args:
            model: Enhanced Medical PhoBERT model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            device: Device to use
            output_dir: Output directory
            task_weights: Weights for different tasks
        """
        self.model = model.to(device)
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set task weights
        self.task_weights = task_weights or {
            "specialty": 1.0,
            "symptoms": 1.0,
            "treatment": 1.0
        }
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Initialize scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Trainer initialized on {device}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        task_losses = {task: 0.0 for task in self.task_weights.keys()}
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                specialty_labels=batch.get('specialty_labels'),
                symptom_labels=batch.get('symptom_labels'),
                treatment_labels=batch.get('treatment_labels')
            )
            
            # Calculate weighted loss
            loss = 0
            for task, weight in self.task_weights.items():
                if f"{task}_loss" in outputs:
                    task_loss = outputs[f"{task}_loss"]
                    weighted_loss = weight * task_loss
                    loss += weighted_loss
                    task_losses[task] += weighted_loss.item()
                    
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                **{f"{task}_loss": losses / (batch_idx + 1) 
                   for task, losses in task_losses.items()}
            })
        
        metrics = {
            'total_loss': total_loss / len(self.train_dataloader),
            **{f"{task}_loss": losses / len(self.train_dataloader) 
               for task, losses in task_losses.items()}
        }
        
        return metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        task_losses = {task: 0.0 for task in self.task_weights.keys()}
        
        specialty_preds = []
        specialty_labels = []
        treatment_preds = []
        treatment_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    specialty_labels=batch.get('specialty_labels'),
                    symptom_labels=batch.get('symptom_labels'),
                    treatment_labels=batch.get('treatment_labels')
                )
                
                # Calculate losses and collect predictions
                for task, weight in self.task_weights.items():
                    if f"{task}_loss" in outputs:
                        task_loss = outputs[f"{task}_loss"]
                        task_losses[task] += task_loss.item()
                        
                        if task == "specialty":
                            preds = torch.argmax(outputs["specialty_logits"], dim=1)
                            specialty_preds.extend(preds.cpu().numpy())
                            specialty_labels.extend(batch['specialty_labels'].cpu().numpy())
                        elif task == "treatment":
                            preds = (torch.sigmoid(outputs["treatment_logits"]) > 0.5).float()
                            treatment_preds.extend(preds.cpu().numpy())
                            treatment_labels.extend(batch['treatment_labels'].cpu().numpy())

        # Calculate metrics
        metrics = {
            'total_loss': total_loss / len(dataloader),
            **{f"{task}_loss": losses / len(dataloader) 
               for task, losses in task_losses.items()}
        }
        
        # Calculate accuracy for specialty prediction
        if specialty_preds:
            specialty_accuracy = np.mean(np.array(specialty_preds) == np.array(specialty_labels))
            metrics['specialty_accuracy'] = specialty_accuracy
        
        # Calculate F1 score for treatment prediction
        if treatment_preds:
            treatment_f1 = f1_score(treatment_labels, treatment_preds, average='macro')
            metrics['treatment_f1'] = treatment_f1
        
        return metrics

    def train(self) -> Dict[str, List[float]]:
        """Complete training process"""
        logger.info("Starting training...")
        history = {
            'train_loss': [],
            'val_loss': [],
            'specialty_accuracy': [],
            'treatment_f1': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['total_loss'])
            
            # Validate
            if self.val_dataloader:
                val_metrics = self.evaluate(self.val_dataloader)
                history['val_loss'].append(val_metrics['total_loss'])
                history['specialty_accuracy'].append(val_metrics.get('specialty_accuracy', 0))
                history['treatment_f1'].append(val_metrics.get('treatment_f1', 0))
                
                # Save best model
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    self.save_model('best_model.pt')
                
                logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Loss: {val_metrics['total_loss']:.4f}, "
                    f"Specialty Acc: {val_metrics.get('specialty_accuracy', 0):.4f}, "
                    f"Treatment F1: {val_metrics.get('treatment_f1', 0):.4f}"
                )
            
            # Save checkpoint
            self.save_model(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Final evaluation on test set
        if self.test_dataloader:
            test_metrics = self.evaluate(self.test_dataloader)
            logger.info("Test Results:")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Save test results
            test_results_path = self.output_dir / 'test_results.json'
            with open(test_results_path, 'w') as f:
                json.dump(test_metrics, f, indent=4)
        
        return history

    def save_model(self, filename: str):
        """
        Save model checkpoint
        
        Args:
            filename: Name of the checkpoint file
        """
        save_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, str(save_path))
        logger.info(f"Model saved to {save_path}")

    def load_model(self, filename: str):
        """
        Load model checkpoint
        
        Args:
            filename: Name of the checkpoint file to load
        """
        load_path = self.output_dir / filename
        checkpoint = torch.load(str(load_path))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.model.to(self.device)
        logger.info(f"Model loaded from {load_path}")