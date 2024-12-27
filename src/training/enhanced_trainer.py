import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import logging
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTrainer:
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        warmup_steps: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "data/models",
        task_weights: Dict[str, float] = None
    ):
        """Initialize trainer"""
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
            [p for p in self.model.parameters() if p.requires_grad],
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
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_weights.keys()}
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        current_loss = 0.0  # Initialize current loss
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get model outputs
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch.get('labels')
            )
            
            # Initialize batch loss
            batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Get specialty loss if available
            if 'specialty_loss' in outputs and isinstance(outputs['specialty_loss'], torch.Tensor):
                batch_loss = outputs['specialty_loss']
                total_loss += batch_loss.item()
                task_losses['specialty'] += batch_loss.item()
                current_loss = batch_loss.item()  # Update current loss
                
                # Backward pass
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': current_loss,
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
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get model outputs
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch.get('labels')
                )
                
                # Get specialty loss if available
                if 'specialty_loss' in outputs:
                    loss = outputs['specialty_loss']
                    total_loss += loss.item()
                    
                    # Get predictions
                    logits = outputs['specialty_logits']
                    preds = torch.argmax(logits, dim=-1)
                    labels = batch['labels']
                    
                    # Only include valid labels (not -100)
                    valid_mask = labels != -100
                    if valid_mask.any():
                        valid_preds = preds[valid_mask].cpu().numpy()
                        valid_labels = labels[valid_mask].cpu().numpy()
                        all_preds.extend(valid_preds)
                        all_labels.extend(valid_labels)
        
        # Calculate metrics
        metrics = {
            'total_loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0,
        }
        
        # Calculate accuracy if we have predictions
        if all_preds:
            accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            metrics['accuracy'] = float(accuracy)
        
        return metrics

    def train(self) -> Dict[str, list]:
        """Complete training process"""
        logger.info("Starting training...")
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
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
                history['val_accuracy'].append(val_metrics.get('accuracy', 0))
                
                # Save best model
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    self.save_checkpoint('best_model.pt')
                
                logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Loss: {val_metrics['total_loss']:.4f}, "
                    f"Val Accuracy: {val_metrics.get('accuracy', 0):.4f}"
                )
            
            # Save checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        save_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        load_path = self.output_dir / filename
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded from {load_path}")