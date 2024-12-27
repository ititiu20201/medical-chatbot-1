import logging
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from ..models.enhanced_phobert import EnhancedMedicalPhoBERT
from ..data.dataset import MedicalDataset
from .enhanced_trainer import EnhancedTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_dataloaders(config: dict) -> tuple:
    """Create data loaders"""
    # Load datasets
    train_dataset = MedicalDataset(
        data_path=config['data']['train_file'],
        tokenizer_name=config['model']['name'],
        max_length=config['model']['max_length']
    )
    
    val_dataset = MedicalDataset(
        data_path=config['data']['val_file'],
        tokenizer_name=config['model']['name'],
        max_length=config['model']['max_length'],
        specialty_map=train_dataset.get_specialty_map()
    )
    
    test_dataset = MedicalDataset(
        data_path=config['data']['test_file'],
        tokenizer_name=config['model']['name'],
        max_length=config['model']['max_length'],
        specialty_map=train_dataset.get_specialty_map()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    return train_loader, val_loader, test_loader, train_dataset.get_specialty_map()

def train_model(config_path: str = 'configs/config.json'):
    """Main training function"""
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device(config['training']['device'])
    
    # Create dataloaders
    train_loader, val_loader, test_loader, specialty_map = create_dataloaders(config)
    
    # Initialize model
    model = EnhancedMedicalPhoBERT(
        model_name=config['model']['name'],
        num_specialties=len(specialty_map),
        num_symptoms=config['model']['num_symptoms'],
        num_treatments=config['model']['num_treatments']
    )
    
    # Define task weights
    task_weights = {
        'specialty': config['training'].get('specialty_weight', 1.0),
        'symptoms': config['training'].get('symptoms_weight', 1.0),
        'treatment': config['training'].get('treatment_weight', 1.0)
    }
    
    # Initialize trainer
    trainer = EnhancedTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        learning_rate=config['model']['learning_rate'],
        num_epochs=config['training']['epochs'],
        warmup_steps=config['model']['warmup_steps'],
        device=device,
        output_dir=config['paths']['model_save_path'],
        task_weights=task_weights
    )
    
    # Train model
    history = trainer.train()
    
    # Save training history
    history_path = Path(config['paths']['model_save_path']) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Save specialty mapping
    specialty_map_path = Path(config['paths']['model_save_path']) / 'specialty_map.json'
    with open(specialty_map_path, 'w') as f:
        json.dump(specialty_map, f, indent=4)
    
    logger.info("Training completed successfully!")

def main():
    """Run training"""
    try:
        train_model()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()