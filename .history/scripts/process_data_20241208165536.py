import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import MedicalDataPreprocessor
from src.data.dataset import MedicalDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Run the complete data processing pipeline
    """
    try:
        # Initialize preprocessor
        logger.info("Initializing data preprocessor...")
        preprocessor = MedicalDataPreprocessor(raw_data_path='data/raw')
        
        # Process and save data
        logger.info("Processing data...")
        preprocessor.save_processed_data(output_path='data/processed')
        
        # Test dataset creation
        logger.info("Testing dataset creation...")
        train_dataset = MedicalDataset(
            data_path='data/processed/train.csv',
            tokenizer_name='vinai/phobert-base',
            max_length=256
        )
        
        logger.info(f"Created dataset with {len(train_dataset)} samples")
        logger.info(f"Number of specialties: {len(train_dataset.get_specialty_map())}")
        
        # Test a single batch
        sample = train_dataset[0]
        logger.info(f"Sample input shape: {sample['input_ids'].shape}")
        
        logger.info("Data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()