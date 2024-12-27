import logging
from pathlib import Path
from typing import Optional

from .preprocessing import MedicalDataPreprocessor
from .validator import DataValidator
from .analyzer import DataAnalyzer
from .dataset import MedicalDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(
        self,
        raw_data_path: str = 'data/raw',
        processed_data_path: str = 'data/processed',
        validate_data: bool = True,
        analyze_data: bool = True
    ):
        """
        Initialize data pipeline
        
        Args:
            raw_data_path: Path to raw data directory
            processed_data_path: Path to processed data directory
            validate_data: Whether to validate data
            analyze_data: Whether to analyze data
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.validate_data = validate_data
        self.analyze_data = analyze_data

        # Create output directory
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self) -> bool:
        """
        Run the complete data processing pipeline
        
        Returns:
            bool: Whether pipeline completed successfully
        """
        try:
            # Step 1: Validate raw data
            if self.validate_data:
                logger.info("Validating raw data...")
                validator = DataValidator()
                if not validator.run_validation(self.raw_data_path, self.processed_data_path):
                    logger.error("Data validation failed!")
                    return False
                validator.save_validation_report(
                    self.processed_data_path / 'validation_report.json'
                )

            # Step 2: Preprocess data
            logger.info("Preprocessing data...")
            preprocessor = MedicalDataPreprocessor(raw_data_path=self.raw_data_path)
            preprocessor.save_processed_data(output_path=str(self.processed_data_path))

            # Step 3: Analyze data
            if self.analyze_data:
                logger.info("Analyzing data...")
                analyzer = DataAnalyzer(
                    raw_data_path=self.raw_data_path,
                    processed_data_path=self.processed_data_path
                )
                analyzer.run_complete_analysis()

            # Step 4: Test dataset creation
            logger.info("Testing dataset creation...")
            train_dataset = MedicalDataset(
                data_path=str(self.processed_data_path / 'train.csv'),
                tokenizer_name='vinai/phobert-base',
                max_length=256
            )
            
            logger.info(
                f"Created dataset with {len(train_dataset)} samples and "
                f"{len(train_dataset.get_specialty_map())} specialties"
            )

            logger.info("Data pipeline completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error in data pipeline: {str(e)}")
            return False

def main():
    """Run the data pipeline"""
    pipeline = DataPipeline()
    success = pipeline.run_pipeline()
    
    if success:
        logger.info("Pipeline completed successfully!")
    else:
        logger.error("Pipeline failed!")

if __name__ == "__main__":
    main()