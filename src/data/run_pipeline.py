import logging
from pathlib import Path
from typing import Optional, Dict, List
import json
import pandas as pd
from datetime import datetime

from src.data.preprocessing import MedicalDataPreprocessor
from src.data.validator import DataValidator
from src.data.analyzer import DataAnalyzer
from src.data.collector import DataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataPipeline:
    def __init__(
        self,
        raw_data_path: str = 'data/raw',
        processed_data_path: str = 'data/processed',
        validate_data: bool = True,
        analyze_data: bool = True
    ):
        """Initialize enhanced data pipeline"""
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.validate_data = validate_data
        self.analyze_data = analyze_data
        
        # Initialize components
        self.preprocessor = MedicalDataPreprocessor(str(raw_data_path))
        self.validator = DataValidator()
        self.analyzer = DataAnalyzer(str(raw_data_path), str(processed_data_path))
        self.collector = DataCollector()
        
        # Create directories if they don't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        (self.processed_data_path / 'models').mkdir(exist_ok=True)
        (self.processed_data_path / 'logs').mkdir(exist_ok=True)

    def process_all_data_files(self):
        """Process all data files"""
        try:
            logger.info("Starting to process all data files...")
            processed_data = self.preprocessor.create_training_data()
            self.preprocessor.save_processed_data()
            logger.info("Data processing completed successfully!")
            return processed_data
        except Exception as e:
            logger.error(f"Error processing data files: {str(e)}")
            raise

    def run_pipeline(self) -> bool:
        """Run the complete pipeline"""
        try:
            # Process data
            self.process_all_data_files()
            
            # Validate if requested
            if self.validate_data:
                logger.info("Validating data...")
                if not self.validator.run_validation(
                    self.raw_data_path,
                    self.processed_data_path
                ):
                    logger.error("Data validation failed!")
                    return False
            
            # Analyze if requested
            if self.analyze_data:
                logger.info("Analyzing data...")
                self.analyzer.run_complete_analysis()
            
            logger.info("Data pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in data pipeline: {str(e)}")
            return False

def main():
    """Run the data pipeline"""
    try:
        pipeline = EnhancedDataPipeline()
        success = pipeline.run_pipeline()
        
        if success:
            logger.info("Pipeline completed successfully!")
        else:
            logger.error("Pipeline failed!")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    main()