import logging
from pathlib import Path
from typing import Optional, Dict, List
import json
import pandas as pd
from datetime import datetime

from .preprocessing import MedicalDataPreprocessor
from .validator import DataValidator
from .analyzer import DataAnalyzer
from .dataset import MedicalDataset
from .collector import DataCollector

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
        self.preprocessor = MedicalDataPreprocessor(raw_data_path)
        self.validator = DataValidator()
        self.analyzer = DataAnalyzer(raw_data_path, processed_data_path)
        self.collector = DataCollector()

    def process_all_data_files(self):
        """Process all available data files"""
        data_files = {
            'json_files': [
                'alpaca_data.json',
                'chatdoctor5k.json',
                'demodata_1.json',
                'demodata_2.json',
                'in_out_data.json'
            ],
            'csv_files': [
                'disease_database_mini.csv',
                'disease_symptom.csv',
                'format_dataset.csv',
                'format_dataset(1).csv',
                'In_Out_data.csv'
            ]
        }
        
        processed_data = {}
        
        # Process JSON files
        for json_file in data_files['json_files']:
            file_path = self.raw_data_path / json_file
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                processed_data[json_file] = self.preprocessor.process_json_data(data)
        
        # Process CSV files
        for csv_file in data_files['csv_files']:
            file_path = self.raw_data_path / csv_file
            if file_path.exists():
                data = pd.read_csv(file_path, sep=';')
                processed_data[csv_file] = self.preprocessor.process_csv_data(data)
        
        return processed_data

    def handle_patient_response(self, patient_id: str, response_data: Dict, category: str):
        """
        Handle new patient response data
        
        Args:
            patient_id: Unique identifier for the patient
            response_data: Dictionary containing patient response data
            category: Category of the response (e.g., 'symptoms', 'medical_history')
        """
        # Save response
        self.collector.collect_patient_response(patient_id, response_data, category)
        
        # Update relevant datasets
        if category == 'symptoms':
            self.collector.update_dataset(response_data, 'symptoms')
        elif category == 'medical_history':
            self.collector.update_dataset(response_data, 'medical_qa')
            
        # Reprocess affected datasets
        self.process_all_data_files()
        
        # Update patient profile
        self.collector.create_patient_profile({
            'patient_id': patient_id,
            'last_response': category,
            'timestamp': datetime.now().isoformat()
        })

    def run_pipeline(self, include_patient_data: bool = True) -> bool:
        """
        Run the complete enhanced data pipeline
        
        Args:
            include_patient_data: Whether to include patient response data
            
        Returns:
            bool: True if pipeline runs successfully, False otherwise
        """
        try:
            # Step 1: Process all static data files
            logger.info("Processing all data files...")
            processed_data = self.process_all_data_files()
            
            # Step 2: Include patient response data if requested
            if include_patient_data:
                logger.info("Processing patient response data...")
                patient_stats = self.collector.get_response_statistics()
                logger.info(f"Processed {patient_stats['total_responses']} patient responses")
            
            # Step 3: Validate all data
            if self.validate_data:
                logger.info("Validating data...")
                validation_result = self.validator.run_validation(
                    self.raw_data_path,
                    self.processed_data_path
                )
                if not validation_result:
                    logger.error("Data validation failed!")
                    return False
            
            # Step 4: Analyze data
            if self.analyze_data:
                logger.info("Analyzing data...")
                self.analyzer.run_complete_analysis()
            
            # Step 5: Create/update datasets
            logger.info("Creating datasets...")
            train_dataset = MedicalDataset(
                data_path=str(self.processed_data_path / 'train.csv'),
                tokenizer_name='vinai/phobert-base',
                max_length=256
            )
            
            logger.info(
                f"Created dataset with {len(train_dataset)} samples and "
                f"{len(train_dataset.get_specialty_map())} specialties"
            )
            
            logger.info("Enhanced data pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in enhanced data pipeline: {str(e)}")
            return False

    def export_data_snapshot(
            self,
            export_path: Optional[str] = None
        ) -> str:
        """
        Export current state of all data
        
        Args:
            export_path: Optional path for export
            
        Returns:
            str: Path to exported data
        """
        if export_path is None:
            export_path = self.processed_data_path / 'data_snapshot'
            export_path.mkdir(exist_ok=True)
            
        try:
            # Export all processed data
            processed_data = self.process_all_data_files()
            for filename, data in processed_data.items():
                file_path = export_path / filename
                if filename.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                else:  # CSV files
                    data.to_csv(file_path, index=False, sep=';')
                    
            # Export patient response statistics
            stats = self.collector.get_response_statistics()
            stats_path = export_path / 'patient_response_stats.json'
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)
                
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting data snapshot: {str(e)}")
            return ""

def main():
    """Run the enhanced data pipeline"""
    pipeline = EnhancedDataPipeline()
    
    # Run initial pipeline
    success = pipeline.run_pipeline()
    
    if success:
        # Export data snapshot
        export_path = pipeline.export_data_snapshot()
        if export_path:
            logger.info(f"Data snapshot exported to {export_path}")
            logger.info("Pipeline completed successfully!")
        else:
            logger.error("Failed to export data snapshot!")
    else:
        logger.error("Pipeline failed!")

if __name__ == "__main__":
    main()