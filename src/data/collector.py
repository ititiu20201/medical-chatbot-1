import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize data collector
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.patient_data_dir = self.data_dir / 'patient_responses'
        
        # Create directories if they don't exist
        for directory in [self.raw_dir, self.processed_dir, self.patient_data_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def collect_patient_response(self, 
                               patient_id: str,
                               response_data: Dict,
                               category: str) -> str:
        """
        Collect and store patient response data
        
        Args:
            patient_id: Unique identifier for the patient
            response_data: Dictionary containing patient's response
            category: Category of the response (e.g., 'symptoms', 'medical_history')
            
        Returns:
            str: Path to saved response file
        """
        # Create patient directory if it doesn't exist
        patient_dir = self.patient_data_dir / patient_id
        patient_dir.mkdir(exist_ok=True)
        
        # Add metadata
        response_with_metadata = {
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'data': response_data
        }
        
        # Save response
        filename = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = patient_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(response_with_metadata, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Saved patient response to {file_path}")
        return str(file_path)

    def get_patient_history(self, patient_id: str) -> Dict[str, List[Dict]]:
        """
        Retrieve patient's response history
        
        Args:
            patient_id: Patient's unique identifier
            
        Returns:
            Dict containing categorized response history
        """
        patient_dir = self.patient_data_dir / patient_id
        if not patient_dir.exists():
            return {}
            
        history = {}
        for file_path in patient_dir.glob('*.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                response = json.load(f)
                
            category = response['category']
            if category not in history:
                history[category] = []
            history[category].append(response)
            
        return history

    def update_dataset(self, new_data: Dict, dataset_type: str):
        """
        Update existing dataset with new data
        
        Args:
            new_data: New data to add
            dataset_type: Type of dataset to update (e.g., 'symptoms', 'specialties')
        """
        # Define file paths based on dataset type
        file_paths = {
            'symptoms': self.raw_dir / 'disease_symptom.csv',
            'diseases': self.raw_dir / 'disease_database_mini.csv',
            'conversations': self.raw_dir / 'chatdoctor5k.json',
            'medical_qa': self.raw_dir / 'alpaca_data.json'
        }
        
        if dataset_type not in file_paths:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        file_path = file_paths[dataset_type]
        
        # Update based on file type
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, sep=';')
            new_df = pd.DataFrame([new_data])
            updated_df = pd.concat([df, new_df], ignore_index=True)
            updated_df.to_csv(file_path, sep=';', index=False)
            
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data.append(new_data)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
        logger.info(f"Updated {dataset_type} dataset with new data")

    def create_patient_profile(self, patient_data: Dict) -> bool:
        """
        Create or update patient profile
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            bool: Success status
        """
        try:
            patient_id = patient_data.get('patient_id')
            if not patient_id:
                raise ValueError("patient_id is required")
                
            profile_path = self.patient_data_dir / patient_id / 'profile.json'
            profile_path.parent.mkdir(exist_ok=True)
            
            # Add metadata
            profile_data = {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'profile': patient_data
            }
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Created/updated profile for patient {patient_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating patient profile: {str(e)}")
            return False

    def get_response_statistics(self, period: Optional[str] = None) -> Dict:
        """
        Get statistics about patient responses
        
        Args:
            period: Time period for statistics (e.g., 'day', 'week', 'month')
            
        Returns:
            Dict containing response statistics
        """
        all_responses = []
        for patient_dir in self.patient_data_dir.iterdir():
            if patient_dir.is_dir():
                for response_file in patient_dir.glob('*.json'):
                    if response_file.name != 'profile.json':
                        with open(response_file, 'r', encoding='utf-8') as f:
                            response = json.load(f)
                            all_responses.append(response)
        
        # Calculate statistics
        stats = {
            'total_responses': len(all_responses),
            'responses_by_category': {},
            'average_response_time': None  # To be implemented based on timestamps
        }
        
        for response in all_responses:
            category = response['category']
            if category not in stats['responses_by_category']:
                stats['responses_by_category'][category] = 0
            stats['responses_by_category'][category] += 1
            
        return stats

    def export_patient_data(self, patient_id: str, format: str = 'json') -> str:
        """
        Export all data for a specific patient
        
        Args:
            patient_id: Patient's unique identifier
            format: Export format ('json' or 'csv')
            
        Returns:
            str: Path to exported file
        """
        patient_dir = self.patient_data_dir / patient_id
        if not patient_dir.exists():
            raise ValueError(f"No data found for patient {patient_id}")
            
        # Collect all patient data
        data = {
            'profile': None,
            'responses': []
        }
        
        # Get profile
        profile_path = patient_dir / 'profile.json'
        if profile_path.exists():
            with open(profile_path, 'r', encoding='utf-8') as f:
                data['profile'] = json.load(f)
        
        # Get responses
        for response_file in patient_dir.glob('*.json'):
            if response_file.name != 'profile.json':
                with open(response_file, 'r', encoding='utf-8') as f:
                    response = json.load(f)
                    data['responses'].append(response)
        
        # Export based on format
        export_path = self.patient_data_dir / f"{patient_id}_export"
        if format == 'json':
            export_path = export_path.with_suffix('.json')
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        elif format == 'csv':
            export_path = export_path.with_suffix('.csv')
            df = pd.json_normalize(data['responses'])
            df.to_csv(export_path, index=False)
            
        return str(export_path)