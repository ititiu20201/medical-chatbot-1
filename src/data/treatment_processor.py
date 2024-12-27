import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TreatmentProcessor:
    def __init__(self, data_path: str = 'data/raw'):
        """
        Initialize treatment processor
        
        Args:
            data_path: Path to data directory containing medical databases
        """
        self.data_path = Path(data_path)
        self.disease_db = None
        self.symptom_db = None
        self.treatment_mapping = defaultdict(list)
        self.load_databases()

    def load_databases(self):
        """Load and prepare medical databases"""
        try:
            # Load disease database
            self.disease_db = pd.read_csv(
                self.data_path / 'disease_database_mini.csv', 
                sep=';'
            )
            
            # Create treatment mapping
            for _, row in self.disease_db.iterrows():
                disease = row['Disease Name']
                specialty = row['Medical Specialty']
                medications = eval(row['Medications']) if isinstance(row['Medications'], str) else []
                tests = eval(row['Medical Tests']) if isinstance(row['Medical Tests'], str) else []
                
                self.treatment_mapping[disease] = {
                    'specialty': specialty,
                    'medications': medications,
                    'tests': tests
                }
            
            logger.info(f"Loaded {len(self.treatment_mapping)} disease treatments")
            
        except Exception as e:
            logger.error(f"Error loading databases: {str(e)}")

    def get_treatment_recommendation(
        self,
        symptoms: List[str],
        medical_history: Optional[Dict] = None
    ) -> Dict:
        """
        Get treatment recommendations based on symptoms
        
        Args:
            symptoms: List of patient symptoms
            medical_history: Optional medical history dict
            
        Returns:
            Dict containing treatment recommendations
        """
        recommendations = {
            'primary_treatments': [],
            'alternative_treatments': [],
            'recommended_tests': [],
            'precautions': [],
            'specialties': set()
        }
        
        # Match symptoms with diseases and their treatments
        matched_diseases = []
        for disease, info in self.treatment_mapping.items():
            disease_symptoms = eval(
                self.disease_db[
                    self.disease_db['Disease Name'] == disease
                ]['Symptom'].iloc[0]
            )
            
            # Calculate symptom match score
            matching_symptoms = set(symptoms) & set(disease_symptoms)
            if matching_symptoms:
                matched_diseases.append({
                    'disease': disease,
                    'match_score': len(matching_symptoms) / len(symptoms),
                    'info': info
                })
        
        # Sort by match score
        matched_diseases.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Get recommendations from top matches
        for match in matched_diseases[:3]:  # Consider top 3 matches
            recommendations['specialties'].add(match['info']['specialty'])
            
            # Add medications
            if match['match_score'] > 0.7:  # High confidence match
                recommendations['primary_treatments'].extend(match['info']['medications'])
            else:  # Lower confidence match
                recommendations['alternative_treatments'].extend(match['info']['medications'])
            
            # Add recommended tests
            recommendations['recommended_tests'].extend(match['info']['tests'])

        # Add precautions based on medical history
        if medical_history:
            allergies = medical_history.get('allergies', [])
            # Filter out medications that patient is allergic to
            recommendations['primary_treatments'] = [
                med for med in recommendations['primary_treatments']
                if not any(allergy in med for allergy in allergies)
            ]
            
            # Add specific precautions based on conditions
            if medical_history.get('chronic_conditions'):
                recommendations['precautions'].append(
                    "Requires special consideration due to existing conditions"
                )

        # Remove duplicates and convert sets to lists
        recommendations['specialties'] = list(recommendations['specialties'])
        recommendations['primary_treatments'] = list(set(recommendations['primary_treatments']))
        recommendations['alternative_treatments'] = list(set(recommendations['alternative_treatments']))
        recommendations['recommended_tests'] = list(set(recommendations['recommended_tests']))

        return recommendations

    def add_new_treatment(
        self,
        disease: str,
        specialty: str,
        medications: List[str],
        tests: List[str]
    ) -> bool:
        """
        Add new treatment information to the database
        
        Args:
            disease: Disease name
            specialty: Medical specialty
            medications: List of medications
            tests: List of medical tests
            
        Returns:
            bool: Success status
        """
        try:
            # Add to treatment mapping
            self.treatment_mapping[disease] = {
                'specialty': specialty,
                'medications': medications,
                'tests': tests
            }
            
            # Update disease database
            new_row = pd.DataFrame([{
                'Disease Name': disease,
                'Medical Specialty': specialty,
                'Medications': str(medications),
                'Medical Tests': str(tests)
            }])
            
            self.disease_db = pd.concat([self.disease_db, new_row], ignore_index=True)
            
            # Save updated database
            self.disease_db.to_csv(
                self.data_path / 'disease_database_mini.csv',
                sep=';',
                index=False
            )
            
            logger.info(f"Added new treatment for {disease}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding new treatment: {str(e)}")
            return False

    def update_treatment(
        self,
        disease: str,
        updates: Dict
    ) -> bool:
        """
        Update existing treatment information
        
        Args:
            disease: Disease name
            updates: Dictionary containing updates
            
        Returns:
            bool: Success status
        """
        try:
            if disease not in self.treatment_mapping:
                raise ValueError(f"Disease {disease} not found in database")
                
            # Update treatment mapping
            self.treatment_mapping[disease].update(updates)
            
            # Update disease database
            mask = self.disease_db['Disease Name'] == disease
            for key, value in updates.items():
                if key in ['medications', 'tests']:
                    self.disease_db.loc[mask, key.capitalize()] = str(value)
                else:
                    self.disease_db.loc[mask, key] = value
            
            # Save updated database
            self.disease_db.to_csv(
                self.data_path / 'disease_database_mini.csv',
                sep=';',
                index=False
            )
            
            logger.info(f"Updated treatment for {disease}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating treatment: {str(e)}")
            return False

    def export_treatment_database(self, export_path: Optional[str] = None) -> str:
        """
        Export current treatment database
        
        Args:
            export_path: Optional path for export
            
        Returns:
            str: Path to exported database
        """
        if export_path is None:
            export_path = self.data_path / 'treatment_database_export.json'
            
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.treatment_mapping, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Exported treatment database to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting treatment database: {str(e)}")
            return ""