import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import Counter
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, raw_data_path: str = 'data/raw', 
                 processed_data_path: str = 'data/processed'):
        """Initialize data analyzer"""
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.analysis_results = {}
        
    def load_disease_data(self) -> pd.DataFrame:
        """Load and preprocess disease data"""
        try:
            # Read the raw text file
            with open(self.raw_data_path / 'disease_symptom.csv', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header
            data = []
            for line in lines[1:]:
                try:
                    # Split on first two semicolons only
                    parts = line.strip().split(';', 2)
                    if len(parts) == 3:
                        specialty, disease, symptoms = parts
                        # Clean disease name, removing any brackets and extra semicolons
                        disease = disease.strip('[]').split(';')[0].strip()
                        data.append({
                            'Medical Specialty': specialty.strip(),
                            'Disease Name': disease,
                            'Symptom': symptoms.strip()
                        })
                except Exception as e:
                    logger.debug(f"Error processing line: {line.strip()}, Error: {str(e)}")
                    continue
            
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} disease records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading disease data: {str(e)}")
            return pd.DataFrame()

    def analyze_medical_specialties(self) -> Dict:
        """Analyze distribution of medical specialties"""
        try:
            df = self.load_disease_data()
            
            # Get Medical Specialty column
            specialty_dist = df['Medical Specialty'].value_counts()
            
            # Calculate metrics
            metrics = {
                'total_specialties': len(specialty_dist),
                'distribution': specialty_dist.to_dict(),
                'most_common': specialty_dist.index[0] if len(specialty_dist) > 0 else "",
                'least_common': specialty_dist.index[-1] if len(specialty_dist) > 0 else ""
            }
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            specialty_dist.plot(kind='bar')
            plt.title('Distribution of Medical Specialties')
            plt.xlabel('Specialty')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.processed_data_path / 'specialty_distribution.png')
            plt.close()
            
            self.analysis_results['specialty_analysis'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing medical specialties: {str(e)}")
            return {}

    def analyze_symptoms(self) -> Dict:
        """Analyze symptom patterns"""
        try:
            df = self.load_disease_data()
            
            # Process symptoms column
            all_symptoms = []
            for symptom_list in df['Symptom']:
                try:
                    # Handle the string list format
                    symptoms = ast.literal_eval(symptom_list)
                    if isinstance(symptoms, list):
                        all_symptoms.extend(symptoms)
                except:
                    continue
            
            # Count symptom frequencies
            symptom_counts = Counter(all_symptoms)
            
            metrics = {
                'total_unique_symptoms': len(symptom_counts),
                'most_common_symptoms': dict(symptom_counts.most_common(10)),
                'average_symptoms_per_disease': len(all_symptoms) / len(df) if len(df) > 0 else 0
            }
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            pd.Series(dict(symptom_counts.most_common(15))).plot(kind='bar')
            plt.title('Top 15 Most Common Symptoms')
            plt.xlabel('Symptom')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.processed_data_path / 'symptom_distribution.png')
            plt.close()
            
            self.analysis_results['symptom_analysis'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing symptoms: {str(e)}")
            return {}

    def analyze_disease_patterns(self) -> Dict:
        """Analyze disease patterns and relationships"""
        try:
            df = self.load_disease_data()
            
            # Analyze disease-specialty relationships
            disease_per_specialty = df.groupby('Medical Specialty')['Disease Name'].count()
            
            # Analyze disease-symptom relationships with safer parsing
            symptom_counts = []
            for symptom_str in df['Symptom']:
                try:
                    # Clean up the symptom string and safely parse it
                    cleaned_str = symptom_str.strip()
                    if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
                        symptoms = ast.literal_eval(cleaned_str)
                        if isinstance(symptoms, list):
                            symptom_counts.append(len(symptoms))
                        else:
                            symptom_counts.append(0)
                    else:
                        symptom_counts.append(0)
                except Exception as e:
                    logger.debug(f"Error parsing symptom string: {str(e)}")
                    symptom_counts.append(0)
            
            # Convert to numpy array for calculations
            symptom_counts = np.array(symptom_counts)
            
            metrics = {
                'total_diseases': int(len(df)),
                'diseases_per_specialty': {k: int(v) for k, v in disease_per_specialty.to_dict().items()},
                'symptom_statistics': {
                    'average_symptoms': float(np.mean(symptom_counts)),
                    'min_symptoms': int(np.min(symptom_counts)),
                    'max_symptoms': int(np.max(symptom_counts))
                }
            }
            
            self.analysis_results['disease_analysis'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing disease patterns: {str(e)}")
            return {}

    def analyze_conversation_data(self) -> Dict:
        """Analyze conversation patterns"""
        try:
            # Load conversation data
            with open(self.raw_data_path / 'alpaca_data.json', 'r', encoding='utf-8') as f:
                alpaca_data = json.load(f)
            with open(self.raw_data_path / 'chatdoctor5k.json', 'r', encoding='utf-8') as f:
                chatdoctor_data = json.load(f)
            
            # Analyze conversation lengths
            alpaca_lengths = [len(item['output'].split()) for item in alpaca_data if 'output' in item]
            chatdoctor_lengths = [len(item['output'].split()) for item in chatdoctor_data if 'output' in item]
            
            metrics = {
                'total_conversations': len(alpaca_data) + len(chatdoctor_data),
                'average_response_length': {
                    'alpaca': float(np.mean(alpaca_lengths)) if alpaca_lengths else 0,
                    'chatdoctor': float(np.mean(chatdoctor_lengths)) if chatdoctor_lengths else 0
                },
                'response_length_distribution': {
                    'alpaca': {
                        'min': int(min(alpaca_lengths)) if alpaca_lengths else 0,
                        'max': int(max(alpaca_lengths)) if alpaca_lengths else 0,
                        'median': float(np.median(alpaca_lengths)) if alpaca_lengths else 0
                    },
                    'chatdoctor': {
                        'min': int(min(chatdoctor_lengths)) if chatdoctor_lengths else 0,
                        'max': int(max(chatdoctor_lengths)) if chatdoctor_lengths else 0,
                        'median': float(np.median(chatdoctor_lengths)) if chatdoctor_lengths else 0
                    }
                }
            }
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            if alpaca_lengths or chatdoctor_lengths:
                plt.hist([alpaca_lengths, chatdoctor_lengths], label=['Alpaca', 'ChatDoctor'])
                plt.title('Distribution of Response Lengths')
                plt.xlabel('Number of Words')
                plt.ylabel('Frequency')
                plt.legend()
            plt.tight_layout()
            plt.savefig(self.processed_data_path / 'response_length_distribution.png')
            plt.close()
            
            self.analysis_results['conversation_analysis'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing conversation data: {str(e)}")
            return {}

    def generate_report(self) -> None:
        """Generate comprehensive analysis report"""
        try:
            # Run all analyses if they haven't been run
            self.analyze_medical_specialties()
            self.analyze_symptoms()
            self.analyze_conversation_data()
            self.analyze_disease_patterns()
            
            # Create report
            report = {
                'summary': {
                    'total_specialties': self.analysis_results.get('specialty_analysis', {}).get('total_specialties', 0),
                    'total_diseases': self.analysis_results.get('disease_analysis', {}).get('total_diseases', 0),
                    'total_conversations': self.analysis_results.get('conversation_analysis', {}).get('total_conversations', 0),
                    'total_symptoms': self.analysis_results.get('symptom_analysis', {}).get('total_unique_symptoms', 0)
                },
                'detailed_analysis': {
                    'specialty_analysis': self.analysis_results.get('specialty_analysis', {}),
                    'symptom_analysis': self.analysis_results.get('symptom_analysis', {}),
                    'conversation_analysis': self.analysis_results.get('conversation_analysis', {}),
                    'disease_analysis': self.analysis_results.get('disease_analysis', {})
                },
                'generated_visualizations': [
                    'specialty_distribution.png',
                    'symptom_distribution.png',
                    'response_length_distribution.png'
                ]
            }
            
            # Save report
            output_path = self.processed_data_path / 'analysis_report.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Analysis report generated and saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")

    def run_complete_analysis(self) -> None:
        """Run complete data analysis pipeline"""
        logger.info("Starting data analysis...")
        
        # Create output directory if it doesn't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Generate comprehensive report
        self.generate_report()
        
        logger.info("Data analysis completed successfully!")