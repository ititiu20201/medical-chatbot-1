import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, raw_data_path: str = 'data/raw', 
                 processed_data_path: str = 'data/processed'):
        """Initialize data analyzer"""
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.analysis_results = {}

    def analyze_medical_specialties(self) -> Dict:
        """Analyze distribution of medical specialties"""
        try:
            # Load disease database
            df = pd.read_csv(self.raw_data_path / 'disease_database_mini.csv', sep=';')
            
            # Analyze specialty distribution
            specialty_dist = df['Medical Specialty'].value_counts()
            
            # Calculate metrics
            metrics = {
                'total_specialties': len(specialty_dist),
                'distribution': specialty_dist.to_dict(),
                'most_common': specialty_dist.index[0],
                'least_common': specialty_dist.index[-1]
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
            # Load disease symptom data
            df = pd.read_csv(self.raw_data_path / 'disease_symptom.csv', sep=';')
            
            # Extract and process symptoms
            all_symptoms = []
            for symptom_list in df['Symptom']:
                try:
                    symptoms = eval(symptom_list)  # Convert string representation to list
                    all_symptoms.extend(symptoms)
                except:
                    continue
            
            # Count symptom frequencies
            symptom_counts = Counter(all_symptoms)
            
            metrics = {
                'total_unique_symptoms': len(symptom_counts),
                'most_common_symptoms': dict(symptom_counts.most_common(10)),
                'average_symptoms_per_disease': len(all_symptoms) / len(df)
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

    def analyze_conversation_data(self) -> Dict:
        """Analyze conversation patterns"""
        try:
            # Load conversation data
            with open(self.raw_data_path / 'alpaca_data.json', 'r') as f:
                alpaca_data = json.load(f)
            with open(self.raw_data_path / 'chatdoctor5k.json', 'r') as f:
                chatdoctor_data = json.load(f)
            
            # Analyze conversation lengths
            alpaca_lengths = [len(item['output'].split()) for item in alpaca_data]
            chatdoctor_lengths = [len(item['output'].split()) for item in chatdoctor_data]
            
            metrics = {
                'total_conversations': len(alpaca_data) + len(chatdoctor_data),
                'average_response_length': {
                    'alpaca': np.mean(alpaca_lengths),
                    'chatdoctor': np.mean(chatdoctor_lengths)
                },
                'response_length_distribution': {
                    'alpaca': {
                        'min': min(alpaca_lengths),
                        'max': max(alpaca_lengths),
                        'median': np.median(alpaca_lengths)
                    },
                    'chatdoctor': {
                        'min': min(chatdoctor_lengths),
                        'max': max(chatdoctor_lengths),
                        'median': np.median(chatdoctor_lengths)
                    }
                }
            }
            
            # Create visualization
            plt.figure(figsize=(12, 6))
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

    def analyze_disease_patterns(self) -> Dict:
        """Analyze disease patterns and relationships"""
        try:
            # Load disease data
            df = pd.read_csv(self.raw_data_path / 'disease_database_mini.csv', sep=';')
            
            # Analyze disease-specialty relationships
            disease_per_specialty = df.groupby('Medical Specialty')['Disease Name'].count()
            
            # Analyze disease-symptom relationships
            symptom_counts = df['Symptom'].apply(lambda x: len(eval(x)))
            
            metrics = {
                'total_diseases': len(df),
                'diseases_per_specialty': disease_per_specialty.to_dict(),
                'symptom_statistics': {
                    'average_symptoms': symptom_counts.mean(),
                    'min_symptoms': symptom_counts.min(),
                    'max_symptoms': symptom_counts.max()
                }
            }
            
            self.analysis_results['disease_analysis'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing disease patterns: {str(e)}")
            return {}

    def generate_report(self) -> None:
        """Generate comprehensive analysis report"""
        try:
            # Run all analyses
            self.analyze_medical_specialties()
            self.analyze_symptoms()
            self.analyze_conversation_data()
            self.analyze_disease_patterns()
            
            # Create report
            report = {
                'summary': {
                    'total_specialties': self.analysis_results['specialty_analysis']['total_specialties'],
                    'total_diseases': self.analysis_results['disease_analysis']['total_diseases'],
                    'total_conversations': self.analysis_results['conversation_analysis']['total_conversations'],
                    'total_symptoms': self.analysis_results['symptom_analysis']['total_unique_symptoms']
                },
                'detailed_analysis': self.analysis_results,
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

if __name__ == "__main__":
    analyzer = DataAnalyzer()
    analyzer.run_complete_analysis()