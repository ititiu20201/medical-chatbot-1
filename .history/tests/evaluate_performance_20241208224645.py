import json
import pandas as pd
from pathlib import Path
import logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceEvaluator:
    def __init__(self, test_results_path: str = "test_results"):
        """Initialize evaluator"""
        self.results_path = Path(test_results_path)
        self.results_path.mkdir(exist_ok=True)
        self.metrics = {
            "response_times": [],
            "specialty_predictions": [],
            "true_specialties": [],
            "error_cases": [],
            "conversation_completions": []
        }

    def evaluate_response_time(self, start_time: float, end_time: float) -> float:
        """Evaluate response time"""
        response_time = end_time - start_time
        self.metrics["response_times"].append(response_time)
        return response_time

    def evaluate_specialty_prediction(self, predicted: str, actual: str):
        """Evaluate specialty prediction accuracy"""
        self.metrics["specialty_predictions"].append(predicted)
        self.metrics["true_specialties"].append(actual)

    def log_error(self, error_type: str, details: dict):
        """Log error cases"""
        self.metrics["error_cases"].append({
            "type": error_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def evaluate_conversation(self, conversation_data: dict):
        """Evaluate full conversation flow"""
        self.metrics["conversation_completions"].append({
            "completed": conversation_data.get("state") == "completed",
            "steps_completed": len(conversation_data.get("collected_info", {})),
            "timestamp": datetime.now().isoformat()
        })

    def generate_report(self):
        """Generate evaluation report"""
        report = {
            "response_time_metrics": {
                "average": sum(self.metrics["response_times"]) / len(self.metrics["response_times"]),
                "max": max(self.metrics["response_times"]),
                "min": min(self.metrics["response_times"])
            },
            "specialty_prediction_metrics": classification_report(
                self.metrics["true_specialties"],
                self.metrics["specialty_predictions"],
                output_dict=True
            ),
            "error_analysis": {
                "total_errors": len(self.metrics["error_cases"]),
                "error_types": pd.Series([e["type"] for e in self.metrics["error_cases"]]).value_counts().to_dict()
            },
            "conversation_metrics": {
                "completion_rate": sum(1 for c in self.metrics["conversation_completions"] if c["completed"]) / len(self.metrics["conversation_completions"]),
                "average_steps": sum(c["steps_completed"] for c in self.metrics["conversation_completions"]) / len(self.metrics["conversation_completions"])
            }
        }

        # Save report
        report_path = self.results_path / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        # Generate visualizations
        self._generate_visualizations()

        return report

    def _generate_visualizations(self):
        """Generate evaluation visualizations"""
        # Response time distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.metrics["response_times"], bins=30)
        plt.title("Response Time Distribution")
        plt.xlabel("Response Time (seconds)")
        plt.ylabel("Frequency")
        plt.savefig(self.results_path / "response_times.png")
        plt.close()

        # Specialty prediction confusion matrix
        plt.figure(figsize=(12, 8))
        cm = confusion_matrix(
            self.metrics["true_specialties"],
            self.metrics["specialty_predictions"]
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=sorted(set(self.metrics["true_specialties"])),
            yticklabels=sorted(set(self.metrics["true_specialties"]))
        )
        plt.title("Specialty Prediction Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(self.results_path / "confusion_matrix.png", bbox_inches='tight')
        plt.close()

        # Error distribution
        error_types = pd.Series([e["type"] for e in self.metrics["error_cases"]])
        plt.figure(figsize=(10, 6))
        error_types.value_counts().plot(kind='bar')
        plt.title("Error Type Distribution")
        plt.xlabel("Error Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_path / "error_distribution.png")
        plt.close()

    def save_test_case(self, test_case: dict):
        """Save individual test case results"""
        case_path = self.results_path / "test_cases"
        case_path.mkdir(exist_ok=True)
        
        file_path = case_path / f"test_case_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_case, f, indent=4, ensure_ascii=False)

def run_evaluation():
    """Run complete evaluation"""
    evaluator = PerformanceEvaluator()
    
    # Test cases
    test_cases = [
        {
            "input": "Đau đầu, sốt",
            "expected_specialty": "Nội khoa"
        },
        {
            "input": "Đau răng",
            "expected_specialty": "Răng hàm mặt"
        }
        # Add more test cases
    ]
    
    # Run evaluations
    for case in test_cases:
        start_time = time.time()
        
        try:
            # Run chatbot prediction (replace with actual chatbot call)
            predicted_specialty = "Nội khoa"  # Mock prediction
            
            end_time = time.time()
            
            # Evaluate response time
            response_time = evaluator.evaluate_response_time(start_time, end_time)
            
            # Evaluate prediction
            evaluator.evaluate_specialty_prediction(
                predicted_specialty,
                case["expected_specialty"]
            )
            
            # Save test case result
            evaluator.save_test_case({
                "input": case["input"],
                "expected": case["expected_specialty"],
                "predicted": predicted_specialty,
                "response_time": response_time
            })
            
        except Exception as e:
            evaluator.log_error("prediction_error", {
                "input": case["input"],
                "error": str(e)
            })
    
    # Generate final report
    report = evaluator.generate_report()
    logger.info("Evaluation completed. Report generated.")
    return report

if __name__ == "__main__":
    run_evaluation()