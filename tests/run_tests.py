import pytest
import subprocess
import logging
from pathlib import Path
import json
from datetime import datetime
from evaluate_performance import run_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_all_tests():
    """Run all tests and evaluations"""
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "unit_tests": {},
        "integration_tests": {},
        "performance_tests": {},
        "frontend_tests": {}
    }

    try:
        # Run unit tests
        logger.info("Running unit tests...")
        unit_result = pytest.main(["tests/test_chatbot.py", "-v"])
        results["unit_tests"]["status"] = "passed" if unit_result == 0 else "failed"
        
        # Run API tests
        logger.info("Running API tests...")
        api_result = pytest.main(["tests/test_api.py", "-v"])
        results["integration_tests"]["api"] = "passed" if api_result == 0 else "failed"
        
        # Run frontend tests
        logger.info("Running frontend tests...")
        frontend_result = pytest.main(["tests/test_frontend.py", "-v"])
        results["frontend_tests"]["status"] = "passed" if frontend_result == 0 else "failed"
        
        # Run performance evaluation
        logger.info("Running performance evaluation...")
        performance_results = run_evaluation()
        results["performance_tests"] = performance_results

    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        results["error"] = str(e)

    # Save results
    results_file = results_dir / f"test_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return results

def main():
    """Main function to run tests"""
    # Setup test environment
    subprocess.run(["playwright", "install"])
    
    # Run tests
    results = run_all_tests()
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info(f"Unit Tests: {results['unit_tests'].get('status', 'error')}")
    logger.info(f"API Tests: {results['integration_tests'].get('api', 'error')}")
    logger.info(f"Frontend Tests: {results['frontend_tests'].get('status', 'error')}")
    
    if "performance_tests" in results:
        perf = results["performance_tests"]
        logger.info("\nPerformance Metrics:")
        logger.info(f"Average Response Time: {perf['response_time_metrics']['average']:.2f}s")
        logger.info(f"Specialty Prediction Accuracy: {perf['specialty_prediction_metrics']['accuracy']:.2%}")
        logger.info(f"Conversation Completion Rate: {perf['conversation_metrics']['completion_rate']:.2%}")

if __name__ == "__main__":
    main()