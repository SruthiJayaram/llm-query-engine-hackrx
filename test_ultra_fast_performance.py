#!/usr/bin/env python3
"""
Performance Test Script for Ultra-Fast HackRx LLM API
Tests response time and accuracy for the optimization goals:
- Average response time < 30 seconds
- Accuracy > 50%
"""

import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor
import statistics

API_URL = "https://hackrx-llm-engine.onrender.com/hackrx/run"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer test-token"
}

# Test cases with expected answers for accuracy measurement
TEST_CASES = [
    {
        "name": "Text Content - Geography",
        "documents": "France is a country in Western Europe. Paris is the capital of France. The Eiffel Tower is a famous landmark in Paris. France is known for its cuisine, wine, and art. The population of France is approximately 67 million people.",
        "questions": [
            "What is the capital of France?",
            "What is France known for?", 
            "What is the population of France?"
        ],
        "expected_keywords": [
            ["paris", "capital"],
            ["cuisine", "wine", "art"],
            ["67", "million", "population"]
        ]
    },
    {
        "name": "Text Content - Technology",
        "documents": "Artificial Intelligence (AI) is transforming various industries. Machine learning algorithms can analyze large datasets quickly. Natural language processing enables computers to understand human language. Deep learning uses neural networks with multiple layers. AI applications include image recognition, speech processing, and autonomous vehicles.",
        "questions": [
            "What does AI transform?",
            "What enables computers to understand language?",
            "What are AI applications?"
        ],
        "expected_keywords": [
            ["industries", "transform"],
            ["natural", "language", "processing"],
            ["image", "speech", "autonomous", "vehicles"]
        ]
    },
    {
        "name": "Text Content - Science",
        "documents": "Photosynthesis is the process by which plants convert sunlight into energy. Chlorophyll in leaves absorbs light energy. Carbon dioxide from the air and water from the roots are used to produce glucose. Oxygen is released as a byproduct. This process is essential for life on Earth.",
        "questions": [
            "What is photosynthesis?",
            "What absorbs light energy?",
            "What is released as a byproduct?"
        ],
        "expected_keywords": [
            ["process", "sunlight", "energy"],
            ["chlorophyll", "absorbs", "light"],
            ["oxygen", "released", "byproduct"]
        ]
    }
]

def test_api_call(test_case):
    """Test a single API call and measure performance"""
    start_time = time.time()
    
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={
                "documents": test_case["documents"],
                "questions": test_case["questions"]
            },
            timeout=35  # 5 seconds buffer over our 30s target
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "name": test_case["name"],
                "success": True,
                "response_time": response_time,
                "answers": data.get("answers", []),
                "expected_keywords": test_case["expected_keywords"],
                "api_processing_time": data.get("processing_time", None)
            }
        else:
            return {
                "name": test_case["name"],
                "success": False,
                "response_time": response_time,
                "error": f"HTTP {response.status_code}: {response.text}",
                "answers": [],
                "expected_keywords": test_case["expected_keywords"]
            }
    except requests.exceptions.Timeout:
        return {
            "name": test_case["name"],
            "success": False,
            "response_time": 35.0,  # Timeout
            "error": "Request timeout (>30s)",
            "answers": [],
            "expected_keywords": test_case["expected_keywords"]
        }
    except Exception as e:
        return {
            "name": test_case["name"],
            "success": False,
            "response_time": time.time() - start_time,
            "error": str(e),
            "answers": [],
            "expected_keywords": test_case["expected_keywords"]
        }

def calculate_accuracy(answers, expected_keywords_list):
    """Calculate accuracy based on keyword matching"""
    if not answers or not expected_keywords_list:
        return 0.0
    
    correct_answers = 0
    
    for i, (answer, expected_keywords) in enumerate(zip(answers, expected_keywords_list)):
        if not answer:
            continue
            
        answer_lower = answer.lower()
        keyword_matches = sum(1 for keyword in expected_keywords 
                            if keyword.lower() in answer_lower)
        
        # Consider answer correct if it contains at least 50% of expected keywords
        if keyword_matches >= len(expected_keywords) * 0.5:
            correct_answers += 1
    
    return (correct_answers / len(answers)) * 100

def run_performance_test():
    """Run comprehensive performance testing"""
    print("🚀 Starting Ultra-Fast API Performance Test")
    print(f"📊 Target: <30s response time, >50% accuracy")
    print("=" * 60)
    
    # Test API availability first
    try:
        response = requests.get("https://hackrx-llm-engine.onrender.com/health", timeout=10)
        if response.status_code != 200:
            print("❌ API is not available. Exiting.")
            return
    except:
        print("❌ API is not responding. Exiting.")
        return
    
    results = []
    
    # Run tests
    print("🧪 Running test cases...")
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n📝 Test {i}/{len(TEST_CASES)}: {test_case['name']}")
        print(f"   Questions: {len(test_case['questions'])}")
        
        result = test_api_call(test_case)
        results.append(result)
        
        if result["success"]:
            accuracy = calculate_accuracy(result["answers"], result["expected_keywords"])
            print(f"   ✅ Success in {result['response_time']:.2f}s")
            print(f"   🎯 Accuracy: {accuracy:.1f}%")
            if result.get("api_processing_time"):
                print(f"   ⚡ API Processing: {result['api_processing_time']:.2f}s")
            
            # Show answers
            for j, (question, answer) in enumerate(zip(test_case["questions"], result["answers"])):
                print(f"   Q{j+1}: {question}")
                print(f"   A{j+1}: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        else:
            print(f"   ❌ Failed: {result['error']}")
            print(f"   ⏱️  Time: {result['response_time']:.2f}s")
    
    # Calculate overall statistics
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    successful_tests = [r for r in results if r["success"]]
    
    if successful_tests:
        response_times = [r["response_time"] for r in successful_tests]
        accuracies = [calculate_accuracy(r["answers"], r["expected_keywords"]) 
                     for r in successful_tests]
        
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        avg_accuracy = statistics.mean(accuracies)
        
        print(f"✅ Successful Tests: {len(successful_tests)}/{len(results)}")
        print(f"⏱️  Average Response Time: {avg_response_time:.2f}s")
        print(f"⏱️  Min/Max Response Time: {min_response_time:.2f}s / {max_response_time:.2f}s")
        print(f"🎯 Average Accuracy: {avg_accuracy:.1f}%")
        
        # Check if goals are met
        speed_goal_met = avg_response_time < 30
        accuracy_goal_met = avg_accuracy > 50
        
        print(f"\n🏆 GOAL ASSESSMENT:")
        print(f"   Speed Goal (<30s): {'✅ PASSED' if speed_goal_met else '❌ FAILED'}")
        print(f"   Accuracy Goal (>50%): {'✅ PASSED' if accuracy_goal_met else '❌ FAILED'}")
        
        if speed_goal_met and accuracy_goal_met:
            print(f"\n🎉 ALL GOALS MET! Ready for HackRx submission!")
        else:
            print(f"\n⚠️  Some goals not met. Need further optimization.")
        
    else:
        print("❌ All tests failed. Check API status.")
    
    # Failed tests summary
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        print(f"\n❌ Failed Tests ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"   • {test['name']}: {test['error']}")

if __name__ == "__main__":
    run_performance_test()
