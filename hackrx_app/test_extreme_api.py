#!/usr/bin/env python3
"""
EXTREME API Performance Test
Testing the actual API endpoint for HYPER performance
"""

import requests
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import sys

API_BASE = "http://localhost:8000"
BEARER_TOKEN = "test-token-hyper-fast"

# Test cases for EXTREME performance testing
EXTREME_TESTS = [
    {
        "name": "EXTREME Speed 1 - Lightning Fast",
        "documents": "AI is transforming business operations worldwide.",
        "questions": ["What is AI doing?"]
    },
    {
        "name": "EXTREME Speed 2 - Parallel Test",
        "documents": "Cloud computing enables scalable infrastructure for modern applications.",
        "questions": ["What does cloud computing enable?", "What type of infrastructure?"]
    },
    {
        "name": "EXTREME Speed 3 - Multiple Rapid",
        "documents": "Cybersecurity threats are evolving rapidly. Organizations need robust defense mechanisms.",
        "questions": ["How are threats evolving?", "What do organizations need?", "Why is this important?"]
    },
    {
        "name": "EXTREME Speed 4 - PDF Test",
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": ["What type of document is this?"]
    }
]

def test_api_endpoint(test_data):
    """Test a single API call"""
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE}/hackrx/run",
            headers={
                "Authorization": f"Bearer {BEARER_TOKEN}",
                "Content-Type": "application/json"
            },
            json=test_data,
            timeout=15  # Aggressive timeout
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "response_time": response_time,
                "answers": data.get("answers", []),
                "processing_time": data.get("processing_time", 0),
                "status": data.get("status", "unknown")
            }
        else:
            return {
                "success": False,
                "response_time": response_time,
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }
    
    except Exception as e:
        response_time = time.time() - start_time
        return {
            "success": False,
            "response_time": response_time,
            "error": str(e)
        }

def run_extreme_api_test():
    print("🔥 EXTREME API Performance Test")
    print("=" * 50)
    print("Testing actual API endpoints for HYPER speed")
    print()
    
    # First check if API is running
    try:
        health_response = requests.get(f"{API_BASE}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ API is not running! Start the server first:")
            print("   cd hackrx_app && python main_hyper_fast.py")
            return False
        print("✅ API is running and healthy")
    except:
        print("❌ Cannot connect to API! Start the server first:")
        print("   cd hackrx_app && python main_hyper_fast.py")
        return False
    
    print()
    
    all_times = []
    successful_tests = 0
    
    for i, test in enumerate(EXTREME_TESTS, 1):
        print(f"🔥 Running {test['name']}...")
        
        test_data = {
            "documents": test["documents"],
            "questions": test["questions"]
        }
        
        result = test_api_endpoint(test_data)
        
        if result["success"]:
            all_times.append(result["response_time"])
            successful_tests += 1
            
            print(f"  ✅ SUCCESS! Response: {result['response_time']:.3f}s")
            print(f"     Processing: {result['processing_time']:.3f}s")
            print(f"     Status: {result['status']}")
            print(f"     Answers: {len(result['answers'])}")
            
            # Show sample answers
            for j, answer in enumerate(result["answers"][:2]):  # Show first 2
                print(f"     A{j+1}: {answer[:60]}...")
            
            # Performance rating
            if result["response_time"] < 2:
                print("     🚀 EXTREME SPEED! < 2s")
            elif result["response_time"] < 5:
                print("     ⚡ HYPER FAST! < 5s")
            elif result["response_time"] < 10:
                print("     🎯 TARGET MET! < 10s")
            else:
                print("     ⚠️  Above target")
        else:
            print(f"  ❌ FAILED: {result['error']}")
            all_times.append(30.0)  # Penalty
        
        print()
    
    # Final results
    if successful_tests > 0:
        avg_time = sum(all_times) / len(all_times)
        fastest = min(all_times)
        slowest = max(all_times)
        
        print("🏆 EXTREME API PERFORMANCE RESULTS")
        print("=" * 50)
        print(f"Successful tests: {successful_tests}/{len(EXTREME_TESTS)}")
        print(f"Average response time: {avg_time:.3f}s")
        print(f"Fastest response: {fastest:.3f}s")
        print(f"Slowest response: {slowest:.3f}s")
        print()
        
        # Performance classification
        if avg_time < 1:
            print("🚀 EXTREME PERFORMANCE! Sub-second average!")
        elif avg_time < 3:
            print("⚡ HYPER PERFORMANCE! Average < 3s")
        elif avg_time < 10:
            print("🎯 EXCELLENT! Target achieved < 10s")
        else:
            print("⚠️  Above target but functional")
        
        # Speed breakdown
        under_1s = sum(1 for t in all_times if t < 1)
        under_3s = sum(1 for t in all_times if t < 3)
        under_10s = sum(1 for t in all_times if t < 10)
        
        print(f"Under 1s: {under_1s}/{len(all_times)} ({under_1s/len(all_times)*100:.1f}%)")
        print(f"Under 3s: {under_3s}/{len(all_times)} ({under_3s/len(all_times)*100:.1f}%)")
        print(f"Under 10s: {under_10s}/{len(all_times)} ({under_10s/len(all_times)*100:.1f}%)")
        
        return avg_time < 10
    else:
        print("❌ All API tests failed!")
        return False

def stress_test_parallel():
    """Run parallel stress test"""
    print("\n🔥 EXTREME PARALLEL STRESS TEST")
    print("=" * 50)
    
    stress_data = {
        "documents": "Machine learning algorithms process data to find patterns and make predictions.",
        "questions": ["What do ML algorithms do?", "How do they work?"]
    }
    
    def single_stress_test():
        return test_api_endpoint(stress_data)
    
    # Run 5 parallel requests
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(single_stress_test) for _ in range(5)]
        results = [f.result() for f in futures]
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    
    print(f"Parallel requests: 5")
    print(f"Successful: {successful}/5")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average per request: {total_time/5:.3f}s")
    
    if successful >= 4:
        print("✅ PARALLEL STRESS TEST PASSED!")
        return True
    else:
        print("⚠️  Parallel stress test had issues")
        return False

if __name__ == "__main__":
    print("Starting EXTREME API testing...")
    print("Make sure the API server is running!")
    print()
    
    success = run_extreme_api_test()
    
    if success:
        stress_success = stress_test_parallel()
        if stress_success:
            print("\n🏆 ALL EXTREME TESTS PASSED!")
            print("🚀 READY FOR HACKRX DOMINATION!")
        else:
            print("\n⚡ Primary tests passed, stress test needs work")
    else:
        print("\n⚠️  API tests need optimization")
    
    sys.exit(0 if success else 1)
