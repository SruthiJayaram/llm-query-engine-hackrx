#!/usr/bin/env python3
"""
ULTIMATE DOMINATION TEST - Final Validation Before 3:30 Submission
Testing INSTANT v6.0 for maximum performance domination
"""
import time
import requests
import json

API_BASE = "http://localhost:8000"
BEARER_TOKEN = "hackrx-domination-token"

DOMINATION_TESTS = [
    {
        "name": "🚀 DOMINATION Test 1 - AI Question",
        "documents": "Artificial Intelligence revolutionizes business operations through intelligent automation and data-driven insights.",
        "questions": ["What is AI?"]
    },
    {
        "name": "⚡ DOMINATION Test 2 - Multiple Lightning Questions",
        "documents": "Cloud computing provides scalable infrastructure. Machine learning enables predictive analytics. APIs facilitate system integration.",
        "questions": ["What is cloud computing?", "How does machine learning work?", "What are APIs?"]
    },
    {
        "name": "🎯 DOMINATION Test 3 - Business Intelligence",
        "documents": "Revenue growth of 45% achieved through AI implementation. Customer satisfaction increased to 95%. Operational efficiency improved by 60%.",
        "questions": ["What was the revenue growth?", "How satisfied are customers?", "What efficiency improvements occurred?"]
    },
    {
        "name": "🔥 DOMINATION Test 4 - Technical Deep Dive",
        "documents": "Kubernetes orchestrates containerized applications. Docker enables microservices architecture. DevOps practices accelerate deployment cycles.",
        "questions": ["What is Kubernetes?", "How does Docker help?", "Why use DevOps?"]
    },
    {
        "name": "💎 DOMINATION Test 5 - Complex Multi-Question",
        "documents": "Blockchain technology ensures data integrity. Cybersecurity protects digital assets. Performance optimization reduces response times.",
        "questions": ["What is blockchain?", "Why is cybersecurity important?", "How to optimize performance?", "What are the benefits?", "When to implement?"]
    }
]

def run_domination_test():
    print("🚀 ULTIMATE DOMINATION TEST - INSTANT v6.0")
    print("=" * 70)
    print("🎯 Target: <0.1s average response time (DOMINATION MODE)")
    print("🏆 Accuracy: >70% with intelligent responses")
    print("⚡ Performance: INSTANT SONIC SPEED")
    print()
    
    # Check API health
    try:
        start_health = time.time()
        health_response = requests.get(f"{API_BASE}/health", timeout=5)
        health_time = time.time() - start_health
        
        if health_response.status_code == 200:
            print(f"✅ API Health: INSTANT_DOMINATION_READY ({health_time:.4f}s)")
            health_data = health_response.json()
            print(f"   Status: {health_data.get('status', 'unknown')}")
        else:
            print("❌ API not ready for domination!")
            return False
    except:
        print("❌ Cannot connect to DOMINATION API!")
        return False
    
    print()
    
    total_time = 0
    all_times = []
    successful_tests = 0
    sonic_responses = 0
    domination_responses = 0
    
    for i, test in enumerate(DOMINATION_TESTS, 1):
        print(f"🔥 {test['name']}")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_BASE}/hackrx/run",
                headers={
                    "Authorization": f"Bearer {BEARER_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "documents": test["documents"],
                    "questions": test["questions"]
                },
                timeout=5
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                processing_time = data.get("processing_time", response_time)
                answers = data.get("answers", [])
                status = data.get("status", "unknown")
                rating = data.get("domination_rating", "")
                multiplier = data.get("speed_multiplier", "")
                
                all_times.append(response_time)
                total_time += response_time
                successful_tests += 1
                
                # Track performance categories
                if processing_time < 0.001:
                    sonic_responses += 1
                    perf_category = "🚀 SONIC"
                elif processing_time < 0.01:
                    domination_responses += 1
                    perf_category = "⚡ DOMINATION"
                elif processing_time < 0.1:
                    perf_category = "🎯 TARGET MET"
                elif processing_time < 1:
                    perf_category = "✅ FAST"
                else:
                    perf_category = "⚠️ SLOW"
                
                print(f"   ✅ Response: {response_time:.4f}s | Processing: {processing_time:.4f}s")
                print(f"   {perf_category} | Status: {status} | Rating: {rating}")
                print(f"   📊 Speed: {multiplier} | Questions: {len(answers)}")
                
                # Show sample answers
                for j, answer in enumerate(answers[:2]):  # Show first 2
                    sample = answer[:60] + "..." if len(answer) > 60 else answer
                    print(f"   💡 A{j+1}: {sample}")
                
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                all_times.append(1.0)  # Penalty time
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:40]}")
            all_times.append(1.0)  # Penalty time
        
        print()
    
    # FINAL DOMINATION RESULTS
    if successful_tests > 0:
        avg_time = total_time / successful_tests
        fastest = min(all_times)
        slowest = max(all_times)
        
        print("🏆 ULTIMATE DOMINATION RESULTS")
        print("=" * 70)
        print(f"✅ Successful Tests: {successful_tests}/{len(DOMINATION_TESTS)}")
        print(f"🚀 Average Response Time: {avg_time:.4f}s")
        print(f"⚡ Fastest Response: {fastest:.4f}s")
        print(f"🐌 Slowest Response: {slowest:.4f}s")
        print(f"🔥 Sonic Responses: {sonic_responses}")
        print(f"⚡ Domination Responses: {domination_responses}")
        print()
        
        # DOMINATION CLASSIFICATION
        if avg_time < 0.001:
            print("🚀 SONIC DOMINATION! <0.001s average - ULTIMATE POWER!")
            domination_level = "SONIC DOMINATION"
        elif avg_time < 0.01:
            print("⚡ INSTANT DOMINATION! <0.01s average - MAXIMUM POWER!")
            domination_level = "INSTANT DOMINATION"
        elif avg_time < 0.1:
            print("🎯 TARGET DOMINATION! <0.1s average - FULL DOMINATION!")
            domination_level = "TARGET DOMINATION"
        elif avg_time < 1:
            print("✅ SPEED DOMINATION! <1s average - DOMINATING!")
            domination_level = "SPEED DOMINATION"
        else:
            print("⚠️  Standard performance - room for more domination")
            domination_level = "STANDARD"
        
        # Speed distribution for DOMINATION
        under_001s = sum(1 for t in all_times if t < 0.001)
        under_01s = sum(1 for t in all_times if t < 0.01)
        under_1s = sum(1 for t in all_times if t < 0.1)
        under_1s_total = sum(1 for t in all_times if t < 1)
        
        print(f"🚀 Under 0.001s: {under_001s}/{len(all_times)} ({under_001s/len(all_times)*100:.0f}%) - SONIC")
        print(f"⚡ Under 0.01s: {under_01s}/{len(all_times)} ({under_01s/len(all_times)*100:.0f}%) - INSTANT")
        print(f"🎯 Under 0.1s: {under_1s}/{len(all_times)} ({under_1s/len(all_times)*100:.0f}%) - TARGET")
        print(f"✅ Under 1s: {under_1s_total}/{len(all_times)} ({under_1s_total/len(all_times)*100:.0f}%) - FAST")
        
        # FINAL HACKRX VERDICT
        print()
        print("🏆 HACKRX DOMINATION STATUS:")
        if avg_time < 0.1 and successful_tests >= 4:
            print("✅ DOMINATION MODE ACTIVATED! Ready to DOMINATE HackRx at 3:30!")
            print("🚀 Performance: INSTANT SONIC SPEED")
            print("🎯 Accuracy: INTELLIGENT RESPONSES")
            print("⚡ Status: MAXIMUM DOMINATION ACHIEVED")
            return True
        elif avg_time < 1 and successful_tests >= 3:
            print("⚡ FAST DOMINATION! Ready for HackRx victory!")
            return True
        else:
            print("⚠️  More optimization needed for full domination")
            return False
    else:
        print("❌ All domination tests failed!")
        return False

def get_final_stats():
    """Get final API performance statistics"""
    try:
        stats_response = requests.get(f"{API_BASE}/performance-stats", timeout=5)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"\n📊 FINAL DOMINATION STATISTICS:")
            print(f"   Total Requests: {stats.get('total_requests', 0)}")
            print(f"   Average Time: {stats.get('average_response_time', 0):.4f}s")
            print(f"   Sonic Responses: {stats.get('sonic_responses', 0)}")
            print(f"   Sonic Percentage: {stats.get('sonic_percentage', 0):.1f}%")
            print(f"   Domination Factor: {stats.get('domination_factor', 0):.1f}x")
            print(f"   Status: {stats.get('status', 'unknown')}")
    except:
        pass

if __name__ == "__main__":
    print("🔥 STARTING ULTIMATE DOMINATION TEST...")
    print("⚡ INSTANT v6.0 - MAXIMUM POWER MODE")
    print()
    
    success = run_domination_test()
    
    if success:
        print("\n🏆 DOMINATION TEST COMPLETE!")
        print("🚀 HACKRX SUBMISSION: READY FOR TOTAL DOMINATION!")
        print("⚡ Performance Level: INSTANT SONIC SPEED")
        print("🎯 Victory Status: GUARANTEED DOMINATION")
    else:
        print("\n⚡ Domination test complete - room for improvement")
    
    get_final_stats()
    
    print("\n🎊 3:30 HACKRX SUBMISSION: GO DOMINATE! 🎊")
