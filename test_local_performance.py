"""
Ultra-Fast Test Script - Local Performance Testing
This tests the optimization locally to ensure we meet the targets
"""

import time
import statistics
from hackrx_app.utils_ultra_fast import process_questions_parallel, cleanup_cache

# Multiple test scenarios
TEST_SCENARIOS = [
    {
        "name": "Geography Test",
        "document": "France is a country in Western Europe. Paris is the capital of France. The Eiffel Tower is located in Paris. France is known for its cuisine, wine, and culture. The French population is about 67 million people. The country covers an area of 643,801 square kilometers.",
        "questions": [
            "What is the capital of France?",
            "What is France known for?",
            "What is the population of France?"
        ]
    },
    {
        "name": "Technology Test", 
        "document": "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming. Deep learning uses neural networks with multiple layers. Natural language processing helps computers understand human language. Computer vision allows machines to interpret visual information from images and videos.",
        "questions": [
            "What is machine learning?",
            "What does deep learning use?",
            "What does computer vision do?"
        ]
    },
    {
        "name": "Science Test",
        "document": "Photosynthesis is the process plants use to convert sunlight into energy. Chlorophyll absorbs light energy. Carbon dioxide and water are converted into glucose and oxygen. This process is essential for life on Earth as it produces oxygen and removes carbon dioxide from the atmosphere.",
        "questions": [
            "What is photosynthesis?",
            "What does chlorophyll do?",
            "Why is photosynthesis essential?"
        ]
    }
]

def run_local_performance_test():
    print("🚀 ULTRA-FAST LOCAL PERFORMANCE TEST")
    print("=" * 50)
    print("Target: <30 seconds average response time")
    print("Target: >50% accuracy (qualitative assessment)")
    print("=" * 50)
    
    all_times = []
    
    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"\n🧪 Test {i}: {scenario['name']}")
        print(f"📝 Document: {len(scenario['document'])} chars")
        print(f"❓ Questions: {len(scenario['questions'])}")
        
        # Run test multiple times for average
        times = []
        for run in range(3):
            start_time = time.time()
            answers = process_questions_parallel(
                scenario['document'], 
                scenario['questions'], 
                max_workers=3
            )
            processing_time = time.time() - start_time
            times.append(processing_time)
            
            if run == 0:  # Show results from first run
                print(f"\n📊 Results (Run {run + 1}):")
                for j, (q, a) in enumerate(zip(scenario['questions'], answers)):
                    print(f"  Q{j+1}: {q}")
                    print(f"  A{j+1}: {a[:80]}{'...' if len(a) > 80 else ''}")
                print()
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        all_times.extend(times)
        
        print(f"⏱️  Average Time: {avg_time:.2f}s")
        print(f"⏱️  Min/Max: {min_time:.2f}s / {max_time:.2f}s")
        print(f"🎯 Speed Goal: {'✅ PASSED' if avg_time < 30 else '❌ FAILED'}")
        
        # Clean cache between tests
        cleanup_cache()
    
    # Overall statistics
    print("\n" + "=" * 50)
    print("📈 OVERALL PERFORMANCE SUMMARY")
    print("=" * 50)
    
    overall_avg = statistics.mean(all_times)
    overall_min = min(all_times)
    overall_max = max(all_times)
    
    print(f"📊 Total Tests: {len(all_times)}")
    print(f"⏱️  Average Response Time: {overall_avg:.2f} seconds")
    print(f"⏱️  Min Response Time: {overall_min:.2f} seconds") 
    print(f"⏱️  Max Response Time: {overall_max:.2f} seconds")
    print(f"📈 Tests under 10s: {sum(1 for t in all_times if t < 10)}/{len(all_times)}")
    print(f"📈 Tests under 20s: {sum(1 for t in all_times if t < 20)}/{len(all_times)}")
    print(f"📈 Tests under 30s: {sum(1 for t in all_times if t < 30)}/{len(all_times)}")
    
    # Final assessment
    speed_goal_met = overall_avg < 30
    print(f"\n🏆 FINAL ASSESSMENT:")
    print(f"   Speed Goal (<30s): {'✅ PASSED' if speed_goal_met else '❌ FAILED'}")
    
    if speed_goal_met:
        print(f"\n🎉 OPTIMIZATION SUCCESSFUL!")
        print(f"   Ready for HackRx submission at 3:45!")
        print(f"   Expected API performance: ~{overall_avg:.1f}s per request")
    else:
        print(f"\n⚠️  NEEDS MORE OPTIMIZATION")
        print(f"   Current avg: {overall_avg:.2f}s")
        print(f"   Need to improve by: {overall_avg - 30:.2f}s")

if __name__ == "__main__":
    run_local_performance_test()
