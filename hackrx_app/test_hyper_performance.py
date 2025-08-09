#!/usr/bin/env python3
"""
HYPER-AGGRESSIVE Performance Test - Version 4.0
Target: <10 second average response time
"""

import time
import sys
import os
sys.path.append('/workspaces/llm-query-engine-hackrx/hackrx_app')

from utils_hyper_fast import (
    process_document_input_hyper_fast,
    process_questions_hyper_parallel,
    cleanup_cache_hyper_aggressive
)

# Test cases optimized for speed
HYPER_FAST_TESTS = [
    {
        "name": "Speed Test 1 - Short Document",
        "document": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
        "questions": ["What is machine learning?", "How does it relate to AI?"]
    },
    {
        "name": "Speed Test 2 - Technical Document",
        "document": "FastAPI is a modern web framework for building APIs with Python. It provides automatic API documentation, type checking, and high performance through async support.",
        "questions": ["What is FastAPI?", "What are its key features?"]
    },
    {
        "name": "Speed Test 3 - Business Document",
        "document": "The company's Q3 revenue increased by 25% compared to the previous quarter. This growth was driven by strong performance in the cloud services division and new product launches.",
        "questions": ["What was the Q3 revenue growth?", "What drove this growth?"]
    },
    {
        "name": "Speed Test 4 - Multiple Questions",
        "document": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is natural, human activities have accelerated these changes through greenhouse gas emissions.",
        "questions": ["What is climate change?", "What causes it?", "Is it natural?", "How do humans affect it?"]
    },
    {
        "name": "Speed Test 5 - Complex Document",
        "document": "Blockchain technology is a distributed ledger system that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data.",
        "questions": ["What is blockchain?", "How are blocks secured?", "What does each block contain?"]
    }
]

def run_hyper_performance_test():
    print("🚀 HYPER-AGGRESSIVE Performance Test v4.0")
    print("=" * 50)
    print("Target: <10 second average response time")
    print("Accuracy Target: >60%")
    print()
    
    total_time = 0
    successful_tests = 0
    all_times = []
    
    for i, test in enumerate(HYPER_FAST_TESTS, 1):
        print(f"⚡ Running {test['name']}...")
        start_time = time.time()
        
        try:
            # Process document
            doc_start = time.time()
            document_text = process_document_input_hyper_fast(test["document"])
            doc_time = time.time() - doc_start
            
            # Process questions
            qa_start = time.time()
            answers = process_questions_hyper_parallel(document_text, test["questions"])
            qa_time = time.time() - qa_start
            
            total_test_time = time.time() - start_time
            all_times.append(total_test_time)
            total_time += total_test_time
            successful_tests += 1
            
            print(f"  ✅ Completed in {total_test_time:.3f}s")
            print(f"     📄 Document processing: {doc_time:.3f}s")
            print(f"     ❓ Questions processing: {qa_time:.3f}s")
            print(f"     📝 Answers generated: {len(answers)}")
            
            # Show sample answers
            for j, (question, answer) in enumerate(zip(test["questions"], answers)):
                print(f"     Q{j+1}: {question[:50]}...")
                print(f"     A{j+1}: {answer[:80]}...")
            
            # Performance check
            if total_test_time < 10:
                print(f"     🎯 HYPER TARGET MET! ({total_test_time:.3f}s < 10s)")
            elif total_test_time < 15:
                print(f"     ⚡ FAST! ({total_test_time:.3f}s < 15s)")
            else:
                print(f"     ⚠️  Slower than target ({total_test_time:.3f}s)")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            all_times.append(30.0)  # Penalty time
            total_time += 30.0
        
        print()
        
        # Cleanup between tests
        if i % 2 == 0:
            cleanup_cache_hyper_aggressive()
    
    # Final statistics
    if successful_tests > 0:
        average_time = total_time / len(HYPER_FAST_TESTS)
        fastest_time = min(all_times)
        slowest_time = max(all_times)
        
        print("🏆 HYPER-AGGRESSIVE PERFORMANCE RESULTS")
        print("=" * 50)
        print(f"Total tests: {len(HYPER_FAST_TESTS)}")
        print(f"Successful: {successful_tests}")
        print(f"Average time: {average_time:.3f}s")
        print(f"Fastest time: {fastest_time:.3f}s")
        print(f"Slowest time: {slowest_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
        print()
        
        # Performance evaluation
        if average_time < 5:
            print("🚀 HYPER-SONIC PERFORMANCE! Average < 5s")
        elif average_time < 10:
            print("⚡ HYPER TARGET ACHIEVED! Average < 10s")
        elif average_time < 20:
            print("🎯 ULTRA-FAST! Average < 20s")
        elif average_time < 30:
            print("✅ TARGET MET! Average < 30s")
        else:
            print("⚠️  Above target, but still functional")
        
        print()
        print(f"Performance ratio: {10.0/average_time:.2f}x target")
        
        # Speed distribution
        under_5s = sum(1 for t in all_times if t < 5)
        under_10s = sum(1 for t in all_times if t < 10)
        under_20s = sum(1 for t in all_times if t < 20)
        under_30s = sum(1 for t in all_times if t < 30)
        
        print(f"Tests under 5s: {under_5s}/{len(all_times)} ({under_5s/len(all_times)*100:.1f}%)")
        print(f"Tests under 10s: {under_10s}/{len(all_times)} ({under_10s/len(all_times)*100:.1f}%)")
        print(f"Tests under 20s: {under_20s}/{len(all_times)} ({under_20s/len(all_times)*100:.1f}%)")
        print(f"Tests under 30s: {under_30s}/{len(all_times)} ({under_30s/len(all_times)*100:.1f}%)")
        
        return average_time < 10
    else:
        print("❌ All tests failed!")
        return False

if __name__ == "__main__":
    success = run_hyper_performance_test()
    if success:
        print("\n🏆 HYPER-AGGRESSIVE OPTIMIZATION SUCCESS!")
        print("Ready for HackRx submission with <10s response time!")
    else:
        print("\n⚠️  Performance needs more optimization")
    
    sys.exit(0 if success else 1)
