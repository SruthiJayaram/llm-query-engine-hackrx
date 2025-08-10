#!/usr/bin/env python3
"""
HackRx Competition Final Validation Test
Tests all major insurance policy questions to ensure 100% accuracy
"""

import requests
import json
import time
from typing import List, Dict, Any

# Competition endpoint and auth
API_URL = "https://hackrx-llm-engine.onrender.com/hackrx/run"
AUTH_TOKEN = "d66bf9184ca9c85d9572b80ca40659dc122772c748e81bbbb1c2ec5ff0d87d42"
PDF_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# All key competition questions based on common insurance policy topics
COMPETITION_QUESTIONS = [
    {
        "question": "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "expected_keywords": ["grace period", "thirty days", "premium payment", "due date"]
    },
    {
        "question": "What is the waiting period for maternity coverage?",
        "expected_keywords": ["maternity", "24 months", "continuously covered", "deliveries"]
    },
    {
        "question": "What is the maximum sum insured per individual under the National Parivar Mediclaim Plus Policy?",
        "expected_keywords": ["maximum", "sum insured", "plan", "individual requirements"]
    },
    {
        "question": "What benefits are covered for organ donor expenses?",
        "expected_keywords": ["organ donor", "medical expenses", "harvesting", "Transplantation of Human Organs Act"]
    },
    {
        "question": "What is the No Claim Discount benefit?",
        "expected_keywords": ["No Claim Discount", "5%", "base premium", "renewal", "no claims"]
    },
    {
        "question": "What is the extent of coverage for AYUSH treatment?",
        "expected_keywords": ["AYUSH", "Ayurveda", "Yoga", "Naturopathy", "inpatient treatment"]
    },
    {
        "question": "How is a hospital defined under this policy?",
        "expected_keywords": ["hospital", "inpatient beds", "10", "15", "nursing staff", "operation theatre"]
    },
    {
        "question": "What are the room rent sub-limits for Plan A?",
        "expected_keywords": ["room rent", "1%", "ICU", "2%", "Sum Insured", "Plan A"]
    },
    {
        "question": "What is the waiting period for pre-existing diseases?",
        "expected_keywords": ["waiting period", "36 months", "pre-existing diseases", "continuous coverage"]
    },
    {
        "question": "Are health check-up benefits covered?",
        "expected_keywords": ["health check-up", "two continuous policy years", "renewed", "Table of Benefits"]
    }
]

def test_question(question: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """Test a single question and validate the response"""
    print(f"\n🔍 Testing: {question}")
    
    payload = {
        "documents": PDF_URL,
        "questions": [question]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answers", [""])[0]
            
            # Check if expected keywords are present
            keywords_found = []
            keywords_missing = []
            
            answer_lower = answer.lower()
            for keyword in expected_keywords:
                if keyword.lower() in answer_lower:
                    keywords_found.append(keyword)
                else:
                    keywords_missing.append(keyword)
            
            success = len(keywords_missing) == 0
            
            result = {
                "question": question,
                "answer": answer,
                "processing_time": processing_time,
                "api_processing_time": data.get("processing_time", 0),
                "keywords_found": keywords_found,
                "keywords_missing": keywords_missing,
                "success": success,
                "status_code": response.status_code
            }
            
            if success:
                print(f"✅ SUCCESS - All keywords found")
                print(f"📝 Answer: {answer[:100]}...")
            else:
                print(f"❌ MISSING KEYWORDS: {keywords_missing}")
                print(f"📝 Answer: {answer}")
            
            return result
            
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return {
                "question": question,
                "success": False,
                "error": f"HTTP {response.status_code}",
                "response_text": response.text
            }
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return {
            "question": question,
            "success": False,
            "error": str(e)
        }

def run_full_competition_test():
    """Run comprehensive test of all competition questions"""
    print("🚀 HACKRX COMPETITION VALIDATION TEST")
    print("=" * 50)
    
    results = []
    total_questions = len(COMPETITION_QUESTIONS)
    successful_questions = 0
    total_processing_time = 0
    
    for i, test_case in enumerate(COMPETITION_QUESTIONS, 1):
        print(f"\n📋 Question {i}/{total_questions}")
        result = test_question(test_case["question"], test_case["expected_keywords"])
        results.append(result)
        
        if result.get("success", False):
            successful_questions += 1
            total_processing_time += result.get("processing_time", 0)
        
        # Small delay between requests
        time.sleep(1)
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 FINAL VALIDATION SUMMARY")
    print("=" * 50)
    print(f"✅ Successful Questions: {successful_questions}/{total_questions}")
    print(f"🎯 Success Rate: {(successful_questions/total_questions)*100:.1f}%")
    print(f"⏱️  Average Processing Time: {total_processing_time/successful_questions:.2f}s" if successful_questions > 0 else "⏱️  No successful requests")
    
    if successful_questions == total_questions:
        print("\n🎉 ALL TESTS PASSED! READY FOR COMPETITION SUBMISSION! 🎉")
    else:
        print(f"\n⚠️  {total_questions - successful_questions} questions need attention")
    
    # Detailed results
    print("\n📋 DETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        status = "✅" if result.get("success", False) else "❌"
        question = result.get("question", "Unknown")[:60]
        print(f"{status} Q{i}: {question}...")
        
        if not result.get("success", False):
            if "keywords_missing" in result:
                print(f"    Missing: {result['keywords_missing']}")
            if "error" in result:
                print(f"    Error: {result['error']}")
    
    return results

if __name__ == "__main__":
    results = run_full_competition_test()
    
    # Save results to file
    with open("competition_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: competition_validation_results.json")
