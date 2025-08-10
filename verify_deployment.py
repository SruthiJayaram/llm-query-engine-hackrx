#!/usr/bin/env python3
"""
LIGHTNING ACCURACY v18.0 DEPLOYMENT VERIFICATION
Tests the deployed system to verify it's working correctly
"""

import requests
import time
import json

def test_deployed_system():
    """Test the deployed Lightning Accuracy v18.0 system"""
    print("🔍 LIGHTNING ACCURACY v18.0 DEPLOYMENT VERIFICATION")
    print("=" * 60)
    
    base_url = "https://hackrx-llm-engine.onrender.com"
    token = "d66bf9184ca9c85d9572b80ca40659dc122772c748e81bbbb1c2ec5ff0d87d42"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    # Test 1: Check if service is running
    print("📊 TEST 1: Service Status Check...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"✅ Service is running: {response.json()}")
        
        # Check if it's Lightning Accuracy v18.0
        service_info = response.json()
        if "Lightning Accuracy" in service_info.get("message", ""):
            print("✅ Lightning Accuracy v18.0 detected!")
        else:
            print("⚠️ Old version still deployed - wait for redeploy")
            return False
            
    except Exception as e:
        print(f"❌ Service check failed: {e}")
        return False
    
    # Test 2: API Functionality Test
    print("\n📊 TEST 2: API Functionality Test...")
    test_data = {
        "documents": """
        Medical Research Report
        Study Overview: We successfully recruited 1,847 participants for this comprehensive clinical trial.
        Results: The overall success rate achieved was 94.3% across all test groups.
        Location: The research was conducted in multiple facilities across Germany.
        Budget: Total expenditure for this study reached 27.8 million dollars.
        Complications: Minor complications were observed in 3.1% of participants.
        """,
        "questions": [
            "How many participants were recruited?",
            "What was the success rate?",
            "Which country conducted the study?",
            "What was the total expenditure in millions?",
            "What percentage had complications?"
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/hackrx/run", 
                               json=test_data, 
                               headers=headers,
                               timeout=30)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        print(f"⚡ Response Time: {execution_time:.3f} seconds")
        print(f"📡 HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API is working correctly!")
            print(f"🎯 System: {result.get('optimization', 'Unknown')}")
            print(f"📊 Processing Time: {result.get('processing_time', 0):.3f}s")
            
            # Check answers quality
            answers = result.get('answers', [])
            print(f"\n🔍 Sample Answers:")
            for i, answer in enumerate(answers[:3]):
                print(f"  Q{i+1}: {answer[:80]}...")
            
            # Verify expected patterns
            expected_answers = ["1,847", "94.3", "Germany", "27.8", "3.1"]
            correct_count = 0
            
            for i, (answer, expected) in enumerate(zip(answers, expected_answers)):
                if expected in answer:
                    print(f"✅ Q{i+1}: Found '{expected}' in answer")
                    correct_count += 1
                else:
                    print(f"❌ Q{i+1}: Expected '{expected}' not found in answer")
            
            accuracy = (correct_count / len(expected_answers)) * 100
            print(f"\n🏆 DEPLOYMENT VERIFICATION RESULTS:")
            print(f"✅ Speed: {execution_time:.3f}s ({'PASSED' if execution_time < 20 else 'FAILED'})")
            print(f"🎯 Accuracy: {accuracy}% ({'EXCELLENT' if accuracy >= 80 else 'NEEDS IMPROVEMENT'})")
            print(f"🚀 System: Lightning Accuracy v18.0 {'READY' if accuracy >= 80 and execution_time < 20 else 'NEEDS WORK'}")
            
            return accuracy >= 80 and execution_time < 20
            
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 Starting deployment verification...")
    print("⏰ Waiting 30 seconds for deployment to complete...")
    time.sleep(30)
    
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        print(f"\n🔄 Attempt {attempt}/{max_attempts}")
        
        if test_deployed_system():
            print("\n🏆 SUCCESS! Lightning Accuracy v18.0 is deployed and working!")
            print("🎯 Your competition submission should now score much higher!")
            print("📡 API URL: https://hackrx-llm-engine.onrender.com/hackrx/run")
            return True
        
        if attempt < max_attempts:
            print(f"⏰ Waiting 60 seconds before retry {attempt + 1}...")
            time.sleep(60)
    
    print("\n❌ Deployment verification failed after all attempts")
    print("🔧 The system may still be deploying - try again in a few minutes")
    return False

if __name__ == "__main__":
    main()
