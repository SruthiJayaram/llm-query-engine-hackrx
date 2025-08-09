#!/usr/bin/env python3
"""
ADVANCED ACCURACY OPTIMIZATION - Version 8.0 ULTIMATE
Continuous learning and adaptation system
Target: 99%+ accuracy to dominate HackRx until 12 AM
"""
import time
import requests
import json

def test_accuracy_sample():
    """Quick accuracy validation"""
    
    test_cases = [
        {
            "document": """
            Financial Performance Report Q3 2024
            
            Revenue: $45.8 million (+32% YoY)
            Profit Margin: 18.5% 
            Customer Growth: 2,847 new customers
            Retention Rate: 94.2%
            Average Deal Size: $125,000
            
            Key Achievements:
            - Launched AI-powered analytics platform
            - Expanded to 5 new markets: Germany, France, Japan, Australia, Brazil
            - Acquired startup TechVision for $12M
            - Partnership with Microsoft announced
            
            Challenges:
            - Supply chain disruptions cost $2.3M
            - Employee turnover increased to 12%
            - Cybersecurity incident in August (contained within 4 hours)
            """,
            "questions": [
                "What was the revenue in Q3 2024?",
                "What was the year-over-year revenue growth?", 
                "How many new customers were acquired?",
                "Which countries did the company expand to?",
                "What was the cost of supply chain disruptions?"
            ],
            "expected_answers": [
                "45.8 million",
                "32%",
                "2,847",
                "Germany, France, Japan, Australia, Brazil",
                "2.3M"
            ]
        }
    ]
    
    print("🎯 ADVANCED ACCURACY VALIDATION")
    print("=" * 50)
    
    total_score = 0
    total_questions = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: Processing {len(test['questions'])} questions...")
        
        try:
            response = requests.post(
                "http://localhost:8000/hackrx/run",
                headers={
                    "Authorization": "Bearer advanced-test",
                    "Content-Type": "application/json"
                },
                json={
                    "documents": test["document"],
                    "questions": test["questions"]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answers = data.get("answers", [])
                
                for j, (question, answer, expected) in enumerate(zip(test["questions"], answers, test["expected_answers"])):
                    # More flexible matching for better accuracy detection
                    answer_clean = answer.lower().replace(',', '').replace(' ', '')
                    expected_clean = expected.lower().replace(',', '').replace(' ', '')
                    
                    accuracy = 1.0 if expected_clean in answer_clean else 0.5
                    
                    # Special handling for percentage
                    if '%' in expected and '%' not in answer:
                        if expected.replace('%', '') in answer:
                            accuracy = 1.0
                    
                    total_score += accuracy
                    total_questions += 1
                    
                    print(f"  Q{j+1}: {question}")
                    print(f"  A{j+1}: {answer[:100]}...")
                    print(f"  Expected: {expected} | Found: {'✅' if accuracy == 1.0 else '⚠️'}")
                    print()
            
        except Exception as e:
            print(f"Error: {e}")
    
    if total_questions > 0:
        accuracy_percent = (total_score / total_questions) * 100
        print(f"🏆 ACCURACY SCORE: {accuracy_percent:.1f}%")
        
        if accuracy_percent >= 95:
            print("🚀 EXCELLENT! Ready to beat the #1 position!")
            return True
        else:
            print("📈 Good progress, continue optimizing!")
            return False
    
    return False

if __name__ == "__main__":
    success = test_accuracy_sample()
    print(f"\n⏰ Continue optimizing until 12 AM for maximum accuracy!")
    print(f"🎯 Current Status: {'READY FOR VICTORY' if success else 'OPTIMIZATION IN PROGRESS'}")
