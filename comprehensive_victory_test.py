#!/usr/bin/env python3
"""
COMPREHENSIVE VICTORY VALIDATION - v9.0
Final validation to confirm 99%+ accuracy across ALL document types
Designed to beat the 95% leader with consistent performance
"""
import time
import requests
import json

def run_comprehensive_validation():
    """Comprehensive validation across multiple document types"""
    
    test_suite = [
        {
            "name": "Financial Report Test",
            "document": """
            Q4 2024 Financial Results
            
            Total Revenue: $125.6 million (+28% YoY)
            Net Profit: $23.4 million
            New Customer Acquisitions: 4,521 customers
            Market Expansion: Successfully entered 3 new regions: Canada, UK, India
            R&D Investment: $8.7 million in product development
            Operational Costs: Marketing campaigns cost $12.5M this quarter
            """,
            "questions": [
                "What was the total revenue?",
                "What was the year-over-year growth?",
                "How many new customers were acquired?",
                "Which regions did the company enter?",
                "What was the cost of marketing campaigns?"
            ],
            "expected_keywords": ["125.6", "28%", "4521", "Canada, UK, India", "12.5"]
        },
        {
            "name": "Scientific Research Test", 
            "document": """
            Clinical Trial Results - Phase III
            
            Study Population: 2,847 participants
            Success Rate: 89.3% efficacy
            Geographic Distribution: Sites across Germany, France, Japan, Brazil, Australia
            Funding: Total study cost $15.2 million
            Duration: 18-month study period (+12% faster than predicted)
            """,
            "questions": [
                "How many participants were in the study?",
                "What was the efficacy rate?", 
                "Which countries had study sites?",
                "What was the total study cost?",
                "How much faster was the study than predicted?"
            ],
            "expected_keywords": ["2847", "89.3%", "Germany, France, Japan, Brazil, Australia", "15.2", "12%"]
        },
        {
            "name": "Business Report Test",
            "document": """
            Annual Operations Summary
            
            Employee Count: 1,234 total employees (+15% YoY growth)
            Office Locations: Expanded to 7 new cities: Tokyo, London, Berlin, Sydney, Toronto, Mumbai, São Paulo
            Training Investment: $3.8 million in employee development
            Technology Upgrade: Infrastructure improvements cost $6.2M
            Productivity Increase: 23% improvement in efficiency metrics
            """,
            "questions": [
                "How many total employees?",
                "What was the year-over-year employee growth?",
                "Which cities did they expand to?", 
                "What was spent on training?",
                "What was the productivity improvement?"
            ],
            "expected_keywords": ["1234", "15%", "Tokyo, London, Berlin, Sydney, Toronto, Mumbai, São Paulo", "3.8", "23%"]
        }
    ]
    
    print("🏆 COMPREHENSIVE VICTORY VALIDATION")
    print("🎯 Testing MAXIMUM PRECISION v9.0 across document types")
    print("=" * 60)
    
    total_score = 0
    total_questions = 0
    
    for i, test in enumerate(test_suite, 1):
        print(f"\n📋 Test {i}: {test['name']}")
        print(f"Processing {len(test['questions'])} questions...")
        
        try:
            response = requests.post(
                "http://localhost:8000/hackrx/run",
                headers={
                    "Authorization": "Bearer comprehensive-test",
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
                
                test_score = 0
                for j, (question, answer, expected) in enumerate(zip(test["questions"], answers, test["expected_keywords"])):
                    # Flexible accuracy checking
                    answer_clean = answer.lower().replace(',', '').replace(' ', '')
                    expected_clean = expected.lower().replace(',', '').replace(' ', '')
                    
                    is_accurate = expected_clean in answer_clean
                    accuracy = 1.0 if is_accurate else 0.0
                    
                    test_score += accuracy
                    total_score += accuracy
                    total_questions += 1
                    
                    status = "✅ PERFECT" if is_accurate else "❌ NEEDS FIX"
                    print(f"  Q{j+1}: {question}")
                    print(f"  A{j+1}: {answer[:80]}...")
                    print(f"  Expected: {expected} | Status: {status}")
                
                test_accuracy = (test_score / len(test["questions"])) * 100
                print(f"\n🎯 {test['name']} Accuracy: {test_accuracy:.1f}%")
                
            else:
                print(f"❌ Request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if total_questions > 0:
        overall_accuracy = (total_score / total_questions) * 100
        print(f"\n" + "=" * 60)
        print(f"🏆 OVERALL ACCURACY: {overall_accuracy:.1f}%")
        
        if overall_accuracy >= 95:
            print("🚀 VICTORY CONFIRMED! Ready to DOMINATE HackRx!")
            print("💪 MAXIMUM PRECISION v9.0 is CRUSHING the competition!")
            return True
        else:
            print("📈 Good progress, continue optimizing!")
            return False
    
    return False

if __name__ == "__main__":
    print("🔥 Starting COMPREHENSIVE VICTORY VALIDATION")
    print("⏰ Optimizing until 12 AM to maintain #1 position")
    
    success = run_comprehensive_validation()
    
    print(f"\n🎯 Final Status: {'🏆 VICTORY MODE - READY TO DOMINATE!' if success else '📈 OPTIMIZATION CONTINUES'}")
    print("💪 MAXIMUM PRECISION v9.0 - Beating 95% leader with advanced entity extraction!")
