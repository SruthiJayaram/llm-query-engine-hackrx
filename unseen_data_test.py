#!/usr/bin/env python3
"""
PROPER TRAIN/TEST VALIDATION - v11.0
Testing on COMPLETELY UNSEEN data patterns
No overfitting to training examples
"""
import time
import requests
import json

def run_unseen_data_validation():
    """Test on completely different document types and patterns than training data"""
    
    # COMPLETELY DIFFERENT test data - no overlap with training patterns
    unseen_test_suite = [
        {
            "name": "Medical Research Report (UNSEEN FORMAT)",
            "document": """
            Cardiovascular Treatment Efficacy Study - Phase II Results
            
            Patient Enrollment: 3,245 individuals recruited globally
            Primary Endpoint Achievement: 92.7% success rate documented
            Treatment Duration: 14-month observation period
            Global Sites: Research conducted across Sweden, Netherlands, Singapore, Chile, South Korea
            Study Budget: Total expenditure reached €18.3 million
            Adverse Events: Treatment complications observed in 4.2% of cases
            Patient Demographics: Age range 45-78 years, 58% female participation
            """,
            "questions": [
                "How many individuals were recruited?",
                "What was the success rate documented?",
                "Which countries conducted research?",
                "What was the total expenditure?",
                "What percentage had complications?"
            ],
            "expected_contains": ["3245", "92.7", "Sweden", "18.3", "4.2"]
        },
        {
            "name": "Technology Startup Report (UNSEEN FORMAT)",
            "document": """
            TechFlow Inc. - Series B Funding Announcement
            
            Funding Round: Successfully raised $47.2 million in Series B
            Investor Participation: Led by venture capital firms from 4 regions
            User Base Expansion: Platform now serves 1,876,543 active users monthly
            Engineering Team Growth: Hired 127 software engineers this year
            Market Presence: Operations launched in Dubai, Cairo, Lagos, Nairobi, Istanbul
            Burn Rate: Monthly operational expenses total $3.8 million
            Growth Trajectory: 340% increase in user engagement year-over-year
            """,
            "questions": [
                "How much was raised in Series B?",
                "How many active users monthly?",
                "How many software engineers hired?",
                "Which cities have operations?",
                "What are monthly operational expenses?"
            ],
            "expected_contains": ["47.2", "1876543", "127", "Dubai", "3.8"]
        },
        {
            "name": "Educational Institution Report (UNSEEN FORMAT)",
            "document": """
            Global University Network - Academic Year 2024-25 Summary
            
            Student Population: Total enrollment of 89,456 students across all campuses
            Faculty Strength: 4,312 professors and lecturers employed full-time
            Research Funding: Secured $23.7 million in grants for scientific research
            International Programs: Exchange partnerships established with institutions in 
            Norway, Finland, Denmark, Iceland, Estonia
            Digital Learning: Online course completion rate improved to 87.3%
            Infrastructure Investment: Campus modernization cost $15.6 million
            Graduation Success: 94.1% of students completed their programs successfully
            """,
            "questions": [
                "What is the total enrollment?",
                "How many professors employed?",
                "How much secured in research grants?",
                "Which countries have exchange partnerships?",
                "What was the campus modernization cost?"
            ],
            "expected_contains": ["89456", "4312", "23.7", "Norway", "15.6"]
        },
        {
            "name": "Manufacturing Industry Report (UNSEEN FORMAT)", 
            "document": """
            Industrial Manufacturing Consortium - Quarterly Operations Brief
            
            Production Volume: Manufactured 2,847,392 units in Q3 2024
            Quality Metrics: Defect rate maintained at 0.8% across all product lines
            Supply Chain: Raw materials sourced from facilities in Vietnam, Thailand, Malaysia, Indonesia, Philippines
            Workforce Data: 15,673 production workers employed across multiple shifts
            Energy Efficiency: Reduced power consumption by 18.5% through automation
            Capital Investment: Equipment upgrades totaled $31.4 million this quarter
            Export Markets: Products shipped to 67 different countries worldwide
            """,
            "questions": [
                "How many units manufactured in Q3?",
                "What is the defect rate percentage?",
                "Which countries supply raw materials?",
                "How many production workers employed?",
                "How much spent on equipment upgrades?"
            ],
            "expected_contains": ["2847392", "0.8", "Vietnam", "15673", "31.4"]
        }
    ]
    
    print("🧪 UNSEEN DATA VALIDATION TEST")
    print("🎯 Testing GENERALIZED PRECISION on completely new patterns")
    print("📊 No overlap with training data - True generalization test")
    print("=" * 70)
    
    total_score = 0
    total_questions = 0
    test_results = []
    
    for i, test in enumerate(unseen_test_suite, 1):
        print(f"\n📋 Test {i}: {test['name']}")
        print(f"Processing {len(test['questions'])} questions on unseen format...")
        
        try:
            response = requests.post(
                "http://localhost:8000/hackrx/run",
                headers={
                    "Authorization": "Bearer unseen-data-test",
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
                test_details = []
                
                for j, (question, answer, expected) in enumerate(zip(test["questions"], answers, test["expected_contains"])):
                    # Flexible matching for unseen data
                    answer_clean = answer.lower().replace(',', '').replace(' ', '').replace('$', '').replace('€', '')
                    expected_clean = expected.lower().replace(',', '').replace(' ', '')
                    
                    # Check if expected content is found in answer
                    is_correct = expected_clean in answer_clean
                    
                    # Special handling for different number formats
                    if not is_correct and expected_clean.isdigit():
                        # Try different number representations
                        expected_num = expected_clean
                        if expected_num in answer_clean or expected_num.replace('.', '') in answer_clean:
                            is_correct = True
                    
                    accuracy = 1.0 if is_correct else 0.0
                    test_score += accuracy
                    total_score += accuracy
                    total_questions += 1
                    
                    status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                    print(f"  Q{j+1}: {question}")
                    print(f"  A{j+1}: {answer[:100]}...")
                    print(f"  Expected: {expected} | Status: {status}")
                    
                    test_details.append({
                        'question': question,
                        'answer': answer,
                        'expected': expected,
                        'correct': is_correct
                    })
                
                test_accuracy = (test_score / len(test["questions"])) * 100
                print(f"\n🎯 {test['name']} Accuracy: {test_accuracy:.1f}%")
                
                test_results.append({
                    'name': test['name'],
                    'accuracy': test_accuracy,
                    'details': test_details
                })
                
            else:
                print(f"❌ Request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n" + "=" * 70)
    
    if total_questions > 0:
        overall_accuracy = (total_score / total_questions) * 100
        print(f"🏆 OVERALL UNSEEN DATA ACCURACY: {overall_accuracy:.1f}%")
        print(f"📊 Total Questions Tested: {total_questions}")
        print(f"✅ Correct Answers: {int(total_score)}")
        print(f"❌ Incorrect Answers: {total_questions - int(total_score)}")
        
        print(f"\n📈 INDIVIDUAL TEST PERFORMANCE:")
        for result in test_results:
            print(f"  • {result['name']}: {result['accuracy']:.1f}%")
        
        if overall_accuracy >= 70:
            print(f"\n🚀 EXCELLENT GENERALIZATION! System works on unseen data!")
            print(f"💪 Ready to beat 95% leader with true generalization!")
            return True
        elif overall_accuracy >= 50:
            print(f"\n📈 GOOD GENERALIZATION! Some improvements needed.")
            return False
        else:
            print(f"\n⚠️ OVERFITTING DETECTED! System needs better generalization.")
            return False
    
    return False

if __name__ == "__main__":
    print("🔬 STARTING UNSEEN DATA GENERALIZATION TEST")
    print("🎯 Testing true AI capability on completely new patterns")
    print("📊 This is the REAL test - no training data overlap!")
    
    success = run_unseen_data_validation()
    
    print(f"\n🏁 FINAL RESULT: {'🏆 GENERALIZATION SUCCESS!' if success else '📈 NEEDS IMPROVEMENT'}")
    print("💡 This test proves real AI capability, not memorization!")
