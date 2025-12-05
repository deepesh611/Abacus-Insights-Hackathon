"""
Agent Orchestrator - Coordinates Multi-Agent Fraud Detection System
Manages workflow between Detection, Investigation, and Explanation agents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fraud.rules import FraudDetector
from agents.investigation_agent import InvestigationAgent
from agents.explanation_agent import ExplanationAgent
from agents.query_agent import QueryAgent
import pandas as pd
import sqlite3


class FraudDetectionOrchestrator:
    """
    Coordinates the multi-agent fraud detection system
    Manages workflow: Detection → Investigation → Explanation → Query
    """
    
    def __init__(self, db_path='data/processed/fraud_detection.db'):
        self.db_path = db_path
        self.detection_agent = FraudDetector(db_path)
        self.investigation_agent = InvestigationAgent(db_path)
        self.explanation_agent = ExplanationAgent(db_path)
        self.query_agent = QueryAgent()
    
    def run_full_pipeline(self, investigate_top_n=10):
        """
        Execute complete fraud detection pipeline
        1. Detection Agent: Run fraud rules
        2. Investigation Agent: Deep analysis of top cases
        3. Explanation Agent: Generate reports
        """
        print("\n" + "="*80)
        print("🚀 Multi-Agent Fraud Detection System")
        print("="*80)
        
        # STEP 1: Detection Agent
        print("\n[AGENT 1] 🎯 Detection Agent - Running Fraud Rules...")
        self.detection_agent.load_data()
        detection_results = self.detection_agent.run_all_rules()
        
        # STEP 2: Investigation Agent
        print(f"\n[AGENT 2] 🔍 Investigation Agent - Analyzing Top {investigate_top_n} Cases...")
        investigation_results = self.investigation_agent.investigate_top_cases(limit=investigate_top_n)
        
        # STEP 3: Explanation Agent
        print(f"\n[AGENT 3] 📝 Explanation Agent - Generating Reports...")
        
        # Get top fraud case IDs
        conn = sqlite3.connect(self.db_path)
        top_cases = pd.read_sql_query(f"""
            SELECT claim_id FROM fraud_flags 
            WHERE fraud_detected = 1 
            ORDER BY fraud_score DESC 
            LIMIT {investigate_top_n}
        """, conn)
        conn.close()
        
        explanations = self.explanation_agent.generate_fraud_report(
            top_cases['claim_id'].tolist()
        )
        
        # Summary
        print("\n" + "="*80)
        print("✅ Multi-Agent Pipeline Complete!")
        print("="*80)
        print(f"\nProcessed: {len(detection_results)} claims")
        print(f"Fraud detected: {detection_results['fraud_detected'].sum()} cases")
        print(f"Deep investigations: {len(investigation_results)} cases")
        print(f"Reports generated: {len(explanations)} explanations")
        print("="*80)
        
        return {
            'detection': detection_results,
            'investigations': investigation_results,
            'explanations': explanations
        }
    
    def investigate_single_claim(self, claim_id):
        """
        Full agent workflow for a single claim
        Returns comprehensive fraud assessment
        """
        print(f"\n🔍 Analyzing Claim: {claim_id}\n")
        print("="*60)
        
        # Step 1: Get claim data
        conn = sqlite3.connect(self.db_path)
        
        # Get claim details
        claim_data = pd.read_sql_query(
            f"SELECT * FROM claims WHERE claim_id = '{claim_id}'", 
            conn
        )
        
        if len(claim_data) == 0:
            conn.close()
            return {"error": "Claim not found"}
        
        claim_info = claim_data.iloc[0].to_dict()
        
        # Get fraud flags
        fraud_flag = pd.read_sql_query(
            f"SELECT * FROM fraud_flags WHERE claim_id = '{claim_id}'", 
            conn
        )
        conn.close()
        
        if len(fraud_flag) == 0:
            return {"error": "Fraud flag data not found"}
        
        fraud_info = fraud_flag.iloc[0]
        
        print(f"[AGENT 1] Detection Result:")
        print(f"  Fraud Score: {fraud_info['fraud_score']}/100")
        print(f"  Rules: {fraud_info['rules_triggered']}")
        
        # Step 2: Investigation
        if fraud_info['fraud_detected']:
            print(f"\n[AGENT 2] Deep Investigation:")
            investigation = self.investigation_agent.investigate_claim(claim_id)
            print(f"  {investigation['analysis']}")
            
            # Step 3: Explanation
            print(f"\n[AGENT 3] Business Explanation:")
            explanation = self.explanation_agent.explain_fraud_case(claim_id, investigation)
            print(f"  {explanation}")
            
            print("="*60)
            
            return {
                'claim_id': claim_id,
                'claim_data': claim_info,
                'fraud_detected': True,
                'fraud_score': fraud_info['fraud_score'],
                'rules_triggered': fraud_info['rules_triggered'].split(',') if fraud_info['rules_triggered'] else [],
                'investigation': investigation,
                'explanation': explanation
            }
        else:
            print("\n  ✅ No fraud detected")
            print("="*60)
            return {
                'claim_id': claim_id,
                'claim_data': claim_info,
                'fraud_detected': False,
                'fraud_score': fraud_info.get('fraud_score', 0)
            }

    def query_system(self, question):
        """
        [AGENT 4] Query Agent - Ask questions about fraud data
        """
        print(f"\n[AGENT 4] 💬 Query Agent - Processing: '{question}'")
        result = self.query_agent.answer_question(question)
        print(f"\n🤖 Answer: {result['answer']}")
        print(f"📚 Sources: {', '.join(result['sources'])}")
        return result


# ===========================
# Main Execution
# ===========================
if __name__ == "__main__":
    print("\n🤖 Starting Multi-Agent Fraud Detection System\n")
    
    orchestrator = FraudDetectionOrchestrator()
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("1. 🚀 Run Full Fraud Detection Pipeline")
        print("2. 🔍 Investigate Single Claim")
        print("3. 💬 Ask a Question (Query Agent)")
        print("4. 🚪 Exit")
        print("="*60)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            orchestrator.run_full_pipeline(investigate_top_n=3)
            
        elif choice == '2':
            claim_id = input("Enter Claim ID (e.g., C00034): ").strip()
            orchestrator.investigate_single_claim(claim_id)
            
        elif choice == '3':
            question = input("Enter your question: ").strip()
            orchestrator.query_system(question)
            
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("Invalid choice, please try again.")
