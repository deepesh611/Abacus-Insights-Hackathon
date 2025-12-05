"""
Explanation Agent - Human-Readable Fraud Reports
Generates clear, non-technical explanations of fraud findings
"""

import sqlite3
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import LLMClient


class ExplanationAgent:
    """
    Generates clear, business-friendly fraud explanations
    Translates technical fraud flags into actionable insights
    """
    
    def __init__(self, db_path='data/processed/fraud_detection.db'):
        self.db_path = db_path
        self.llm = LLMClient()
    
    def explain_fraud_case(self, claim_id, investigation_result=None):
        """
        Generate human-readable explanation for a fraud case
        """
        # Get fraud flags
        fraud_info = self._get_fraud_info(claim_id)
        if fraud_info is None:
            return f"No fraud information found for claim {claim_id}"
        
        # Get claim details
        claim = self._get_claim_details(claim_id)
        
        # Build explanation prompt
        prompt = self._build_explanation_prompt(claim, fraud_info, investigation_result)
        
        # Get LLM explanation
        explanation = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.7)
        
        return explanation
    
    def generate_fraud_report(self, claim_ids):
        """
        Generate batch fraud report for multiple claims
        """
        print(f"\n📝 Generating Fraud Report for {len(claim_ids)} Claims\n")
        print("="*80)
        
        report = []
        for claim_id in claim_ids:
            explanation = self.explain_fraud_case(claim_id)
            report.append({
                'claim_id': claim_id,
                'explanation': explanation
            })
            print(f"\n{claim_id}:")
            print(explanation)
            print("-"*80)
        
        return report
    
    # ===========================
    # Helper Methods
    # ===========================
    
    def _get_fraud_info(self, claim_id):
        """Get fraud detection info for claim"""
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM fraud_flags WHERE claim_id = '{claim_id}'"
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(result) == 0:
            return None
        
        return result.iloc[0].to_dict()
    
    def _get_claim_details(self, claim_id):
        """Get claim details"""
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM claims WHERE claim_id = '{claim_id}'"
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(result) == 0:
            return None
        
        return result.iloc[0].to_dict()
    
    def _build_explanation_prompt(self, claim, fraud_info, investigation=None):
        """Build prompt for generating explanation"""
        
        prompt = f"""You are explaining a fraud case to a non-technical insurance adjuster. 

CLAIM INFORMATION:
- Claim ID: {claim['claim_id']}
- Amount: ${claim['claim_amount']:,.2f}
- Provider: {claim['provider_specialty']}
- Procedure: {claim['procedure_code']}
- Diagnosis: {claim['diagnosis_code']}

FRAUD DETECTION RESULTS:
- Fraud Score: {fraud_info['fraud_score']}/100
- Rules Triggered: {fraud_info['rules_triggered']}
- Technical Explanation: {fraud_info['explanation']}

Generate a clear, concise explanation (2-3 sentences) that:
1. States WHY this claim is suspicious
2. Highlights the KEY red flag
3. Is understandable to someone without technical knowledge

Do not use jargon like "Z-score" or "statistical outlier". Use plain language like "unusually high amount" or "multiple claims in short time".

Example format:
"This claim is flagged as suspicious because [main reason]. Specifically, [key red flag]. This pattern is commonly associated with fraudulent billing."

Your explanation:"""

        return prompt


# ===========================
# Main Execution
# ===========================
if __name__ == "__main__":
    print("\n📝 Starting Explanation Agent\n")
    
    # Initialize agent
    agent = ExplanationAgent()
    
    # Get some fraud cases
    conn = sqlite3.connect('data/processed/fraud_detection.db')
    fraud_cases = pd.read_sql_query("""
        SELECT claim_id FROM fraud_flags 
        WHERE fraud_detected = 1 
        ORDER BY fraud_score DESC 
        LIMIT 5
    """, conn)
    conn.close()
    
    # Generate explanations
    claim_ids = fraud_cases['claim_id'].tolist()
    report = agent.generate_fraud_report(claim_ids)
    
    print("\n" + "="*80)
    print("✅ Explanation Agent Complete!")
    print("="*80)
