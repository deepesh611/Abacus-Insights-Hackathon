"""
Investigation Agent - LLM-Powered Deep Fraud Analysis
Analyzes flagged claims using AI to provide expert fraud assessment
"""

import sqlite3
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import LLMClient


class InvestigationAgent:
    """
    AI-powered agent that investigates suspicious claims
    Uses LLM to analyze context and provide expert fraud assessment
    """
    
    def __init__(self, db_path='data/processed/fraud_detection.db'):
        self.db_path = db_path
        self.llm = LLMClient()
    
    def investigate_claim(self, claim_id):
        """
        Deep investigation of a specific claim
        Returns: dict with fraud likelihood, red flags, and priority
        """
        # 1. Get claim details
        claim = self._get_claim_details(claim_id)
        if claim is None:
            return {"error": f"Claim {claim_id} not found"}
        
        # 2. Get provider context
        provider_context = self._get_provider_context(claim['provider_id'])
        
        # 3. Get patient context
        patient_context = self._get_patient_context(claim['patient_id'])
        
        # 4. Get fraud flags
        fraud_flags = self._get_fraud_flags(claim_id)
        
        # 5. Build investigation prompt
        prompt = self._build_investigation_prompt(claim, provider_context, 
                                                  patient_context, fraud_flags)
        
        # 6. Get LLM analysis
        analysis = self.llm.analyze_fraud(prompt, "")
        
        # 7. Parse and return results
        return {
            'claim_id': claim_id,
            'analysis': analysis,
            'claim_amount': claim['claim_amount'],
            'provider': claim['provider_id'],
            'specialty': claim['provider_specialty']
        }
    
    def investigate_top_cases(self, limit=10):
        """Investigate the top N flagged cases by fraud score"""
        print(f"\n🔍 Investigating Top {limit} Fraud Cases\n")
        print("="*80)
        
        # Get top fraud cases
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT claim_id, fraud_score, rules_triggered, explanation
            FROM fraud_flags
            WHERE fraud_detected = 1
            ORDER BY fraud_score DESC
            LIMIT {limit}
        """
        top_cases = pd.read_sql_query(query, conn)
        conn.close()
        
        results = []
        for idx, case in top_cases.iterrows():
            print(f"\n[{idx+1}/{limit}] Investigating Claim: {case['claim_id']}")
            print(f"  Fraud Score: {case['fraud_score']}/100")
            print(f"  Rules Triggered: {case['rules_triggered']}")
            
            # Investigate
            investigation = self.investigate_claim(case['claim_id'])
            results.append(investigation)
            
            print(f"\n  🤖 AI Investigation:")
            print(f"  {investigation['analysis']}")
            print("-"*80)
        
        return results
    
    # ===========================
    # Helper Methods
    # ===========================
    
    def _get_claim_details(self, claim_id):
        """Get full details of a specific claim"""
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM claims WHERE claim_id = '{claim_id}'"
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(result) == 0:
            return None
        
        return result.iloc[0].to_dict()
    
    def _get_provider_context(self, provider_id):
        """Get provider history and statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get provider summary
        query = f"SELECT * FROM providers WHERE provider_id = '{provider_id}'"
        provider = pd.read_sql_query(query, conn)
        
        # Get provider's recent claims
        query = f"""
            SELECT claim_amount, claim_date, status, is_fraud
            FROM claims
            WHERE provider_id = '{provider_id}'
            ORDER BY claim_date DESC
            LIMIT 10
        """
        recent_claims = pd.read_sql_query(query, conn)
        conn.close()
        
        return {
            'summary': provider.iloc[0].to_dict() if len(provider) > 0 else {},
            'recent_claims': recent_claims
        }
    
    def _get_patient_context(self, patient_id):
        """Get patient history and statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get patient summary
        query = f"SELECT * FROM patients WHERE patient_id = '{patient_id}'"
        patient = pd.read_sql_query(query, conn)
        
        # Get patient's recent claims
        query = f"""
            SELECT claim_amount, claim_date, provider_specialty, is_fraud
            FROM claims
            WHERE patient_id = '{patient_id}'
            ORDER BY claim_date DESC
            LIMIT 10
        """
        recent_claims = pd.read_sql_query(query, conn)
        conn.close()
        
        return {
            'summary': patient.iloc[0].to_dict() if len(patient) > 0 else {},
            'recent_claims': recent_claims
        }
    
    def _get_fraud_flags(self, claim_id):
        """Get fraud detection results for this claim"""
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM fraud_flags WHERE claim_id = '{claim_id}'"
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(result) == 0:
            return None
        
        return result.iloc[0].to_dict()
    
    def _build_investigation_prompt(self, claim, provider_ctx, patient_ctx, fraud_flags):
        """Build comprehensive prompt for LLM investigation"""
        
        provider = provider_ctx['summary']
        patient = patient_ctx['summary']
        
        prompt = f"""You are an expert fraud investigator analyzing an insurance claim flagged as potentially fraudulent.

CLAIM DETAILS:
- Claim ID: {claim['claim_id']}
- Amount: ${claim['claim_amount']:,.2f}
- Date: {claim['claim_date']}
- Provider: {claim['provider_id']} ({claim['provider_specialty']})
- Procedure: {claim['procedure_code']}
- Diagnosis: {claim['diagnosis_code']}
- Status: {claim['status']}

FRAUD FLAGS TRIGGERED:
- Rules: {fraud_flags['rules_triggered']}
- Fraud Score: {fraud_flags['fraud_score']}/100
- Explanation: {fraud_flags['explanation']}

PROVIDER CONTEXT:
- Total claims submitted: {provider.get('total_claims', 'N/A')}
- Average claim amount: ${provider.get('avg_claim_amount', 0):,.2f}
- Total billed: ${provider.get('total_billed', 0):,.2f}
- Previous fraud cases: {provider.get('fraud_claims', 0)}

PATIENT CONTEXT:
- Total claims: {patient.get('total_claims', 'N/A')}
- Total spent: ${patient.get('total_spent', 0):,.2f}
- Previous fraud cases: {patient.get('fraud_claims', 0)}

CLAIM Z-SCORE: {claim.get('amount_zscore', 0):.2f} (higher = more unusual for specialty)

Based on this information, provide a concise fraud investigation report in the following format:

FRAUD LIKELIHOOD: [1-10 score]
KEY RED FLAGS: [2-3 specific concerns]
INVESTIGATION PRIORITY: [Low/Medium/High/Critical]
RECOMMENDATION: [Approve/Deny/Request More Info/Escalate]

Keep your response concise and focus on the most important fraud indicators."""

        return prompt


# ===========================
# Main Execution
# ===========================
if __name__ == "__main__":
    print("\n🤖 Starting Investigation Agent\n")
    
    # Initialize agent
    agent = InvestigationAgent()
    
    # Investigate top fraud cases
    results = agent.investigate_top_cases(limit=5)
    
    print("\n" + "="*80)
    print("✅ Investigation Agent Complete!")
    print("="*80)
