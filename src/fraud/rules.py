"""
Fraud Detection Rules Engine
Implements 6 fraud detection rules for insurance claims
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


class FraudDetector:
    """Main fraud detection engine that applies all rules"""
    
    def __init__(self, db_path='data/processed/fraud_detection.db'):
        self.db_path = db_path
        self.df = None
        
    def load_data(self):
        """Load processed claims from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql_query("SELECT * FROM claims", conn)
        conn.close()
        print(f"✅ Loaded {len(self.df)} claims from database")
        return self
    
    def run_all_rules(self):
        """Execute all 6 fraud detection rules"""
        print("\n" + "="*60)
        print("🔍 Running Fraud Detection Rules")
        print("="*60)
        
        results = []
        
        for idx, row in self.df.iterrows():
            fraud_flags = []
            fraud_score = 0
            explanations = []
            
            # Rule 1: Duplicate Claims
            is_fraud, explanation = self.rule_duplicate_claims(row)
            if is_fraud:
                fraud_flags.append("DUPLICATE")
                fraud_score += 20
                explanations.append(explanation)
            
            # Rule 2: Amount Anomalies
            is_fraud, explanation = self.rule_amount_anomaly(row)
            if is_fraud:
                fraud_flags.append("AMOUNT_ANOMALY")
                fraud_score += 25
                explanations.append(explanation)
            
            # Rule 3: Code Mismatches
            is_fraud, explanation = self.rule_code_mismatch(row)
            if is_fraud:
                fraud_flags.append("CODE_MISMATCH")
                fraud_score += 15
                explanations.append(explanation)
            
            # Rule 4: Velocity Fraud
            is_fraud, explanation = self.rule_velocity_fraud(row)
            if is_fraud:
                fraud_flags.append("VELOCITY_FRAUD")
                fraud_score += 20
                explanations.append(explanation)
            
            # Rule 5: Provider Outliers
            is_fraud, explanation = self.rule_provider_outlier(row)
            if is_fraud:
                fraud_flags.append("PROVIDER_OUTLIER")
                fraud_score += 15
                explanations.append(explanation)
            
            # Rule 6: Impossible Scenarios
            is_fraud, explanation = self.rule_impossible_scenario(row)
            if is_fraud:
                fraud_flags.append("IMPOSSIBLE_SCENARIO")
                fraud_score += 30
                explanations.append(explanation)
            
            # Cap score at 100
            fraud_score = min(fraud_score, 100)
            
            # Store results
            results.append({
                'claim_id': row['claim_id'],
                'fraud_detected': len(fraud_flags) > 0,
                'fraud_score': fraud_score,
                'rules_triggered': ', '.join(fraud_flags) if fraud_flags else 'NONE',
                'explanation': ' | '.join(explanations) if explanations else 'No fraud detected',
                'actual_fraud': row['is_fraud']  # For validation
            })
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Print summary
        self._print_summary(results_df)
        
        # Save results
        self._save_results(results_df)
        
        return results_df
    
    # ===========================
    # RULE 1: Duplicate Claims
    # ===========================
    def rule_duplicate_claims(self, row) -> Tuple[bool, str]:
        """Detect duplicate claim IDs"""
        if row['duplicate_count'] > 1:
            return True, f"Duplicate claim: {row['claim_id']} appears {row['duplicate_count']} times"
        return False, ""
    
    # ===========================
    # RULE 2: Amount Anomalies
    # ===========================
    def rule_amount_anomaly(self, row) -> Tuple[bool, str]:
        """Detect abnormally high claim amounts using Z-score"""
        # Z-score > 3 means more than 3 standard deviations from mean
        if abs(row['amount_zscore']) > 3:
            return True, f"Amount ${row['claim_amount']:,.2f} is {abs(row['amount_zscore']):.1f}σ from specialty average"
        
        # Also flag claims > $50k
        if row['is_high_amount']:
            return True, f"Abnormally high amount: ${row['claim_amount']:,.2f}"
        
        return False, ""
    
    # ===========================
    # RULE 3: Code Mismatches
    # ===========================
    def rule_code_mismatch(self, row) -> Tuple[bool, str]:
        """Detect procedure codes that don't match diagnosis"""
        if row['is_code_mismatch']:
            return True, f"Procedure {row['procedure_code']} doesn't match diagnosis {row['diagnosis_code']}"
        return False, ""
    
    # ===========================
    # RULE 4: Velocity Fraud
    # ===========================
    def rule_velocity_fraud(self, row) -> Tuple[bool, str]:
        """Detect multiple claims in short time period"""
        # 5+ claims in 7 days is suspicious
        if row['patient_claims_7d'] >= 5:
            return True, f"Patient has {row['patient_claims_7d']} claims in past 7 days"
        
        # 10+ claims in 30 days is also suspicious
        if row['patient_claims_30d'] >= 10:
            return True, f"Patient has {row['patient_claims_30d']} claims in past 30 days"
        
        return False, ""
    
    # ===========================
    # RULE 5: Provider Outliers
    # ===========================
    def rule_provider_outlier(self, row) -> Tuple[bool, str]:
        """Detect providers billing significantly more than peers"""
        if row['is_provider_outlier']:
            provider_avg = row['provider_avg_amount']
            specialty_avg = row['specialty_avg_amount']
            ratio = provider_avg / specialty_avg if specialty_avg > 0 else 0
            return True, f"Provider bills ${provider_avg:,.2f} avg (${specialty_avg:,.2f} specialty avg, {ratio:.1f}x)"
        return False, ""
    
    # ===========================
    # RULE 6: Impossible Scenarios
    # ===========================
    def rule_impossible_scenario(self, row) -> Tuple[bool, str]:
        """Detect physically impossible scenarios"""
        # Same-day surgeries
        if row['same_day_surgeries']:
            return True, f"Patient has {row['same_day_claim_count']} surgeries on same day"
        
        # Multiple surgeries in short period (if surgery)
        if row['is_surgery'] and row['patient_claims_7d'] >= 3:
            return True, f"Patient has {row['patient_claims_7d']} surgical claims in 7 days"
        
        return False, ""
    
    # ===========================
    # Utility Methods
    # ===========================
    def _print_summary(self, results_df):
        """Print fraud detection summary"""
        total = len(results_df)
        detected = results_df['fraud_detected'].sum()
        actual_fraud = results_df['actual_fraud'].sum()
        
        # True positives, false positives, false negatives
        tp = ((results_df['fraud_detected'] == True) & (results_df['actual_fraud'] == True)).sum()
        fp = ((results_df['fraud_detected'] == True) & (results_df['actual_fraud'] == False)).sum()
        fn = ((results_df['fraud_detected'] == False) & (results_df['actual_fraud'] == True)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n" + "="*60)
        print("📊 Fraud Detection Results")
        print("="*60)
        print(f"Total claims analyzed: {total:,}")
        print(f"Fraud detected: {detected:,} ({detected/total*100:.1f}%)")
        print(f"Actual fraud cases: {actual_fraud:,}")
        print(f"\nPerformance Metrics:")
        print(f"  True Positives:  {tp:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  False Negatives: {fn:,}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall:    {recall:.2%}")
        print(f"  F1 Score:  {f1_score:.2%}")
        
        # Rule breakdown
        print(f"\nRules Triggered:")
        for rule in ['DUPLICATE', 'AMOUNT_ANOMALY', 'CODE_MISMATCH', 
                     'VELOCITY_FRAUD', 'PROVIDER_OUTLIER', 'IMPOSSIBLE_SCENARIO']:
            count = results_df['rules_triggered'].str.contains(rule).sum()
            print(f"  {rule}: {count:,}")
        
        print("="*60)
    
    def _save_results(self, results_df):
        """Save fraud detection results to database"""
        conn = sqlite3.connect(self.db_path)
        results_df.to_sql('fraud_flags', conn, if_exists='replace', index=False)
        conn.close()
        print(f"\n💾 Saved fraud flags to database: fraud_flags table")


# ===========================
# Main Execution
# ===========================
if __name__ == "__main__":
    print("\n🚀 Starting Fraud Detection System")
    
    # Initialize detector
    detector = FraudDetector()
    
    # Load data and run detection
    detector.load_data()
    results = detector.run_all_rules()
    
    # Show some examples
    print("\n" + "="*60)
    print("📋 Sample Fraud Cases Detected:")
    print("="*60)
    fraud_cases = results[results['fraud_detected'] == True].head(10)
    
    for idx, row in fraud_cases.iterrows():
        print(f"\nClaim ID: {row['claim_id']}")
        print(f"  Fraud Score: {row['fraud_score']}/100")
        print(f"  Rules Triggered: {row['rules_triggered']}")
        print(f"  Explanation: {row['explanation']}")
        print(f"  Actual Fraud: {'✓ Yes' if row['actual_fraud'] else '✗ No'}")
    
    print("\n" + "="*60)
    print("✅ Phase 4 Complete: Fraud Detection Rules Implemented!")
    print("="*60)
