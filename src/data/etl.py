import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

class FraudETLPipeline:
    def __init__(self):
        self.raw_data_path = 'data/raw/claims_data.csv'
        self.db_path = 'data/processed/fraud_detection.db'
        self.df = None
        Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    def extract(self):
        print("\n[1/3] EXTRACT: Loading raw data...")
        self.df = pd.read_csv(self.raw_data_path)
        print(f"✅ Loaded {len(self.df)} claims from CSV")
        return self
    
    def transform(self):
        print("\n[2/3] TRANSFORM: Engineering fraud detection features...")
        self.df['claim_date'] = pd.to_datetime(self.df['claim_date'])
        
        # 1. Provider Statistics
        print("  → Calculating provider statistics...")
        provider_stats = self.df.groupby('provider_id')['claim_amount'].agg([
            ('provider_avg_amount', 'mean'),
            ('provider_total_claims', 'count')
        ]).reset_index()
        self.df = self.df.merge(provider_stats, on='provider_id', how='left')
        
        # 2. Specialty Statistics
        print("  → Calculating specialty averages...")
        specialty_stats = self.df.groupby('provider_specialty')['claim_amount'].agg([
            ('specialty_avg_amount', 'mean'),
            ('specialty_std_amount', 'std')
        ]).reset_index()
        self.df = self.df.merge(specialty_stats, on='provider_specialty', how='left')
        
        # 3. Z-Score for Amounts (within specialty)
        print("  → Computing Z-scores for claim amounts...")
        self.df['amount_zscore'] = (
            (self.df['claim_amount'] - self.df['specialty_avg_amount']) / 
            self.df['specialty_std_amount']
        )
        self.df['amount_zscore'] = self.df['amount_zscore'].fillna(0)
        
        # 4. Provider Outlier Flag (billing 2x+ specialty average)
        print("  → Flagging provider outliers...")
        self.df['is_provider_outlier'] = (
            self.df['provider_avg_amount'] > self.df['specialty_avg_amount'] * 2
        )
        
        # 5. Patient Claim Frequency
        print("  → Calculating patient claim frequencies...")
        
        # Sort by date for rolling calculations
        self.df = self.df.sort_values(['patient_id', 'claim_date'])
        
        # Count claims per patient in last 7 days
        self.df['patient_claims_7d'] = self.df.groupby('patient_id').apply(
            lambda x: x['claim_date'].apply(
                lambda date: ((x['claim_date'] >= date - pd.Timedelta(days=7)) & 
                             (x['claim_date'] <= date)).sum()
            )
        ).reset_index(level=0, drop=True)
        
        # Count claims per patient in last 30 days
        self.df['patient_claims_30d'] = self.df.groupby('patient_id').apply(
            lambda x: x['claim_date'].apply(
                lambda date: ((x['claim_date'] >= date - pd.Timedelta(days=30)) & 
                             (x['claim_date'] <= date)).sum()
            )
        ).reset_index(level=0, drop=True)
        
        # Same-day claims for same patient
        print("  → Detecting same-day claims...")
        same_day_counts = self.df.groupby(['patient_id', 'claim_date']).size().reset_index(name='same_day_claim_count')
        self.df = self.df.merge(same_day_counts, on=['patient_id', 'claim_date'], how='left')
        
        # 6. Code Mismatch Detection
        print("  → Validating procedure/diagnosis codes...")
        self.df['is_code_mismatch'] = self._detect_code_mismatches()
        
        # 7. Date Features
        print("  → Extracting date features...")
        self.df['day_of_week'] = self.df['claim_date'].dt.dayofweek
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])
        self.df['month'] = self.df['claim_date'].dt.month
        
        # 8. Duplicate Detection
        print("  → Detecting duplicate claims...")
        duplicate_counts = self.df.groupby('claim_id').size().reset_index(name='duplicate_count')
        self.df = self.df.merge(duplicate_counts, on='claim_id', how='left')
        self.df['is_duplicate'] = self.df['duplicate_count'] > 1
        
        # 9. High Amount Flag
        print("  → Flagging abnormally high amounts...")
        self.df['is_high_amount'] = self.df['claim_amount'] > 50000
        
        # 10. Surgery Detection (for impossible scenarios)
        print("  → Detecting surgical procedures...")
        surgery_codes = ['SURG001', 'SURG002', 'SURG003', 'SURG004', 
                        'CARD003', 'ORTH003', 'ONC002']
        self.df['is_surgery'] = self.df['procedure_code'].isin(surgery_codes)
        
        # Flag same-day surgeries (impossible)
        self.df['same_day_surgeries'] = self.df.apply(
            lambda row: (row['same_day_claim_count'] > 1 and row['is_surgery']), 
            axis=1
        )
        
        print(f"✅ Engineered {len(self.df.columns)} total features")
        return self
    
    def _detect_code_mismatches(self):
        """Detect procedure codes that don't match diagnosis specialty"""
        # Define valid procedure prefixes for each diagnosis prefix
        valid_combos = {
            'D_HEART': ['CARD'],
            'D_CARDIAC': ['CARD'],
            'D_VASCULAR': ['CARD'],
            'D_CANCER': ['ONC'],
            'D_TUMOR': ['ONC'],
            'D_LEUKEMIA': ['ONC'],
            'D_LYMPHOMA': ['ONC'],
            'D_CHILD': ['PED'],
            'D_INFANT': ['PED'],
            'D_FEVER': ['PED', 'GP'],
            'D_ALLERGY': ['PED', 'GP'],
            'D_BONE': ['ORTH'],
            'D_JOINT': ['ORTH'],
            'D_FRACTURE': ['ORTH'],
            'D_SPRAIN': ['ORTH'],
            'D_ROUTINE': ['GP'],
            'D_CHECKUP': ['GP'],
            'D_FLU': ['GP', 'ER'],
            'D_COMMON': ['GP'],
            'D_ACCIDENT': ['ER', 'ORTH'],
            'D_TRAUMA': ['ER', 'SURG'],
            'D_URGENT': ['ER'],
            'D_CRITICAL': ['ER', 'SURG'],
            'D_SURGICAL': ['SURG'],
            'D_OPERATION': ['SURG'],
            'D_PROCEDURE': ['SURG'],
            'D_INVASIVE': ['SURG'],
            'D_XRAY': ['RAD'],
            'D_SCAN': ['RAD'],
            'D_IMAGING': ['RAD'],
            'D_MRI': ['RAD']
        }
        
        mismatches = []
        for _, row in self.df.iterrows():
            diagnosis = row['diagnosis_code']
            procedure = row['procedure_code']
            
            # Extract prefix
            diag_prefix = '_'.join(diagnosis.split('_')[:2])  # e.g., "D_HEART"
            proc_prefix = procedure.split('0')[0]  # e.g., "CARD"
            
            # Check if valid combo
            if diag_prefix in valid_combos:
                is_valid = proc_prefix in valid_combos[diag_prefix]
                mismatches.append(not is_valid)
            else:
                mismatches.append(False)  # Unknown combo, assume valid
        
        return mismatches
    
    def load(self):
        print("\n[3/3] LOAD: Saving to SQLite database...")
        
        # Connect to SQLite
        conn = sqlite3.connect(self.db_path)
        
        # Save main claims table
        self.df.to_sql('claims', conn, if_exists='replace', index=False)
        print(f"✅ Saved 'claims' table ({len(self.df)} rows)")
        
        # Create provider summary table
        provider_summary = self.df.groupby(['provider_id', 'provider_specialty']).agg({
            'claim_amount': ['mean', 'sum', 'count'],
            'is_fraud': 'sum'
        }).reset_index()
        provider_summary.columns = ['provider_id', 'provider_specialty', 
                                   'avg_claim_amount', 'total_billed', 
                                   'total_claims', 'fraud_claims']
        provider_summary.to_sql('providers', conn, if_exists='replace', index=False)
        print(f"✅ Saved 'providers' table ({len(provider_summary)} rows)")
        
        # Create patient summary table
        patient_summary = self.df.groupby('patient_id').agg({
            'claim_amount': ['sum', 'count'],
            'is_fraud': 'sum'
        }).reset_index()
        patient_summary.columns = ['patient_id', 'total_spent', 
                                  'total_claims', 'fraud_claims']
        patient_summary.to_sql('patients', conn, if_exists='replace', index=False)
        print(f"✅ Saved 'patients' table ({len(patient_summary)} rows)")
        
        conn.close()
        print(f"\n💾 Database saved to: {self.db_path}")
        return self
    
    def run(self):
        """Execute full ETL pipeline"""
        print("="*60)
        print("🔄 Starting ETL Pipeline")
        print("="*60)
        
        self.extract()
        self.transform()
        self.load()
        
        # Summary statistics
        print("\n" + "="*60)
        print("📊 ETL Summary")
        print("="*60)
        print(f"Total claims processed: {len(self.df)}")
        print(f"Features engineered: {len(self.df.columns)}")
        print(f"\nFraud Detection Flags:")
        print(f"  - Duplicates: {self.df['is_duplicate'].sum()}")
        print(f"  - High amounts: {self.df['is_high_amount'].sum()}")
        print(f"  - Code mismatches: {self.df['is_code_mismatch'].sum()}")
        print(f"  - Provider outliers: {self.df['is_provider_outlier'].sum()}")
        print(f"  - Same-day surgeries: {self.df['same_day_surgeries'].sum()}")
        print(f"  - Velocity fraud (7d): {(self.df['patient_claims_7d'] >= 5).sum()}")
        print(f"\nActual fraud cases: {self.df['is_fraud'].sum()}")
        print("="*60)
        print("✅ ETL Pipeline Complete!")
        print("="*60)


if __name__ == "__main__":
    pipeline = FraudETLPipeline()
    pipeline.run()
