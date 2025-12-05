import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class FraudDataGenerator:
    def __init__(self, num_rows=2500, random_seed=42):
        self.num_rows = num_rows
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Specialties and their typical claim ranges
        self.specialties = {
            'Cardiology': (1000, 15000),
            'Oncology': (2000, 20000),
            'Pediatrics': (500, 5000),
            'Orthopedics': (1500, 12000),
            'General Practice': (300, 3000),
            'Emergency': (800, 8000),
            'Surgery': (5000, 25000),
            'Radiology': (600, 4000)
        }
        
        # Valid procedure codes per specialty
        self.valid_procedures = {
            'Cardiology': ['CARD001', 'CARD002', 'CARD003', 'CARD004'],
            'Oncology': ['ONC001', 'ONC002', 'ONC003', 'ONC004'],
            'Pediatrics': ['PED001', 'PED002', 'PED003', 'PED004'],
            'Orthopedics': ['ORTH001', 'ORTH002', 'ORTH003', 'ORTH004'],
            'General Practice': ['GP001', 'GP002', 'GP003', 'GP004'],
            'Emergency': ['ER001', 'ER002', 'ER003', 'ER004'],
            'Surgery': ['SURG001', 'SURG002', 'SURG003', 'SURG004'],
            'Radiology': ['RAD001', 'RAD002', 'RAD003', 'RAD004']
        }
        
        # Diagnosis codes per specialty
        self.diagnoses = {
            'Cardiology': ['D_HEART01', 'D_HEART02', 'D_CARDIAC03', 'D_VASCULAR04'],
            'Oncology': ['D_CANCER01', 'D_TUMOR02', 'D_LEUKEMIA03', 'D_LYMPHOMA04'],
            'Pediatrics': ['D_CHILD01', 'D_INFANT02', 'D_FEVER03', 'D_ALLERGY04'],
            'Orthopedics': ['D_BONE01', 'D_JOINT02', 'D_FRACTURE03', 'D_SPRAIN04'],
            'General Practice': ['D_ROUTINE01', 'D_CHECKUP02', 'D_FLU03', 'D_COMMON04'],
            'Emergency': ['D_ACCIDENT01', 'D_TRAUMA02', 'D_URGENT03', 'D_CRITICAL04'],
            'Surgery': ['D_SURGICAL01', 'D_OPERATION02', 'D_PROCEDURE03', 'D_INVASIVE04'],
            'Radiology': ['D_XRAY01', 'D_SCAN02', 'D_IMAGING03', 'D_MRI04']
        }
        
        # Surgery codes (for impossible scenarios)
        self.surgery_codes = ['SURG001', 'SURG002', 'SURG003', 'SURG004', 
                             'CARD003', 'ORTH003', 'ONC002']
    
    def generate_base_data(self):
        data = []
        
        # Generate providers (100 providers)
        num_providers = 100
        providers = []
        for i in range(num_providers):
            specialty = random.choice(list(self.specialties.keys()))
            providers.append({
                'provider_id': f'DR{i:03d}',
                'specialty': specialty
            })
        
        # Generate patients (500 patients)
        num_patients = 500
        patients = [f'P{i:04d}' for i in range(num_patients)]
        
        # Generate claims
        start_date = datetime.now() - timedelta(days=365)
        
        for i in range(self.num_rows):
            provider = random.choice(providers)
            patient = random.choice(patients)
            specialty = provider['specialty']
            
            # Get valid procedure and diagnosis for specialty
            procedure = random.choice(self.valid_procedures[specialty])
            diagnosis = random.choice(self.diagnoses[specialty])
            
            # Generate amount within specialty range
            min_amt, max_amt = self.specialties[specialty]
            amount = round(random.uniform(min_amt, max_amt), 2)
            
            # Random date in last year
            days_ago = random.randint(0, 365)
            claim_date = start_date + timedelta(days=days_ago)
            
            # Status (90% approved, 10% denied)
            status = 'approved' if random.random() < 0.9 else 'denied'
            
            data.append({
                'claim_id': f'C{i:05d}',
                'patient_id': patient,
                'provider_id': provider['provider_id'],
                'provider_specialty': specialty,
                'procedure_code': procedure,
                'diagnosis_code': diagnosis,
                'claim_amount': amount,
                'claim_date': claim_date.strftime('%Y-%m-%d'),
                'status': status,
                'is_fraud': False
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} base claims")
        return df
    
    def inject_duplicates(self, df, num=75):
        """Inject duplicate claim_ids"""
        # Select random claims to duplicate
        duplicate_indices = random.sample(range(len(df)), num)
        duplicate_rows = df.iloc[duplicate_indices].copy()
        duplicate_rows['is_fraud'] = True
        
        # Append duplicates
        df = pd.concat([df, duplicate_rows], ignore_index=True)
        print(f"✅ Injected {num} duplicate claims")
        return df
    
    def inject_abnormal_amounts(self, df, num=40):
        """Inject claims with abnormally high amounts"""
        # Find non-fraud claims with low amounts
        candidates = df[(df['is_fraud'] == False) & 
                       (df['claim_amount'] < 10000)].index.tolist()
        
        if len(candidates) < num:
            num = len(candidates)
        
        abnormal_indices = random.sample(candidates, num)
        
        # Set abnormally high amounts
        for idx in abnormal_indices:
            df.at[idx, 'claim_amount'] = round(random.uniform(50000, 100000), 2)
            df.at[idx, 'is_fraud'] = True
            df.at[idx, 'status'] = 'denied'  # Usually denied
        
        print(f"✅ Injected {num} abnormal amount claims")
        return df
    
    def inject_code_mismatches(self, df, num=50):
        """Inject procedure/diagnosis code mismatches"""
        candidates = df[df['is_fraud'] == False].index.tolist()
        
        if len(candidates) < num:
            num = len(candidates)
        
        mismatch_indices = random.sample(candidates, num)
        
        for idx in mismatch_indices:
            current_specialty = df.at[idx, 'provider_specialty']
            
            # Pick a different specialty
            other_specialties = [s for s in self.specialties.keys() 
                               if s != current_specialty]
            wrong_specialty = random.choice(other_specialties)
            
            # Use wrong diagnosis code
            df.at[idx, 'diagnosis_code'] = random.choice(self.diagnoses[wrong_specialty])
            df.at[idx, 'is_fraud'] = True
        
        print(f"✅ Injected {num} code mismatch claims")
        return df
    
    def inject_impossible_scenarios(self, df, num=25):
        """Same patient, multiple major surgeries same day"""
        candidates = df[df['is_fraud'] == False].index.tolist()
        
        # We need pairs, so num must be even
        if num % 2 != 0:
            num += 1
        
        if len(candidates) < num:
            num = len(candidates)
        
        impossible_indices = random.sample(candidates, num)
        
        # Process in pairs
        for i in range(0, len(impossible_indices), 2):
            if i + 1 >= len(impossible_indices):
                break
                
            idx1 = impossible_indices[i]
            idx2 = impossible_indices[i + 1]
            
            # Same patient, same date
            patient = df.at[idx1, 'patient_id']
            date = df.at[idx1, 'claim_date']
            
            df.at[idx2, 'patient_id'] = patient
            df.at[idx2, 'claim_date'] = date
            
            # Both are surgeries
            df.at[idx1, 'procedure_code'] = random.choice(self.surgery_codes)
            df.at[idx2, 'procedure_code'] = random.choice(self.surgery_codes)
            
            df.at[idx1, 'is_fraud'] = True
            df.at[idx2, 'is_fraud'] = True
        
        print(f"✅ Injected {len(impossible_indices)} impossible scenario claims")
        return df
    
    def inject_provider_outliers(self, df, num=35):
        """Provider billing 3x average for their specialty"""
        # Pick a provider to make an outlier
        outlier_providers = random.sample(df['provider_id'].unique().tolist(), 
                                         min(5, len(df['provider_id'].unique())))
        
        injected = 0
        for provider in outlier_providers:
            provider_claims = df[df['provider_id'] == provider].index.tolist()
            
            # Inflate some of their claims
            claims_to_inflate = random.sample(provider_claims, 
                                            min(num // 5, len(provider_claims)))
            
            for idx in claims_to_inflate:
                if df.at[idx, 'is_fraud']:
                    continue
                    
                # Multiply amount by 3x
                current_amount = df.at[idx, 'claim_amount']
                df.at[idx, 'claim_amount'] = round(current_amount * 3, 2)
                df.at[idx, 'is_fraud'] = True
                injected += 1
                
                if injected >= num:
                    break
            
            if injected >= num:
                break
        
        print(f"✅ Injected {injected} provider outlier claims")
        return df
    
    def inject_velocity_fraud(self, df, num=25):
        """Multiple claims for same patient in same week"""
        candidates = df[df['is_fraud'] == False].index.tolist()
        
        if len(candidates) < num:
            num = len(candidates)
        
        velocity_indices = random.sample(candidates, num)
        
        # Group into sets of 5 claims per patient
        for i in range(0, len(velocity_indices), 5):
            batch = velocity_indices[i:i+5]
            if len(batch) < 2:
                continue
            
            # Same patient
            patient = f'P{random.randint(1000, 9999)}'
            
            # Same week
            base_date = datetime.now() - timedelta(days=random.randint(0, 300))
            
            for j, idx in enumerate(batch):
                df.at[idx, 'patient_id'] = patient
                claim_date = base_date + timedelta(days=j)
                df.at[idx, 'claim_date'] = claim_date.strftime('%Y-%m-%d')
                df.at[idx, 'is_fraud'] = True
        
        print(f"✅ Injected {len(velocity_indices)} velocity fraud claims")
        return df
    
    def generate(self):
        """Main generation function"""
        print("\n" + "="*60)
        print("🏥 Generating Fraud Detection Dataset")
        print("="*60)
        
        print("\n[1/7] Generating base data...")
        df = self.generate_base_data()
        
        print("\n[2/7] Injecting duplicate claims...")
        df = self.inject_duplicates(df, num=75)
        
        print("\n[3/7] Injecting abnormal amounts...")
        df = self.inject_abnormal_amounts(df, num=40)
        
        print("\n[4/7] Injecting code mismatches...")
        df = self.inject_code_mismatches(df, num=50)
        
        print("\n[5/7] Injecting impossible scenarios...")
        df = self.inject_impossible_scenarios(df, num=24)
        
        print("\n[6/7] Injecting provider outliers...")
        df = self.inject_provider_outliers(df, num=35)
        
        print("\n[7/7] Injecting velocity fraud...")
        df = self.inject_velocity_fraud(df, num=25)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        output_path = 'data/raw/claims_data.csv'
        df.to_csv(output_path, index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Dataset Summary")
        print("="*60)
        print(f"Total claims: {len(df)}")
        
        fraud_count = df['is_fraud'].sum()
        fraud_pct = (fraud_count / len(df)) * 100
        print(f"Fraud cases: {fraud_count} ({fraud_pct:.1f}%)")
        print(f"Legitimate cases: {len(df) - fraud_count} ({100-fraud_pct:.1f}%)")
        
        print(f"\n💾 Saved to: {output_path}")
        
        # Show sample
        print("\n📋 Sample data:")
        print(df.head(10).to_string())
        
        print("\n" + "="*60)
        print("✅ Phase 2 Complete!")
        print("="*60)
        
        return df


if __name__ == "__main__":
    generator = FraudDataGenerator(num_rows=2500)
    df = generator.generate()