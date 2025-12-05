"""
Complete Data Pipeline Setup Script
Runs: Data Generation → ETL → Fraud Detection → Embeddings
"""

import os
import sys
import time

# Add src to path
sys.path.append('src')

def print_step(step_num, step_name):
    """Print formatted step header"""
    print("\n" + "="*80)
    print(f"STEP {step_num}: {step_name}")
    print("="*80 + "\n")

def main():
    print("\n🚀 MULTI-AGENT FRAUD DETECTION SYSTEM - DATA PIPELINE")
    print("=" * 80)
    print("This script will:")
    print("  1. Generate mock insurance claims data (2,575 claims)")
    print("  2. Run ETL pipeline with fraud feature engineering")
    print("  3. Execute fraud detection rules (6 rules)")
    print("  4. Generate vector embeddings for RAG")
    print("=" * 80)
    
    input("\nPress ENTER to start the pipeline...")
    
    start_time = time.time()
    
    # ============================================================================
    # STEP 1: Generate Mock Data
    # ============================================================================
    print_step(1, "Generating Mock Insurance Claims Data")
    
    try:
        from data.generator import ClaimsGenerator
        
        generator = ClaimsGenerator()
        generator.generate_claims(num_claims=2500)
        
        print("✅ Successfully generated claims data!")
        print(f"   Output: data/raw/claims_data.csv")
        
    except Exception as e:
        print(f"❌ Error in data generation: {e}")
        return
    
    # ============================================================================
    # STEP 2: Run ETL Pipeline
    # ============================================================================
    print_step(2, "Running ETL Pipeline (Feature Engineering)")
    
    try:
        from data.etl import FraudETLPipeline
        
        pipeline = FraudETLPipeline()
        pipeline.run()
        
        print("✅ Successfully completed ETL pipeline!")
        print(f"   Output: data/processed/fraud_detection.db")
        print(f"   Tables: claims, providers, patients")
        
    except Exception as e:
        print(f"❌ Error in ETL pipeline: {e}")
        return
    
    # ============================================================================
    # STEP 3: Run Fraud Detection
    # ============================================================================
    print_step(3, "Executing Fraud Detection Rules (6 Rules)")
    
    try:
        from fraud.rules import FraudDetector
        
        detector = FraudDetector()
        results_df = detector.detect_fraud()
        
        print("✅ Successfully completed fraud detection!")
        print(f"   Total claims: {len(results_df)}")
        print(f"   Fraud detected: {results_df['fraud_detected'].sum()}")
        print(f"   Fraud rate: {results_df['fraud_detected'].mean()*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error in fraud detection: {e}")
        return
    
    # ============================================================================
    # STEP 4: Generate Embeddings
    # ============================================================================
    print_step(4, "Generating Vector Embeddings for RAG")
    
    try:
        from rag.embeddings import FraudEmbeddingsGenerator
        
        embeddings_gen = FraudEmbeddingsGenerator()
        embeddings_gen.generate_embeddings()
        
        print("✅ Successfully generated embeddings!")
        print(f"   Output: data/embeddings/")
        print(f"   Files: fraud_embeddings.npy, documents.pkl, metadata.pkl")
        
    except Exception as e:
        print(f"❌ Error in embeddings generation: {e}")
        return
    
    # ============================================================================
    # Summary
    # ============================================================================
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("🎉 DATA PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Total time: {elapsed_time:.1f} seconds")
    
    print("\n📊 Database Summary:")
    import sqlite3
    conn = sqlite3.connect('data/processed/fraud_detection.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM claims")
    total_claims = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM fraud_flags WHERE fraud_detected=1")
    fraud_cases = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM providers")
    total_providers = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"   • Total Claims: {total_claims:,}")
    print(f"   • Fraud Cases: {fraud_cases:,} ({fraud_cases/total_claims*100:.1f}%)")
    print(f"   • Providers: {total_providers:,}")
    print(f"   • Patients: {total_patients:,}")
    
    print("\n🚀 Next Steps:")
    print("   1. Run the Streamlit app: streamlit run src/app/app.py")
    print("   2. Or test the orchestrator: python src/orchestrator.py")
    print("   3. Or chat with Query Agent: python src/agents/query_agent.py")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
