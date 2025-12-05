"""
Embeddings Generator for Fraud Cases
Converts fraud case data into vector embeddings for RAG
"""

import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
from pathlib import Path


class FraudEmbeddingsGenerator:
    """
    Generates embeddings for fraud cases to enable semantic search
    """
    
    def __init__(self, db_path='data/processed/fraud_detection.db', 
                 model_name='all-MiniLM-L6-v2'):
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.embeddings_path = 'data/embeddings'
        Path(self.embeddings_path).mkdir(parents=True, exist_ok=True)
    
    def create_fraud_documents(self):
        """
        Create text documents from fraud cases for embedding
        Each document contains: claim details + fraud flags + context
        """
        conn = sqlite3.connect(self.db_path)
        
        # Join claims with fraud flags
        query = """
            SELECT 
                c.claim_id,
                c.claim_amount,
                c.claim_date,
                c.provider_id,
                c.provider_specialty,
                c.procedure_code,
                c.diagnosis_code,
                c.status,
                f.fraud_detected,
                f.fraud_score,
                f.rules_triggered,
                f.explanation
            FROM claims c
            LEFT JOIN fraud_flags f ON c.claim_id = f.claim_id
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(df)} claims")
        
        # Create text documents
        documents = []
        metadata = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating documents"):
            # Create rich text representation
            doc_text = self._create_document_text(row)
            documents.append(doc_text)
            
            # Store metadata
            metadata.append({
                'claim_id': row['claim_id'],
                'fraud_detected': row['fraud_detected'],
                'fraud_score': row['fraud_score'] if pd.notna(row['fraud_score']) else 0,
                'claim_amount': row['claim_amount']
            })
        
        return documents, metadata
    
    def _create_document_text(self, row):
        """Create a rich text representation of a fraud case"""
        
        fraud_status = "FRAUD" if row['fraud_detected'] else "LEGITIMATE"
        
        text = f"""
Claim {row['claim_id']}: {fraud_status}
Amount: ${row['claim_amount']:,.2f}
Date: {row['claim_date']}
Provider: {row['provider_id']} ({row['provider_specialty']})
Procedure: {row['procedure_code']}
Diagnosis: {row['diagnosis_code']}
Status: {row['status']}
"""
        
        if row['fraud_detected']:
            text += f"""
Fraud Score: {row['fraud_score']}/100
Rules Triggered: {row['rules_triggered']}
Fraud Explanation: {row['explanation']}
"""
        
        return text.strip()
    
    def generate_embeddings(self):
        """Generate embeddings for all fraud cases"""
        print("\n🔄 Generating Embeddings for Fraud Cases\n")
        print("="*60)
        
        # Create documents
        documents, metadata = self.create_fraud_documents()
        
        # Generate embeddings
        print(f"\n📊 Embedding {len(documents)} documents...")
        embeddings = self.model.encode(
            documents, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Save embeddings and metadata
        self._save_embeddings(embeddings, documents, metadata)
        
        print(f"\n✅ Embeddings saved to {self.embeddings_path}")
        print(f"   - Shape: {embeddings.shape}")
        print(f"   - Dimension: {embeddings.shape[1]}")
        print("="*60)
        
        return embeddings, documents, metadata
    
    def _save_embeddings(self, embeddings, documents, metadata):
        """Save embeddings and metadata to disk"""
        
        # Save embeddings as numpy array
        np.save(
            f'{self.embeddings_path}/fraud_embeddings.npy', 
            embeddings
        )
        
        # Save documents
        with open(f'{self.embeddings_path}/documents.pkl', 'wb') as f:
            pickle.dump(documents, f)
        
        # Save metadata
        with open(f'{self.embeddings_path}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_embeddings(self):
        """Load previously generated embeddings"""
        embeddings = np.load(f'{self.embeddings_path}/fraud_embeddings.npy')
        
        with open(f'{self.embeddings_path}/documents.pkl', 'rb') as f:
            documents = pickle.load(f)
        
        with open(f'{self.embeddings_path}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"✅ Loaded {len(documents)} embeddings")
        return embeddings, documents, metadata


if __name__ == "__main__":
    print("\n🚀 Starting Embeddings Generation\n")
    
    # Generate embeddings
    generator = FraudEmbeddingsGenerator()
    embeddings, documents, metadata = generator.generate_embeddings()
    
    print("\n✅ Embeddings generation complete!")
