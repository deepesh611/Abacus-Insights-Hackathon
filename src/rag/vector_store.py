"""
Vector Store using FAISS for Fraud Case Search
Enables semantic search over fraud cases
"""

import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


class FraudVectorStore:
    """
    FAISS-based vector store for semantic search over fraud cases
    """
    
    def __init__(self, embeddings_path='data/embeddings'):
        self.embeddings_path = embeddings_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = None
        self.metadata = None
    
    def load_index(self):
        """Load FAISS index and associated data"""
        # Load embeddings
        embeddings = np.load(f'{self.embeddings_path}/fraud_embeddings.npy')
        
        # Load documents
        with open(f'{self.embeddings_path}/documents.pkl', 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load metadata
        with open(f'{self.embeddings_path}/metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✅ Loaded vector store with {len(self.documents)} documents")
        return self
    
    def search(self, query, k=5, fraud_only=False):
        """
        Search for most relevant fraud cases
        
        Args:
            query: Natural language query
            k: Number of results to return
            fraud_only: If True, only return fraud cases
        
        Returns:
            List of (document, metadata, distance) tuples
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            k=k*5 if fraud_only else k  # Get more if filtering
        )
        
        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if fraud_only and not self.metadata[idx]['fraud_detected']:
                continue
            
            results.append({
                'document': self.documents[idx],
                'metadata': self.metadata[idx],
                'distance': float(dist)
            })
            
            if len(results) >= k:
                break
        
        return results
    
    def get_fraud_statistics(self):
        """Get statistics about fraud cases in the vector store"""
        total = len(self.metadata)
        fraud_cases = sum(1 for m in self.metadata if m['fraud_detected'])
        
        return {
            'total_cases': total,
            'fraud_cases': fraud_cases,
            'legitimate_cases': total - fraud_cases,
            'fraud_percentage': fraud_cases / total * 100
        }


if __name__ == "__main__":
    print("\n🔍 Testing Vector Store\n")
    
    # Load vector store
    vector_store = FraudVectorStore()
    vector_store.load_index()
    
    # Test searches
    test_queries = [
        "duplicate claims",
        "high amount fraud",
        "provider outliers",
        "impossible scenarios"
    ]
    
    for query in test_queries:
        print(f"\n📊 Query: '{query}'")
        results = vector_store.search(query, k=3, fraud_only=True)
        
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (distance: {result['distance']:.3f}):")
            print(f"  Claim: {result['metadata']['claim_id']}")
            print(f"  Fraud Score: {result['metadata']['fraud_score']}/100")
    
    # Statistics
    stats = vector_store.get_fraud_statistics()
    print(f"\n📈 Statistics:")
    print(f"  Total: {stats['total_cases']}")
    print(f"  Fraud: {stats['fraud_cases']} ({stats['fraud_percentage']:.1f}%)")
