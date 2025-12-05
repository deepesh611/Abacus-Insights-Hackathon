"""
Query Agent - RAG-Powered Fraud Q&A System
Answers natural language questions about fraud cases using retrieval and LLM
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import FraudVectorStore
from utils.llm_client import LLMClient


class QueryAgent:
    """
    RAG-powered agent that answers questions about fraud cases
    Retrieves relevant cases and uses LLM to generate answers
    """
    
    def __init__(self, db_path='data/processed/fraud_detection.db'):
        self.vector_store = FraudVectorStore()
        self.vector_store.load_index()
        self.llm = LLMClient()
        self.db_path = db_path
    
    def answer_question(self, question, k=10):
        """
        Answer a question about fraud cases using RAG + Database for statistics
        
        Args:
            question: Natural language question
            k: Number of cases to retrieve for context
        
        Returns:
            Dictionary with answer and sources
        """
        print(f"\n❓ Question: {question}\n")
        
        # Check if it's a statistical or list query
        is_stats_query = any(word in question.lower() for word in 
                            ['how many', 'count', 'total', 'number of', 'statistics', 'how much', 
                             'list all', 'show all', 'what are the', 'which'])
        
        # Get database statistics if it's a stats/list query
        db_stats = self._get_database_stats(question) if is_stats_query else None
        
        # 1. Retrieve relevant cases (for context/examples)
        results = self.vector_store.search(question, k=k, fraud_only=True)
        
        # 2. Build context from retrieved cases
        context = self._build_context(results)
        
        # 3. Generate answer using LLM
        prompt = self._build_prompt(question, context, results, db_stats)
        answer = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
        
        # 4. Return answer with sources
        return {
            'question': question,
            'answer': answer,
            'sources': [r['metadata']['claim_id'] for r in results]
        }
    
    def _get_database_stats(self, question=""):
        """Get statistics from database using dynamic SQL generation"""
        from agents.sql_agent import SQLQueryAgent
        
        sql_agent = SQLQueryAgent(self.db_path)
        
        stats = {}
        
        # Use SQL agent to dynamically generate and execute query
        result = sql_agent.execute_query(question)
        
        if result['success']:
            # Store the SQL query and result
            stats['sql_query'] = result['sql']
            stats['sql_result'] = result['result']
            
            # Extract the count/value from result
            if len(result['result']) > 0 and len(result['result'].columns) > 0:
                # Get the first value from the first row
                stats['answer'] = result['result'].iloc[0, 0]
        else:
            # Fallback: get basic stats if SQL generation fails
            import sqlite3
            import pandas as pd
            
            conn = sqlite3.connect(self.db_path)
            stats['total_fraud_cases'] = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM fraud_flags WHERE fraud_detected=1", conn
            )['count'][0]
            conn.close()
            stats['sql_error'] = result['error']
        
        return stats
    
    def _build_context(self, results):
        """Build context string from retrieved results"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"Case {i}:\n{result['document']}\n")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question, context, results, db_stats=None):
        """Build prompt for LLM"""
        
        stats_section = ""
        if db_stats:
            if 'sql_query' in db_stats and 'sql_result' in db_stats:
                # Dynamic SQL was generated and executed
                result_df = db_stats['sql_result']
                
                # Format result based on type
                if len(result_df) > 0:
                    if len(result_df) == 1 and len(result_df.columns) == 1:
                        # Single value (count/sum)
                        result_value = result_df.iloc[0, 0]
                        stats_section = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATABASE QUERY RESULT (EXECUTED ON FULL DATABASE):

SQL Query: {db_stats['sql_query']}
Result: **{result_value}**

This is the EXACT answer from querying the complete database.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
                    else:
                        # Multiple rows (list)
                        result_lines = []
                        for idx, row in result_df.iterrows():
                            # Get first column value
                            value = row.iloc[0]
                            result_lines.append(f"  • {value}")
                        
                        result_list = "\n".join(result_lines[:20])  # Limit to 20 items
                        total_count = len(result_df)
                        
                        stats_section = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATABASE QUERY RESULT (EXECUTED ON FULL DATABASE):

SQL Query: {db_stats['sql_query']}

Results ({total_count} total):
{result_list}
{'... (showing first 20)' if total_count > 20 else ''}

This is the COMPLETE LIST from the database.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            elif 'total_fraud_cases' in db_stats:
                # Fallback stats
                stats_section = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATABASE STATISTICS:
- Total fraud cases: {db_stats['total_fraud_cases']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        prompt = f"""You are an expert fraud analyst. Answer the following question based on the provided fraud case data.

QUESTION:
{question}

{stats_section}RELEVANT FRAUD CASES (EXAMPLES ONLY - NOT FOR COUNTING):
{context}

CRITICAL INSTRUCTIONS:
1. **If DATABASE QUERY RESULT is shown above, that is the EXACT ANSWER from the full database**
2. For "list all" questions, use the complete list shown above, NOT the examples below
3. For "how many" questions, use the count from the database result
4. The {len(results)} fraud cases listed are EXAMPLES to provide context
5. You can reference specific claim IDs from examples to illustrate patterns
6. Keep your answer concise and professional

Your answer:"""
        
        return prompt
    
    def chat(self):
        """Interactive chat mode"""
        print("\n💬 Fraud Detection Q&A System")
        print("="*60)
        print("Ask me anything about the fraud cases!")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Get answer
            result = self.answer_question(question)
            
            print(f"\n🤖 Answer: {result['answer']}")
            print(f"\n📚 Sources: {', '.join(result['sources'])}\n")
            print("-"*60 + "\n")


# ===========================
# Example Questions
# ===========================
EXAMPLE_QUESTIONS = [
    "What are the most common fraud patterns?",
    "Show me claims with duplicate IDs",
    "Which providers have the most fraud flags?",
    "What is the highest fraud score?",
    "Tell me about impossible scenario frauds",
    "How many fraud cases involve amounts over $50,000?",
]


if __name__ == "__main__":
    print("\n🤖 Starting Query Agent\n")
    
    # Initialize agent
    agent = QueryAgent()
    
    # Show example questions
    print("\n📋 Example Questions:")
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"  {i}. {q}")
    
    # Answer a few examples
    print("\n" + "="*80)
    print("Testing with sample questions...")
    print("="*80)
    
    for question in EXAMPLE_QUESTIONS[:3]:
        result = agent.answer_question(question, k=5)
        print(f"\n🤖 {result['answer']}")
        print(f"   Sources: {', '.join(result['sources'][:3])}")
        print("-"*80)
    
    # Uncomment to start interactive mode:
    # agent.chat()
