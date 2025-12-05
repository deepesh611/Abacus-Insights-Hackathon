"""
SQL Query Agent - Converts natural language to SQL queries
Uses LLM to generate SQL for statistical questions about fraud data
"""

import sqlite3
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import LLMClient


class SQLQueryAgent:
    """
    Converts natural language questions to SQL queries
    Executes them against the fraud detection database
    """
    
    def __init__(self, db_path='data/processed/fraud_detection.db'):
        self.db_path = db_path
        self.llm = LLMClient()
        self.schema = self._get_schema()
    
    def _get_schema(self):
        """Get database schema for context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema_info = []
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            schema_info.append({
                'table': table_name,
                'columns': [(col[1], col[2]) for col in columns]  # (name, type)
            })
        
        conn.close()
        return schema_info
    
    def question_to_sql(self, question):
        """Convert natural language question to SQL query"""
        
        # Build schema description for LLM
        schema_text = "DATABASE SCHEMA:\n\n"
        for table in self.schema:
            schema_text += f"Table: {table['table']}\n"
            for col_name, col_type in table['columns']:
                schema_text += f"  - {col_name} ({col_type})\n"
            schema_text += "\n"
        
        prompt = f"""{schema_text}

QUESTION: {question}

IMPORTANT RULES:
1. Generate ONLY the SQL query, no explanation
2. Use COUNT(*), SUM(), AVG(), etc. for statistics
3. Join claims table with fraud_flags table on claim_id when needed
4. fraud_detected=1 means fraud case
5. Return ONLY executable SQL (SELECT statement)
6. Do NOT use markdown code blocks or formatting
7. The query must be a single line

Example questions and queries:
- "How many fraud cases?" -> SELECT COUNT(*) FROM fraud_flags WHERE fraud_detected=1
- "Fraud cases over $50k?" -> SELECT COUNT(*) FROM claims c JOIN fraud_flags f ON c.claim_id = f.claim_id WHERE f.fraud_detected=1 AND c.claim_amount > 50000
- "What's the total fraud amount?" -> SELECT SUM(c.claim_amount) FROM claims c JOIN fraud_flags f ON c.claim_id = f.claim_id WHERE f.fraud_detected=1

Generate SQL query for the question:"""

        sql_query = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        
        # Clean up the query (remove markdown, extra whitespace)
        sql_query = sql_query.strip()
        sql_query = sql_query.replace('```sql', '').replace('```', '')
        sql_query = sql_query.strip()
        
        return sql_query
    
    def execute_query(self, question):
        """Convert question to SQL and execute it"""
        try:
            # Generate SQL query
            sql_query = self.question_to_sql(question)
            
            print(f"Generated SQL: {sql_query}")
            
            # Execute query
            conn = sqlite3.connect(self.db_path)
            result = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            return {
                'success': True,
                'sql': sql_query,
                'result': result,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'sql': sql_query if 'sql_query' in locals() else None,
                'result': None,
                'error': str(e)
            }


if __name__ == "__main__":
    print("\n🔍 Testing SQL Query Agent\n")
    
    agent = SQLQueryAgent()
    
    # Test questions
    test_questions = [
        "How many fraud cases are there?",
        "How many fraud cases involve amounts over $50,000?",
        "What is the total amount of fraudulent claims?",
        "How many fraud cases involve amounts over $5,000?",
        "Which provider has the most fraud cases?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = agent.execute_query(question)
        
        if result['success']:
            print(f"SQL: {result['sql']}")
            print(f"Result:\n{result['result']}")
        else:
            print(f"Error: {result['error']}")
        print("-" * 80)
