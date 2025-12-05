from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = os.getenv("DEFAULT_MODEL", "openai/gpt-3.5-turbo")
    
    def chat(self, messages, temperature=0.7):
        """Send chat completion request"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def analyze_fraud(self, claim_data, context):
        """Specialized method for fraud analysis"""
        prompt = f"""You are a fraud detection expert. Analyze this claim:
        
Claim Data: {claim_data}
Context: {context}

Provide:
1. Fraud likelihood score (1-10)
2. Key red flags
3. Investigation priority (Low/Medium/High)
"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=0.3)