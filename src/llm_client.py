import requests
import json

class LocalLLMClient:
    def __init__(self, base_url="http://localhost:9091/v1", model="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"):
        self.base_url = base_url
        self.model = model
        self.headers = {"Content-Type": "application/json"}
    
    def call_llm(self, prompt, max_tokens=800, temperature=0):
        """Make API call to local LLM"""
        try:
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert at analyzing question papers and identifying patterns. Be precise and follow instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"LLM API error: {e}")
            return ""