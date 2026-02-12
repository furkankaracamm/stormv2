"""Keyphrase Extractor - Extracts academic keywords from text using local LLM"""
import requests
import json
from typing import List, Optional

class KeyphraseExtractor:
    def __init__(self, llm_url: str = "http://localhost:11434/api/generate"):
        self.llm_url = llm_url

    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extracts top academic keywords/concepts from a given text (abstract/title)"""
        prompt = f"""Extract {max_keywords} most important academic keywords or concepts from this text. 
Focus on research methods, theories, and core variables.
Output ONLY the keywords separated by commas.

TEXT:
{text[:2000]}

KEYWORDS:"""

        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": "llama3", 
                    "prompt": prompt, 
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 100}
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json().get('response', '')
                # Clean and parse
                keywords = [k.strip() for k in result.split(',') if k.strip()]
                return keywords[:max_keywords]
        except Exception as e:
            print(f"[Keyphrase Error] {e}")
            
        return []

if __name__ == "__main__":
    extractor = KeyphraseExtractor()
    test_text = "This study investigates the impact of algorithmic filtering on social media users' perception of reality, focusing on the filter bubble phenomenon and cognitive dissonance."
    keywords = extractor.extract_keywords(test_text)
    print(f"Keywords: {keywords}")
