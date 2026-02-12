"""S2ORC / Semantic Scholar Client - Fallback search and full-text metadata"""
import requests
import os
from typing import List, Dict, Optional

class S2ORCClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("S2_API_KEY")
        self.headers = {"x-api-key": self.api_key} if self.api_key else {}

    def search_papers(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for papers on Semantic Scholar"""
        endpoint = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,citationCount,externalIds,openAccessPdf"
        }
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers, timeout=30)
            if response.status_code == 200:
                return response.json().get('data', [])
        except Exception as e:
            print(f"[S2ORC Error] {e}")
            
        return []

    def get_paper_details(self, paper_id: str) -> Optional[Dict]:
        """Fetch full details and citations for a paper"""
        endpoint = f"{self.BASE_URL}/paper/{paper_id}"
        params = {"fields": "title,abstract,citations,references,embedding,tldr"}
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers, timeout=30)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

if __name__ == "__main__":
    client = S2ORCClient()
    results = client.search_papers("social cognitive theory online behavior")
    for r in results:
        print(f"- {r.get('title')} ({r.get('year')}) [Citations: {r.get('citationCount')}]")
