"""iCite Client - Fetches Relative Citation Ratio (RCR) and impact metrics from NIH"""
import requests
import time
from typing import Dict, List, Optional, Union

class ICiteClient:
    BASE_URL = "https://icite.od.nih.gov/api/v1/pubs"
    
    def __init__(self, rate_limit_delay: float = 0.5):
        self.rate_limit_delay = rate_limit_delay

    def fetch_metrics(self, pmids_or_dois: Union[str, List[str]]) -> List[Dict]:
        """
        Fetches metrics for a list of PMIDs or DOIs.
        Limited to 1000 items per request by iCite API.
        """
        if isinstance(pmids_or_dois, str):
            query_param = pmids_or_dois
        else:
            query_param = ",".join(pmids_or_dois[:1000])
        
        try:
            response = requests.get(f"{self.BASE_URL}?pmids={query_param}", timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            elif response.status_code == 404: # Try DOI search if PMID search failed for single item
                response = requests.get(f"{self.BASE_URL}?dois={query_param}", timeout=30)
                if response.status_code == 200:
                    return response.json().get('data', [])
        except Exception as e:
            print(f"[iCite Error] {e}")
        
        return []

    def get_impact_score(self, identifier: str) -> float:
        """
        Returns the Relative Citation Ratio (RCR) for a paper.
        RCR > 1.0 means higher than average impact.
        """
        results = self.fetch_metrics(identifier)
        if results:
            # We take the first result's relative_citation_ratio
            return results[0].get('relative_citation_ratio', 0.0)
        return 0.0

if __name__ == "__main__":
    # Test with a known high-impact PMID or DOI
    client = ICiteClient()
    # Example PMID for a major paper
    test_pmid = "22513261" 
    metrics = client.fetch_metrics(test_pmid)
    if metrics:
        print(f"Paper: {metrics[0].get('title')}")
        print(f"RCR Score: {metrics[0].get('relative_citation_ratio')}")
    else:
        print("No metrics found.")
