from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class OpenAlexConfig:
    base_url: str = "https://api.openalex.org"
    mailto: Optional[str] = None


class OpenAlexClient:
    def __init__(self, config: OpenAlexConfig | None = None) -> None:
        self.config = config or OpenAlexConfig()

    def is_available(self) -> bool:
        try:
            import requests  # noqa: F401
        except ImportError:
            return False
        return True

    def fetch_works(self, query: str, per_page: int = 5) -> List[Dict]:
        if not self.is_available():
            raise RuntimeError("requests is required to call OpenAlex")

        import requests

        params = {"search": query, "per-page": per_page}
        if self.config.mailto:
            params["mailto"] = self.config.mailto
        response = requests.get(f"{self.config.base_url}/works", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return payload.get("results", [])

    def resolve_doi(self, doi: str) -> Optional[str]:
        """Resolve a DOI to a direct PDF URL using OpenAlex."""
        if not self.is_available():
            return None
            
        import requests
        
        # Clean DOI
        clean_doi = doi.replace("https://doi.org/", "").strip()
        url = f"{self.config.base_url}/works/https://doi.org/{clean_doi}"
        
        try:
            params = {}
            if self.config.mailto:
                params["mailto"] = self.config.mailto
                
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            
            data = response.json()
            open_access = data.get("open_access", {})
            
            # 1. Best OA Location
            if open_access.get("is_oa"):
                return open_access.get("oa_url")
            
            # 2. Check alternate locations
            for loc in data.get("locations", []):
                if loc.get("is_oa") and loc.get("pdf_url"):
                    return loc.get("pdf_url")
                    
        except Exception as e:
            print(f"[OpenAlex] Resolution failed for {doi}: {e}")
            
        return None

    def dry_run(self) -> Dict:
        return {
            "source": "openalex",
            "status": "dry-run",
            "query": "dead internet theory",
            "example_result": {
                "title": "Synthetic example",
                "year": 2024,
                "doi": None,
            },
        }
