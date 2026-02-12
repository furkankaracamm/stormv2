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
