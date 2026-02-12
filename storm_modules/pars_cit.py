from __future__ import annotations

import re
from typing import Dict, List


class ParsCitExtractor:
    def is_available(self) -> bool:
        return True

    def parse_citations_from_text(self, text: str) -> List[Dict]:
        pattern = re.compile(r"\(([A-Z][A-Za-z\s.&-]+),\s*(\d{4})\)")
        citations: List[Dict] = []
        for match in pattern.finditer(text):
            author = match.group(1).strip()
            year = int(match.group(2))
            citations.append(
                {
                    "author": author,
                    "year": year,
                    "raw_text": match.group(0),
                }
            )
        return citations

    def dry_run(self) -> Dict:
        sample = "Recent work shows (Smith, 2021) and (Doe, 2019)."
        return {"source": "parscit", "status": "dry-run", "citations": self.parse_citations_from_text(sample)}
