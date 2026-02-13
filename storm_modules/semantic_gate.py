"""
STORM Semantic Quality Gate
Advisory filter using Semantic Scholar metadata to ensure high-leverage downloads.
"""

import requests
import time
from typing import Optional, Dict, List
from storm_modules.rate_limiter import rate_limited

class SemanticQualityGate:
    def __init__(self):
        self.api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.headers = {"User-Agent": "STORM-Bot/1.0 (Academic Research; mailto:researcher@storm.io)"}
        
        # ALLOWED FIELDS (Broad Academic Scope)
        self.allowed_fields = {
            "Computer Science", 
            "Sociology", 
            "Philosophy", 
            "Psychology", 
            "Political Science",
            "Communication",
            "Art",
            "Engineering" # For algorithmic/network contexts
        }
        
        # BLOCKED FIELDS (Strict Irrelevance)
        # Rejection only occurs if paper is EXCLUSIVELY in these fields
        self.suspicious_fields = {
            "Medicine",
            "Biology",
            "Chemistry",
            "Geology",
            "Agricultural and Food Sciences"
        }

    @rate_limited("semantic_scholar")
    def check_quality(self, title: str) -> bool:
        """
        Consults Semantic Scholar to validate paper relevance.
        
        Returns:
            True: Paper is good (or unknown) -> DOWNLOAD
            False: Paper is confirmed irrelevant -> SKIP
        
        Failure Policy: OPEN (On error/miss, return True)
        """
        if not title or len(title) < 5:
            return True # Malformed title, let downloader handle or fail naturally

        try:
            # 1. Query API (Limit 1, Fields needed for decision only)
            params = {
                "query": title,
                "limit": 1,
                "fields": "title,fieldsOfStudy,citationCount"
            }
            
            # Rate limit protection (handled by decorator)
            
            r = requests.get(self.api_url, params=params, headers=self.headers, timeout=5)
            
            if r.status_code != 200:
                return True # API Error -> Fail Open
                
            data = r.json()
            if not data.get("data"):
                return True # Paper not found -> Fail Open (Could be very new)
                
            paper = data["data"][0]
            
            # 2. Field Analysis
            fields = paper.get("fieldsOfStudy")
            if not fields:
                return True # No field data -> Fail Open
                
            field_set = set(fields)
            
            # 3. Domain Check
            # If valid domain present -> PASS
            if not field_set.isdisjoint(self.allowed_fields):
                return True
                
            # If ONLY suspicious fields present -> REJECT
            # (e.g., Pure Medical paper)
            if field_set.issubset(self.suspicious_fields):
                print(f"[QUALITY GATE] Rejecting '{title[:30]}...' -> Fields: {list(field_set)}")
                return False
                
            # 4. Citation Check (Optional/Weak signal)
            # We do NOT block on low citations (could be new), 
            # but we could theoretically use it here. 
            # Implementing "Quality" as "Relevance" primarily.
                
            return True # Default to allow if ambiguous

        except Exception as e:
            # LOG BUT DO NOT CRASH
            print(f"[QUALITY GATE] API Warning: {e}")
            return True
