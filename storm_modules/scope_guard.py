"""
STORM Scope Guard
Strictly enforces topic boundaries for the downloader subsystem.
"""

from typing import List, Set
from storm_modules.ontology import ResearchOntology

class ScopeGuard:
    def __init__(self, additional_terms: List[str] = None):
        self.ontology = ResearchOntology()
        self.allowed_terms = self._build_allowlist(additional_terms)
        
        # Core axioms that satisfy scope (The "Constitution")
        self.core_axioms = [
            "dead internet theory",
            "simulacra",
            "simulation",
            "hyperreality",
            "algorithmic",
            "bot",
            "artificial intelligence",
            "social media",
            "disinformation",
            "propaganda",
            "surveillance",
            "platform"
        ]

    def _build_allowlist(self, additional: List[str] = None) -> Set[str]:
        """Builds a set of normalized allowed phrases from authoritative sources."""
        terms = set()
        
        # 1. Ontology Terms
        for topic in self.ontology.get_all_topics():
            terms.add(topic.lower())
            
        # 2. Additional Trusted Terms (Seeds, Priority Books)
        if additional:
            for term in additional:
                terms.add(term.lower())
                
        return terms

    def is_safe(self, text: str) -> bool:
        """
        Determines if the text is within the allowable research scope.
        
        Rule:
        1. Must contain at least one CORE AXIOM 
           OR
        2. Must contain at least one ONTOLOGY TERM
        """
        if not text:
            return False
            
        text_lower = text.lower()
        
        # Check Core Axioms
        for axiom in self.core_axioms:
            if axiom in text_lower:
                return True
                
        # Check Ontology
        for term in self.allowed_terms:
            if term in text_lower:
                return True
                
        return False
