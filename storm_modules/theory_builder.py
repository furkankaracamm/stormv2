"""
Communication Theory Database Builder
Uses LLM Gateway for unified LLM access with fallback chain.
"""
import json

from typing import Dict, List, Optional
from storm_modules.config import get_academic_brain_db_path
from storm_modules.llm_gateway import get_llm_gateway


class TheoryDatabaseBuilder:
    """Builds comprehensive theory profiles using LLM."""
    
    THEORY_LIST = [
        'Agenda Setting Theory', 'Uses and Gratifications Theory',
        'Social Cognitive Theory', 'Cultivation Theory', 'Framing Theory',
        'Elaboration Likelihood Model', 'Diffusion of Innovations',
        'Social Identity Theory', 'Spiral of Silence', 'Third Person Effect',
        'Media Dependency Theory', 'Selective Exposure Theory',
        'Cognitive Dissonance Theory', 'Parasocial Interaction Theory',
        'Media Richness Theory', 'Social Presence Theory',
        'Filter Bubble Theory', 'Echo Chamber Theory',
        'Algorithmic Accountability Theory', 'Platform Studies Framework',
        'Participatory Culture Theory', 'Spreadable Media Theory',
        'Affordance Theory', 'Actor Network Theory',
        'Mediation Theory', 'Mediatization Theory',
        'Network Theory', 'Social Information Processing Theory',
        'Hyperpersonal Communication Theory', 'Priming Theory',
        'Dead Internet Theory'
    ]
    
    def __init__(self):
        """Initialize with LLM Gateway instead of direct Ollama."""
        self.llm = get_llm_gateway()
        self.db_path = str(get_academic_brain_db_path())
        self.theories = {}
        from storm_modules.theory_manager import TheoryVersionManager
        self.version_manager = TheoryVersionManager()
        from storm_modules.theory_validator import TheoryValidator
        self.validator = TheoryValidator()
    
    def build_full_database(self) -> Dict[str, int]:
        """Build/update theory database. Returns stats."""
        print("=" * 60)
        print("THEORY DATABASE BUILDER (LLM GATEWAY)")
        print("=" * 60)
        
        stats = {'skipped': 0, 'added': 0, 'failed': 0}
        
        # Load existing theories from DB first
        existing_theories = self._get_existing_theories()
        print(f"[INFO] Found {len(existing_theories)} existing theories in DB.")
        print(f"[INFO] LLM Provider: {self.llm.provider.upper() if self.llm.provider else 'NONE'}")

        for i, theory_name in enumerate(self.THEORY_LIST, 1):
            if theory_name in existing_theories:
                print(f"[{i}/{len(self.THEORY_LIST)}] {theory_name} -> SKIPPED")
                stats['skipped'] += 1
                continue

            print(f"\n[{i}/{len(self.THEORY_LIST)}] {theory_name} -> LEARNING...")
            theory_data = self.enrich_theory(theory_name)
            
            if theory_data:
                self.theories[theory_name] = theory_data
                if self.version_manager.create_or_update_theory(theory_name, theory_data, "Automated generation"):
                    print(f"  ✓ Saved")
                    stats['added'] += 1
                else:
                    stats['failed'] += 1
            else:
                print(f"  ⚠ Failed to generate theory profile")
                stats['failed'] += 1
        
        print(f"\n{'='*60}")
        print(f"COMPLETE: Added={stats['added']}, Skipped={stats['skipped']}, Failed={stats['failed']}")
        return stats
    
    def _get_existing_theories(self) -> set:
        """Get set of existing theory names from DB."""
        existing = set()
        from storm_modules.db_safety import get_db_connection, DatabaseError
        
        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.execute("SELECT name FROM theories")
                rows = cursor.fetchall()
                existing = {r[0] for r in rows}
        except DatabaseError as e:
            print(f"[DB WARNING] Could not load existing theories: {e}")
        return existing
    
    def enrich_theory(self, theory_name: str) -> Optional[Dict]:
        """Generate comprehensive theory profile using LLM Gateway."""
        if not self.llm.is_available():
            print("  ⚠ No LLM provider available")
            return None
        
        prompt = f"""You are an expert communication researcher.

Create comprehensive profile for: {theory_name}

Output ONLY valid JSON:
{{
  "name": "{theory_name}",
  "core_propositions": ["Proposition 1", "Proposition 2", "Proposition 3"],
  "key_concepts": [
    {{"name": "Concept", "definition": "Clear definition", "measurement": "How measured"}}
  ],
  "typical_hypotheses": ["H1 template", "H2 template"],
  "typical_methods": {{
    "design_type": "survey/experiment",
    "sample_size_norm": "300-500",
    "common_tests": ["regression", "ANOVA"],
    "common_measures": ["Scale Name (Author, Year)"]
  }},
  "boundary_conditions": ["Condition 1", "Condition 2"],
  "digital_application": "How applies to social media, bots, algorithms"
}}

NO preamble, ONLY JSON."""

        try:
            # Use LLM Gateway with JSON mode
            result = self.llm.generate(
                prompt, 
                max_tokens=2000, 
                temperature=0.3, 
                json_mode=True
            )
            
            if result:
                # 1. Schema Validation
                from storm_modules.validation_models import TheoryProfile, validate_llm_output
                validated = validate_llm_output(TheoryProfile, result)
                if not validated:
                    print(f"  ⚠ Schema Validation Failed")
                    return None
                
                print(f"  ✓ Schema Validation Passed")
                
                # 2. Quality Validation
                is_valid, errors, score = self.validator.validate_theory(validated.dict())
                
                if not is_valid:
                    print(f"  ⚠ Quality Validation Failed (Score: {score:.2f})")
                    for err in errors:
                        print(f"    - {err}")
                    return None
                
                print(f"  ✓ Quality Validation Passed (Score: {score:.2f})")
                return validated.dict()
                
        except Exception as e:
            print(f"  ⚠ LLM Error: {e}")
        
        return None
    

    



if __name__ == "__main__":
    builder = TheoryDatabaseBuilder()
    builder.build_full_database()
