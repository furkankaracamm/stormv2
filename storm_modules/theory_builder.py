"""
Communication Theory Database Builder
Uses LLM Gateway for unified LLM access with fallback chain.
"""
import json
import sqlite3
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
                if self._save_theory_to_db(theory_name, theory_data):
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
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM theories")
            rows = cursor.fetchall()
            existing = {r[0] for r in rows}
            conn.close()
        except sqlite3.Error as e:
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
                # Clean and parse JSON
                content = result.replace('```json', '').replace('```', '').strip()
                return self._parse_json_safe(content)
                
        except Exception as e:
            print(f"  ⚠ LLM Error: {e}")
        
        return None
    
    def _parse_json_safe(self, content: str) -> Optional[Dict]:
        """Parse JSON with multiple fallback strategies."""
        # Strategy 1: Direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Find JSON object in content
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Fix common issues
        try:
            fixed = content.replace("'", '"').replace('\n', ' ')
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parse failed: {e}")
        
        return None
    
    def _save_theory_to_db(self, theory_name: str, theory_data: Dict) -> bool:
        """Save theory to SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO theories
                (name, core_propositions, key_concepts, typical_hypotheses, 
                 typical_methods, boundary_conditions, digital_application)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                theory_name,
                json.dumps(theory_data.get('core_propositions', [])),
                json.dumps(theory_data.get('key_concepts', [])),
                json.dumps(theory_data.get('typical_hypotheses', [])),
                json.dumps(theory_data.get('typical_methods', {})),
                json.dumps(theory_data.get('boundary_conditions', [])),
                theory_data.get('digital_application', '')
            ))
            conn.commit()
            conn.close()
            return True
        except sqlite3.Error as e:
            print(f"  ⚠ DB error: {e}")
            return False


if __name__ == "__main__":
    builder = TheoryDatabaseBuilder()
    builder.build_full_database()
