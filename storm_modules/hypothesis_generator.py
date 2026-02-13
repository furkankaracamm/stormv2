"""Hypothesis Generator - Theory + Gap → Testable Hypotheses"""
import json

from typing import List, Dict
from storm_modules.config import get_academic_brain_db_path
from storm_modules.llm_gateway import get_llm_gateway

class HypothesisGenerator:
    def __init__(self):
        self.llm = get_llm_gateway()
        self.db_path = str(get_academic_brain_db_path())
        self.theories = self._load_theories()
    
    def _load_theories(self) -> Dict:
        from storm_modules.db_safety import get_db_connection
        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.execute("SELECT name, core_propositions, typical_hypotheses FROM theories")
                rows = cursor.fetchall()
                theories = {}
                for row in rows:
                    theories[row[0]] = {
                        'core_propositions': json.loads(row[1]),
                        'typical_hypotheses': json.loads(row[2])
                    }
            return theories
        except (Exception, json.JSONDecodeError) as e:
            print(f"[THEORY LOAD ERROR] {e}")
            return {}
    
    def generate_hypotheses(self, gap_description: str, theory_name: str, num: int = 5) -> List[Dict]:
        if not self.llm.is_available():
            return []
        
        theory = self.theories.get(theory_name, {})
        if not theory:
            return []
        
        prompt = f"""Generate {num} testable hypotheses.

GAP: {gap_description}
THEORY: {theory_name}
PROPOSITIONS: {theory.get('core_propositions', [])}

Output ONLY valid JSON:
{{
  "hypotheses": [
    {{
      "id": "H1",
      "statement": "Clear testable statement",
      "type": "main_effect",
      "IV": "independent_var",
      "DV": "dependent_var",
      "expected_direction": "positive/negative",
      "expected_effect_size": 0.25,
      "theory_basis": "Which proposition supports this"
    }},
    {{
      "id": "H2",
      "statement": "Moderation hypothesis",
      "type": "moderation",
      "IV": "var", "DV": "var", "moderator": "mod_var",
      "theory_basis": "Basis"
    }}
  ]
}}"""

        try:
            result = self.llm.generate(prompt, max_tokens=2000, temperature=0.4, json_mode=True)
            
            if result:
                from storm_modules.validation_models import HypothesisSet, validate_llm_output
                validated = validate_llm_output(HypothesisSet, result)
                
                if validated:
                    hypotheses = [h.dict() for h in validated.hypotheses]
                    self._save_hypotheses(gap_description, theory_name, hypotheses)
                    return hypotheses
                else:
                    print(f"  ⚠ Hypothesis Validation Failed")
                    
        except Exception as e:
            print(f"[HYPOTHESIS ERROR] {e}")
        return []
    
    def _save_hypotheses(self, gap: str, theory: str, hypotheses: List[Dict]):
        from storm_modules.db_safety import get_db_connection
        try:
            with get_db_connection(self.db_path) as conn:
                gap_id = abs(hash(gap)) % 1000000
                cursor = conn.execute("SELECT id FROM theories WHERE name = ?", (theory,))
                theory_row = cursor.fetchone()
                theory_id = theory_row[0] if theory_row else None
                
                for h in hypotheses:
                    conn.execute("""
                        INSERT INTO hypotheses
                        (gap_id, theory_id, hypothesis_text, hypothesis_type, variables, expected_effect_size)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        gap_id, theory_id, h.get('statement'), h.get('type'),
                        json.dumps(h), h.get('expected_effect_size')
                    ))
        except Exception as e:
            print(f"[HYPOTHESIS SAVE ERROR] {e}")
