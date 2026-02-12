"""Hypothesis Generator - Theory + Gap → Testable Hypotheses"""
import json
import sqlite3
from typing import List, Dict
from storm_modules.config import get_academic_brain_db_path
from storm_modules.llm_gateway import get_llm_gateway

class HypothesisGenerator:
    def __init__(self):
        self.llm = get_llm_gateway()
        self.db_path = str(get_academic_brain_db_path())
        self.theories = self._load_theories()
    
    def _load_theories(self) -> Dict:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, core_propositions, typical_hypotheses FROM theories")
            rows = cursor.fetchall()
            theories = {}
            for row in rows:
                theories[row[0]] = {
                    'core_propositions': json.loads(row[1]),
                    'typical_hypotheses': json.loads(row[2])
                }
            conn.close()
            return theories
        except (sqlite3.Error, json.JSONDecodeError) as e:
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
                content = result.replace('```json', '').replace('```', '').strip()
                parsed = json.loads(content)
                hypotheses = parsed.get('hypotheses', [])
                
                if hypotheses:
                    self._save_hypotheses(gap_description, theory_name, hypotheses)
                return hypotheses
        except Exception as e:
            print(f"[HYPOTHESIS ERROR] {e}")
        return []
    
    def _save_hypotheses(self, gap: str, theory: str, hypotheses: List[Dict]):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            gap_id = abs(hash(gap)) % 1000000
            cursor.execute("SELECT id FROM theories WHERE name = ?", (theory,))
            theory_row = cursor.fetchone()
            theory_id = theory_row[0] if theory_row else None
            
            for h in hypotheses:
                cursor.execute("""
                    INSERT INTO hypotheses
                    (gap_id, theory_id, hypothesis_text, hypothesis_type, variables, expected_effect_size)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    gap_id, theory_id, h.get('statement'), h.get('type'),
                    json.dumps(h), h.get('expected_effect_size')
                ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"[HYPOTHESIS SAVE ERROR] {e}")
