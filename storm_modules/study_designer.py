"""Quantitative Study Designer - Hypotheses → Full Methodology"""
import json

import numpy as np
from typing import Dict, List
from pathlib import Path
from storm_modules.config import get_academic_brain_db_path
from storm_modules.llm_gateway import get_llm_gateway

class QuantitativeStudyDesigner:
    def __init__(self):
        self.llm = get_llm_gateway()
        self.db_path = str(get_academic_brain_db_path())
        self.norms = self._load_norms()
    
    def _load_norms(self) -> Dict:
        try:
            norms_path = Path(self.db_path).parent / 'methods_norms.json'
            if norms_path.exists():
                with open(norms_path) as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[STUDY_DESIGNER] Norms load error: {e}")
        return {'survey_median_n': 400, 'experiment_median_n': 150}
    
    def design_study(self, hypotheses: List[Dict], gap: str, theory: str) -> Dict:
        print(f"\n[STUDY DESIGNER] Designing...")
        
        design_type = 'survey' if not any('manipulat' in h.get('statement', '').lower() for h in hypotheses) else 'experiment'
        sample_size = self.power_analysis(hypotheses, design_type)
        variables = self.extract_variables(hypotheses)
        measures = self.suggest_measures(variables, theory)
        analysis_plan = self.create_analysis_plan(hypotheses)
        
        design = {
            'design_type': design_type,
            'sample_size': sample_size,
            'variables': variables,
            'measures': measures,
            'analysis_plan': analysis_plan
        }
        
        self._save_design(gap, theory, design)
        return design
    
    def power_analysis(self, hypotheses: List[Dict], design_type: str) -> int:
        effect_sizes = [h.get('expected_effect_size', 0.25) for h in hypotheses]
        median_effect = np.median(effect_sizes) if effect_sizes else 0.25
        f_squared = median_effect ** 2
        z_alpha, z_beta = 1.96, 0.84
        n = ((z_alpha + z_beta) ** 2) / f_squared
        n = max(int(np.ceil(n)), 200)
        return int(np.ceil(n / 50) * 50)
    
    def extract_variables(self, hypotheses: List[Dict]) -> Dict:
        variables = {'independent': [], 'dependent': [], 'moderators': [], 'mediators': [],
                    'controls': ['age', 'education', 'gender']}
        for h in hypotheses:
            if 'IV' in h and h['IV'] not in variables['independent']:
                variables['independent'].append(h['IV'])
            if 'DV' in h and h['DV'] not in variables['dependent']:
                variables['dependent'].append(h['DV'])
            if h.get('type') == 'moderation' and 'moderator' in h:
                variables['moderators'].append(h['moderator'])
            if h.get('type') == 'mediation' and 'mediator' in h:
                variables['mediators'].append(h['mediator'])
        return variables
    
    def suggest_measures(self, variables: Dict, theory: str) -> Dict:
        if not self.llm.is_available():
            return {}
        
        all_vars = (variables['independent'] + variables['dependent'] + 
                   variables['moderators'] + variables['mediators'])
        
        prompt = f"""Suggest validated scales for: {', '.join(all_vars)}
Theory: {theory}

Output ONLY JSON:
{{
  "variable_name": {{
    "scale_name": "Name",
    "citation": "Author (Year)",
    "items": 7,
    "response_format": "7-point Likert",
    "expected_alpha": 0.85
  }}
}}"""

        try:
            result = self.llm.generate(prompt, max_tokens=1000, temperature=0.3, json_mode=True)
            if result:
                return json.loads(result.replace('```json', '').replace('```', '').strip())
        except json.JSONDecodeError as e:
            print(f"[STUDY_DESIGNER] JSON parse error: {e}")
        return {}
    
    def create_analysis_plan(self, hypotheses: List[Dict]) -> List[Dict]:
        plan = []
        for h in hypotheses:
            h_type = h.get('type', 'main_effect')
            if h_type == 'main_effect':
                plan.append({
                    'hypothesis': h.get('id'),
                    'test': 'Hierarchical Regression',
                    'expected_beta': h.get('expected_effect_size', 0.25)
                })
            elif h_type == 'moderation':
                plan.append({
                    'hypothesis': h.get('id'),
                    'test': 'Hayes PROCESS Model 1',
                    'bootstrap': 5000
                })
            elif h_type == 'mediation':
                plan.append({
                    'hypothesis': h.get('id'),
                    'test': 'Hayes PROCESS Model 4',
                    'bootstrap': 5000
                })
        return plan
    
    def _save_design(self, gap: str, theory: str, design: Dict):
        from storm_modules.db_safety import get_db_connection
        try:
            with get_db_connection(self.db_path) as conn:
                gap_id = abs(hash(gap)) % 1000000
                cursor = conn.execute("SELECT id FROM theories WHERE name = ?", (theory,))
                theory_row = cursor.fetchone()
                theory_id = theory_row[0] if theory_row else None
                
                conn.execute("""
                    INSERT INTO study_designs
                    (gap_id, theory_id, sample_size, design_type, variables, measures, analysis_plan)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (gap_id, theory_id, design['sample_size'], design['design_type'],
                     json.dumps(design['variables']), json.dumps(design['measures']),
                     json.dumps(design['analysis_plan'])))
        except Exception as e:
            print(f"[SAVE ERROR] {e}")
