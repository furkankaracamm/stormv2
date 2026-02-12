"""Academic Writer - Thesis Sections in Q1 Style"""
import json
from typing import Dict, List
from storm_modules.llm_gateway import get_llm_gateway

class AcademicWriter:
    def __init__(self):
        self.llm = get_llm_gateway()
    
    def write_abstract(self, gap: str, study_design: Dict) -> str:
        if not self.llm.is_available():
            return "[Abstract generation requires LLM]"
        
        prompt = f"""Write academic abstract (250 words).

GAP: {gap}
DESIGN: {study_design.get('design_type')}, N={study_design.get('sample_size')}

Structure:
1. Problem (2 sentences)
2. Gap (1-2 sentences)
3. Purpose (1 sentence)
4. Method (3 sentences)
5. Expected results (2 sentences)
6. Implications (1-2 sentences)

Output ONLY the abstract."""

        result = self.llm.generate(prompt, max_tokens=400, temperature=0.5)
        return result if result else "[Abstract generation failed]"
    
    def write_introduction(self, gap: str, theory: str, lit_summary: str) -> str:
        if not self.llm.is_available():
            return "[Introduction generation requires LLM]"
        
        prompt = f"""Write Introduction (1500 words, 6 paragraphs).

TOPIC: {gap}
THEORY: {theory}
LITERATURE: {lit_summary[:500]}

P1 (150w): Hook + context
P2 (250w): Literature overview + key findings
P3 (250w): Gap identification + significance
P4 (200w): Study purpose + RQs
P5 (150w): Expected contributions
P6 (100w): Paper roadmap

Academic tone, cite (Author, Year).

Output ONLY the Introduction."""

        result = self.llm.generate(prompt, max_tokens=2500, temperature=0.5)
        return result if result else "[Introduction generation failed]"
    
    def write_methodology(self, study_design: Dict) -> str:
        if not self.llm.is_available():
            return "[Methodology generation requires LLM]"
        
        prompt = f"""Write Methodology section (2500 words).

DESIGN: {json.dumps(study_design, indent=2)}

Structure:
1. Research Design (300w)
2. Participants (400w) - N={study_design.get('sample_size')} with justification
3. Procedure (500w)
4. Measures (800w) - Each variable detailed
5. Analytical Strategy (500w)

Past tense, replication detail.

Output ONLY the Methodology."""

        result = self.llm.generate(prompt, max_tokens=4000, temperature=0.5)
        return result if result else "[Methodology generation failed]"
