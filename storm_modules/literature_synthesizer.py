"""Literature Synthesizer - Clusters → Academic Prose"""
import json
from typing import List, Dict
from storm_modules.llm_gateway import get_llm_gateway

class LiteratureSynthesizer:
    def __init__(self):
        self.llm = get_llm_gateway()
    
    def synthesize_cluster(self, cluster_papers: List[Dict], theme: str) -> str:
        if not self.llm.is_available():
            return f"[LLM not available for {theme}]"
        
        papers_str = "\n".join([
            f"- {p.get('authors', 'Unknown')} ({p.get('year', '????')}): {p.get('key_findings', '')[:200]}"
            for p in cluster_papers[:15]
        ])
        
        prompt = f"""Write literature review paragraph for top-tier journal.

THEME: {theme}
PAPERS: {papers_str}

Write ONE paragraph (250-300 words):
1. Summarize main findings
2. Identify patterns/trends
3. Note contradictions
4. Highlight gaps

Academic style. Cite inline: (Author, Year).
SYNTHESIZE, don't just list.

Output ONLY the paragraph."""

        result = self.llm.generate(prompt, max_tokens=500, temperature=0.5)
        return result if result else f"[Error: Could not synthesize {theme}]"
    
    def synthesize_full_review(self, clustered_papers: Dict[str, List[Dict]], gap: str) -> str:
        sections = []
        
        for theme, papers in clustered_papers.items():
            print(f"  Synthesizing: {theme}")
            synthesis = self.synthesize_cluster(papers, theme)
            sections.append((theme, synthesis))
        
        full_review = []
        for title, content in sections:
            full_review.append(f"\n{title}\n\n{content}\n")
        
        return "\n".join(full_review)
