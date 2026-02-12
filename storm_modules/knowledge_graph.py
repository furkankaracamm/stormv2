"""Knowledge Graph Manager - Maps relationships between academic entities"""
import networkx as nx
import os
import pickle
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from storm_modules.config import get_work_dir

class KnowledgeGraphManager:
    def __init__(self, brain_dir: Optional[str] = None):
        self.work_dir = Path(brain_dir) if brain_dir else get_work_dir()
        self.graph_path = self.work_dir / "knowledge_graph.pkl"
        self.graph = self._load_graph()

    def _load_graph(self) -> nx.DiGraph:
        if self.graph_path.exists():
            try:
                with open(self.graph_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return nx.DiGraph()
        return nx.DiGraph()

    def save_graph(self):
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)

    def add_paper(self, paper_id: str, metadata: Dict[str, Any]):
        """Adds a paper node with metadata"""
        self.graph.add_node(paper_id, type='paper', **metadata)
        self.save_graph()

    def add_theory(self, theory_name: str, properties: Dict[str, Any]):
        """Adds a theory node"""
        self.graph.add_node(theory_name, type='theory', **properties)
        self.save_graph()

    def connect(self, source: str, target: str, relationship: str, weight: float = 1.0):
        """
        Creates a directed edge between two entities.
        Relationships: 'CITES', 'SUPPORTS', 'CONTRADICTS', 'APPLIES'
        """
        self.graph.add_edge(source, target, relationship=relationship, weight=weight)
        self.save_graph()

    def get_related_papers(self, entity_id: str, relationship: Optional[str] = None) -> List[str]:
        """Finds papers connected to an entity by a specific relationship"""
        related = []
        for _, neighbor, data in self.graph.edges(entity_id, data=True):
            if relationship:
                if data.get('relationship') == relationship:
                    related.append(neighbor)
            else:
                related.append(neighbor)
        return related

    def find_contradictions(self, theory_name: str) -> List[str]:
        """Finds all papers that contradict a specific theory"""
        return self.get_related_papers(theory_name, relationship='CONTRADICTS')

    def export_visualization(self, output_path: Optional[str] = None):
        """Generates a simple representation or triggers a drawing (requires matplotlib)"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Color nodes by type
        color_map = []
        for node in self.graph:
            node_type = self.graph.nodes[node].get('type', 'unknown')
            if node_type == 'paper': color_map.append('skyblue')
            elif node_type == 'theory': color_map.append('orange')
            else: color_map.append('lightgrey')
        
        nx.draw(self.graph, pos, with_labels=True, node_color=color_map, node_size=2000, font_size=8)
        
        path = output_path or str(self.work_dir / "literature_map.png")
        plt.savefig(path)
        plt.close()
        return path

if __name__ == "__main__":
    # Test Run
    kg = KnowledgeGraphManager()
    kg.add_theory("Social Cognitive Theory", {"desc": "Learning via observation"})
    kg.add_paper("DOI:123/abc", {"title": "Critique of SCT", "year": 2023})
    kg.connect("DOI:123/abc", "Social Cognitive Theory", "CONTRADICTS")
    print(f"Papers contradicting SCT: {kg.find_contradictions('Social Cognitive Theory')}")
