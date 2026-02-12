"""
Research Gap Finder - Hybrid LitStudy + Litmaps approach

Combines:
- LitStudy: NLP topic modeling to find understudied topics
- Litmaps: Citation network analysis to find disconnections

Gap Types Detected:
1. Topic Gaps: Topics with few papers in our collection
2. Citation Gaps: Papers that should cite each other but don't
3. Contradiction Gaps: Claims that conflict with each other
4. Geographic Gaps: Regions not studied
5. Method Gaps: Methods not applied
"""

import sqlite3
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class GapFinder:
    """Hybrid research gap detection combining NLP and network analysis."""
    
    def __init__(self, db_path: str, brain_data_dir: str = None):
        self.db_path = db_path
        self.brain_data_dir = brain_data_dir
        self.papers = self._load_papers()
        self.claims = self._load_claims()
        self.graph = self._load_citation_network()
    
    def _load_papers(self) -> List[Dict]:
        """Load paper metadata from database."""
        papers = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT filename, content FROM metadata")
            for row in cursor.fetchall():
                papers.append({
                    'filename': row[0],
                    'content': row[1] or ""
                })
            conn.close()
        except sqlite3.Error as e:
            print(f"[GAP_FINDER] Papers load error: {e}")
        return papers
    
    def _load_claims(self) -> List[Dict]:
        """Load extracted claims from database."""
        claims = []
        try:
            academic_db = str(Path(self.db_path).parent / "academic_brain.db")
            conn = sqlite3.connect(academic_db)
            cursor = conn.cursor()
            cursor = conn.cursor()
            # [FIX] Schema Alignment: Use 'filename' instead of non-existent 'paper_id'
            cursor.execute("SELECT filename, claim_text, claim_type, confidence FROM paper_claims")
            for row in cursor.fetchall():
                claims.append({
                    'paper_id': row[0], # Map filename to paper_id for compatibility
                    'text': row[1],
                    'type': row[2],
                    'confidence': row[3]
                })
            conn.close()
        except sqlite3.Error as e:
            print(f"[GAP_FINDER] Claims load error: {e}")
        return claims
    
    def _load_citation_network(self) -> Optional[object]:
        """Load citation network from pickle file."""
        if not HAS_NETWORKX or not self.brain_data_dir:
            return None
        try:
            import pickle
            graph_path = Path(self.brain_data_dir) / "knowledge_graph.pkl"
            if graph_path.exists():
                with open(graph_path, 'rb') as f:
                    return pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            print(f"[GAP_FINDER] Graph load error: {e}")
        return nx.DiGraph() if HAS_NETWORKX else None
    
    # =========================================================================
    # GAP TYPE 1: Topic Gaps (LitStudy-style)
    # =========================================================================
    
    def find_topic_gaps(self, num_topics: int = 8, min_papers_threshold: int = 3) -> List[Dict]:
        """
        Find understudied topics using clustering.
        
        Returns topics that have fewer papers than the threshold.
        These represent areas where more research is needed.
        """
        if not HAS_SKLEARN or len(self.papers) < 5:
            return []
        
        # Build TF-IDF matrix
        texts = [p['content'][:2000] for p in self.papers]  # Limit text length
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError as e:
            print(f"[GAP_FINDER] TF-IDF error: {e}")
            return []
        
        # Cluster into topics
        n_clusters = min(num_topics, len(self.papers) // 2)
        if n_clusters < 2:
            return []
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Count papers per cluster
        cluster_counts = defaultdict(int)
        cluster_papers = defaultdict(list)
        
        for i, label in enumerate(cluster_labels):
            cluster_counts[label] += 1
            cluster_papers[label].append(self.papers[i]['filename'])
        
        # Get top terms for each cluster
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        
        gaps = []
        for cluster_id in range(n_clusters):
            count = cluster_counts[cluster_id]
            
            # Get top 5 terms for this cluster
            top_terms = [terms[ind] for ind in order_centroids[cluster_id, :5]]
            
            if count <= min_papers_threshold:
                gaps.append({
                    'type': 'topic_gap',
                    'topic_id': cluster_id,
                    'topic_terms': top_terms,
                    'paper_count': count,
                    'papers': cluster_papers[cluster_id],
                    'gap_score': 1.0 - (count / max(cluster_counts.values())),
                    'suggestion': f"Topic '{' '.join(top_terms[:3])}' has only {count} papers - consider more research here"
                })
        
        return sorted(gaps, key=lambda x: x['gap_score'], reverse=True)
    
    # =========================================================================
    # GAP TYPE 2: Citation Gaps (Litmaps-style)
    # =========================================================================
    
    def find_citation_gaps(self, similarity_threshold: float = 0.6) -> List[Dict]:
        """
        Find papers that are topically similar but don't cite each other.
        
        This is the core Litmaps concept - finding "disconnections" in 
        the citation network between related papers.
        """
        if not HAS_SKLEARN or not HAS_NETWORKX or len(self.papers) < 3:
            return []
        
        if self.graph is None or len(self.graph.nodes()) == 0:
            return []
        
        # Build similarity matrix
        texts = [p['content'][:2000] for p in self.papers]
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except ValueError as e:
            print(f"[GAP_FINDER] Similarity error: {e}")
            return []
        
        gaps = []
        paper_files = [p['filename'] for p in self.papers]
        
        # Find similar papers that don't cite each other
        for i in range(len(self.papers)):
            for j in range(i + 1, len(self.papers)):
                similarity = similarity_matrix[i, j]
                
                if similarity >= similarity_threshold:
                    paper_a = paper_files[i]
                    paper_b = paper_files[j]
                    
                    # Check if they cite each other in the graph
                    a_cites_b = self.graph.has_edge(paper_a, paper_b) if paper_a in self.graph and paper_b in self.graph else False
                    b_cites_a = self.graph.has_edge(paper_b, paper_a) if paper_a in self.graph and paper_b in self.graph else False
                    
                    if not a_cites_b and not b_cites_a:
                        gaps.append({
                            'type': 'citation_gap',
                            'paper_a': paper_a,
                            'paper_b': paper_b,
                            'similarity': float(similarity),
                            'gap_score': float(similarity),  # Higher similarity = bigger gap
                            'suggestion': f"'{paper_a[:30]}...' and '{paper_b[:30]}...' are similar ({similarity:.2f}) but don't cite each other"
                        })
        
        return sorted(gaps, key=lambda x: x['gap_score'], reverse=True)[:20]  # Top 20
    
    # =========================================================================
    # GAP TYPE 3: Contradiction Gaps
    # =========================================================================
    
    def find_contradiction_gaps(self) -> List[Dict]:
        """
        Find claims that contradict each other.
        
        Looks for patterns like:
        - "X increases Y" vs "X decreases Y"
        - "X is significant" vs "X is not significant"
        """
        if len(self.claims) < 2:
            return []
        
        # Contradiction patterns
        increase_pattern = re.compile(r'(increase|enhance|improve|positive|higher)', re.I)
        decrease_pattern = re.compile(r'(decrease|reduce|lower|negative|decline)', re.I)
        significant_pattern = re.compile(r'(significant|substantial|notable)', re.I)
        not_significant_pattern = re.compile(r'(not significant|no significant|insignificant|negligible)', re.I)
        
        gaps = []
        
        for i, claim_a in enumerate(self.claims):
            for j, claim_b in enumerate(self.claims[i+1:], i+1):
                # Skip same paper
                if claim_a['paper_id'] == claim_b['paper_id']:
                    continue
                
                text_a = claim_a['text'].lower()
                text_b = claim_b['text'].lower()
                
                contradiction = False
                reason = ""
                
                # Check increase vs decrease
                if increase_pattern.search(text_a) and decrease_pattern.search(text_b):
                    # Check if they're about similar topics (simple word overlap)
                    words_a = set(text_a.split())
                    words_b = set(text_b.split())
                    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                    if overlap > 0.3:
                        contradiction = True
                        reason = "Direction conflict (increase vs decrease)"
                
                # Check significant vs not significant
                if significant_pattern.search(text_a) and not_significant_pattern.search(text_b):
                    words_a = set(text_a.split())
                    words_b = set(text_b.split())
                    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                    if overlap > 0.3:
                        contradiction = True
                        reason = "Significance conflict"
                
                if contradiction:
                    gaps.append({
                        'type': 'contradiction_gap',
                        'claim_a': claim_a['text'][:100],
                        'claim_b': claim_b['text'][:100],
                        'paper_a': claim_a['paper_id'],
                        'paper_b': claim_b['paper_id'],
                        'reason': reason,
                        'gap_score': 0.8,
                        'suggestion': f"Research opportunity: resolve conflict between papers on '{reason}'"
                    })
        
        return gaps[:10]  # Top 10
    
    # =========================================================================
    # GAP TYPE 4: Geographic Gaps
    # =========================================================================
    
    def find_geographic_gaps(self) -> List[Dict]:
        """Find regions that are understudied in the literature."""
        regions = {
            'Western': ['usa', 'united states', 'uk', 'britain', 'europe', 'germany', 'france'],
            'Asian': ['china', 'japan', 'korea', 'india', 'asia', 'asian'],
            'Middle Eastern': ['turkey', 'iran', 'arab', 'middle east', 'israel'],
            'African': ['africa', 'nigeria', 'south africa', 'kenya', 'egypt'],
            'Latin American': ['brazil', 'mexico', 'argentina', 'latin america']
        }
        
        region_counts = defaultdict(int)
        
        for paper in self.papers:
            content_lower = paper['content'].lower()
            for region, keywords in regions.items():
                if any(kw in content_lower for kw in keywords):
                    region_counts[region] += 1
        
        if not region_counts:
            return []
        
        max_count = max(region_counts.values())
        gaps = []
        
        for region, keywords in regions.items():
            count = region_counts.get(region, 0)
            gap_score = 1.0 - (count / max_count) if max_count > 0 else 1.0
            
            if gap_score > 0.7:  # Very understudied
                gaps.append({
                    'type': 'geographic_gap',
                    'region': region,
                    'paper_count': count,
                    'gap_score': gap_score,
                    'suggestion': f"'{region}' context is understudied ({count} papers) - consider regional study"
                })
        
        return sorted(gaps, key=lambda x: x['gap_score'], reverse=True)
    
    # =========================================================================
    # GAP TYPE 5: Method Gaps
    # =========================================================================
    
    def find_method_gaps(self) -> List[Dict]:
        """Find research methods that are underused."""
        methods = {
            'Survey': ['survey', 'questionnaire', 'self-report'],
            'Experiment': ['experiment', 'experimental', 'manipulation', 'treatment'],
            'Qualitative': ['interview', 'qualitative', 'focus group', 'ethnograph'],
            'Content Analysis': ['content analysis', 'text analysis', 'discourse'],
            'Longitudinal': ['longitudinal', 'panel', 'time series', 'over time'],
            'Meta-Analysis': ['meta-analysis', 'systematic review', 'meta analysis']
        }
        
        method_counts = defaultdict(int)
        
        for paper in self.papers:
            content_lower = paper['content'].lower()
            for method, keywords in methods.items():
                if any(kw in content_lower for kw in keywords):
                    method_counts[method] += 1
        
        if not method_counts:
            return []
        
        max_count = max(method_counts.values())
        gaps = []
        
        for method, keywords in methods.items():
            count = method_counts.get(method, 0)
            gap_score = 1.0 - (count / max_count) if max_count > 0 else 1.0
            
            if gap_score > 0.6:  # Underused
                gaps.append({
                    'type': 'method_gap',
                    'method': method,
                    'paper_count': count,
                    'gap_score': gap_score,
                    'suggestion': f"'{method}' method is underused ({count} papers) - methodological opportunity"
                })
        
        return sorted(gaps, key=lambda x: x['gap_score'], reverse=True)
    
    # =========================================================================
    # MASTER GAP DETECTION
    # =========================================================================
    
    def find_all_gaps(self) -> Dict[str, List[Dict]]:
        """Run all gap detection methods and return combined results."""
        print("[GAP FINDER] Analyzing research gaps...")
        
        results = {
            'topic_gaps': self.find_topic_gaps(),
            'citation_gaps': self.find_citation_gaps(),
            'contradiction_gaps': self.find_contradiction_gaps(),
            'geographic_gaps': self.find_geographic_gaps(),
            'method_gaps': self.find_method_gaps()
        }
        
        # Calculate overall statistics
        total_gaps = sum(len(v) for v in results.values())
        print(f"[GAP FINDER] Found {total_gaps} total research gaps:")
        for gap_type, gaps in results.items():
            if gaps:
                print(f"  - {gap_type}: {len(gaps)}")
        
        return results
    
    def get_top_gaps(self, limit: int = 10) -> List[Dict]:
        """Get the top N most promising research gaps across all types."""
        all_results = self.find_all_gaps()
        
        all_gaps = []
        for gap_type, gaps in all_results.items():
            all_gaps.extend(gaps)
        
        # Sort by gap_score
        sorted_gaps = sorted(all_gaps, key=lambda x: x.get('gap_score', 0), reverse=True)
        return sorted_gaps[:limit]
    
    def save_gaps_to_db(self, gaps: List[Dict]) -> None:
        """Save detected gaps to the database."""
        try:
            academic_db = str(Path(self.db_path).parent / "academic_brain.db")
            conn = sqlite3.connect(academic_db)
            cursor = conn.cursor()
            
            # Create gaps table if not exists
            cursor.execute('''CREATE TABLE IF NOT EXISTS detected_gaps
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 gap_type TEXT,
                 description TEXT,
                 gap_score REAL,
                 suggestion TEXT,
                 metadata TEXT,
                 detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            
            for gap in gaps:
                cursor.execute('''
                    INSERT INTO detected_gaps (gap_type, description, gap_score, suggestion, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    gap.get('type'),
                    str(gap),
                    gap.get('gap_score', 0),
                    gap.get('suggestion', ''),
                    json.dumps(gap)
                ))
            
            conn.commit()
            conn.close()
            print(f"[GAP FINDER] Saved {len(gaps)} gaps to database")
        except Exception as e:
            print(f"[GAP FINDER ERROR] {e}")


# Convenience function for STORM integration
def analyze_gaps(metadata_db: str, brain_dir: str) -> List[Dict]:
    """Quick function to find research gaps."""
    finder = GapFinder(metadata_db, brain_dir)
    return finder.get_top_gaps(limit=10)
