"""Argument Mapper - Identify support/oppose relationships between papers."""

import re
import sqlite3
from typing import List, Dict, Optional
from storm_modules.config import get_academic_brain_db_path


# Signals that indicate support for another work
SUPPORT_SIGNALS = [
    "consistent with",
    "supports",
    "confirms",
    "in line with",
    "corroborates",
    "aligns with",
    "in agreement with",
    "validates",
    "extends",
    "builds on",
    "reinforces",
    "in accordance with"
]

# Signals that indicate opposition or contradiction
OPPOSE_SIGNALS = [
    "contradicts",
    "challenges",
    "disputes",
    "contrary to",
    "inconsistent with",
    "opposes",
    "conflicts with",
    "fails to replicate",
    "in contrast to",
    "differs from",
    "unlike",
    "however",
    "on the other hand"
]

# Signals for limitations or gaps
GAP_SIGNALS = [
    "fails to address",
    "overlooked",
    "gap in",
    "limitation",
    "future research",
    "remains unclear",
    "not yet examined",
    "underexplored"
]


def detect_argument_signals(text: str) -> Dict[str, List[str]]:
    """
    Detect argument signals in text.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with lists of found signals by type
    """
    text_lower = text.lower()
    
    signals = {
        "support": [],
        "oppose": [],
        "gap": []
    }
    
    for signal in SUPPORT_SIGNALS:
        if signal in text_lower:
            signals["support"].append(signal)
    
    for signal in OPPOSE_SIGNALS:
        if signal in text_lower:
            signals["oppose"].append(signal)
    
    for signal in GAP_SIGNALS:
        if signal in text_lower:
            signals["gap"].append(signal)
    
    return signals


def map_citation_arguments(text: str, paper_references: List[Dict]) -> List[Dict]:
    """
    Map argument relationships between current paper and its references.
    
    Args:
        text: Full paper text
        paper_references: List of reference dictionaries from GROBID
    
    Returns:
        List of argument relationship dictionaries
    """
    relations = []
    sentences = re.split(r'[.!?]\s+', text)
    
    for ref in paper_references:
        ref_author = ref.get("author", "")
        ref_year = ref.get("year", "")
        ref_title = ref.get("title", "")
        
        if not ref_author:
            continue
        
        # Find sentences mentioning this reference
        for sent in sentences:
            # Check if reference is mentioned
            author_pattern = ref_author.split(",")[0] if "," in ref_author else ref_author
            if author_pattern.lower() in sent.lower():
                signals = detect_argument_signals(sent)
                
                if signals["support"]:
                    relations.append({
                        "reference": ref,
                        "relation": "SUPPORTS",
                        "sentence": sent,
                        "signals": signals["support"]
                    })
                elif signals["oppose"]:
                    relations.append({
                        "reference": ref,
                        "relation": "OPPOSES",
                        "sentence": sent,
                        "signals": signals["oppose"]
                    })
                elif signals["gap"]:
                    relations.append({
                        "reference": ref,
                        "relation": "EXTENDS",
                        "sentence": sent,
                        "signals": signals["gap"]
                    })
    
    return relations


def save_argument_relations(filename: str, relations: List[Dict], db_path: Optional[str] = None) -> bool:
    """
    Save argument relations to database.
    
    Args:
        filename: Source PDF filename
        relations: List of relation dictionaries
        db_path: Database path (optional)
    
    Returns:
        Success status
    """
    try:
        path = db_path or str(get_academic_brain_db_path())
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_arguments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                ref_author TEXT,
                ref_title TEXT,
                relation_type TEXT,
                context_sentence TEXT,
                signals TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        for rel in relations:
            ref = rel.get("reference", {})
            cursor.execute(
                """INSERT INTO paper_arguments 
                   (filename, ref_author, ref_title, relation_type, context_sentence, signals) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    filename,
                    ref.get("author"),
                    ref.get("title"),
                    rel.get("relation"),
                    rel.get("sentence"),
                    ",".join(rel.get("signals", []))
                )
            )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ARGUMENT DB ERROR] {e}")
        return False


def get_argument_network_stats(db_path: Optional[str] = None) -> Dict:
    """
    Get statistics about the argument network.
    
    Returns:
        Dictionary with network statistics
    """
    try:
        path = db_path or str(get_academic_brain_db_path())
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM paper_arguments WHERE relation_type = 'SUPPORTS'")
        supports = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM paper_arguments WHERE relation_type = 'OPPOSES'")
        opposes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM paper_arguments WHERE relation_type = 'EXTENDS'")
        extends = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "support_relations": supports,
            "oppose_relations": opposes,
            "extend_relations": extends,
            "total_relations": supports + opposes + extends
        }
    except Exception as e:
        print(f"[ARGUMENT STATS ERROR] {e}")
        return {}
