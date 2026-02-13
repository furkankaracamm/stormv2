"""
ResearchPlanner - Strategic Mediator for LLM Influence
Manages weight-based influence, structured insight persistence, and tactical research directions.
"""

import os
import json

import time
from typing import List, Dict, Optional
from pathlib import Path
from storm_modules.config import get_academic_brain_db_path

class ResearchPlanner:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(get_academic_brain_db_path())
        self.decay_rate = 0.1  # 10% weight decay per cycle
        self.failure_penalty = 0.5  # 50% penalty on failure
        self._ensure_table()

    def _ensure_table(self):
        """Redundant check for table existence."""
        from storm_modules.db_safety import get_db_connection
        try:
            with get_db_connection(self.db_path) as conn:
                # Table already defined in schema.py, but we ensure it here if running standalone
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS research_insights (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        insight_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        target_query TEXT,
                        weight REAL DEFAULT 1.0,
                        status TEXT DEFAULT 'ACTIVE',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
        except Exception as e:
            print(f"[PLANNER ERROR] Table init failed: {e}")

    def add_insight(self, insight_type: str, content: Dict, target_query: str, weight: float = 1.0):
        """Save a new structured insight to the database."""
        from storm_modules.db_safety import get_db_connection
        try:
            with get_db_connection(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO research_insights (insight_type, content, target_query, weight)
                    VALUES (?, ?, ?, ?)
                ''', (insight_type, json.dumps(content), target_query, weight))
            return True
        except Exception as e:
            print(f"[PLANNER ERROR] Failed to add insight: {e}")
            return False

    def get_top_strategic_targets(self, limit: int = 3) -> List[Dict]:
        """Retrieve the highest-weighted active strategic targets."""
        targets = []
        from storm_modules.db_safety import get_db_connection
        try:
            with get_db_connection(self.db_path) as conn:
                # Only pull ACTIVE insights with significant weight
                cursor = conn.execute('''
                    SELECT id, target_query, weight, insight_type 
                    FROM research_insights 
                    WHERE status = 'ACTIVE' AND weight > 0.1
                    ORDER BY weight DESC 
                    LIMIT ?
                ''', (limit,))
                
                for row in cursor.fetchall():
                    targets.append({
                        "id": row[0],
                        "query": row[1],
                        "weight": row[2],
                        "type": row[3]
                    })
        except Exception as e:
            print(f"[PLANNER ERROR] Query failed: {e}")
        return targets

    def apply_cycle_decay(self):
        """Reduce weights of all active insights (Epistemic Decay)."""
        from storm_modules.db_safety import get_db_connection
        try:
            with get_db_connection(self.db_path) as conn:
                # Reduce weight and archive if too low
                conn.execute('UPDATE research_insights SET weight = weight * (1.0 - ?) WHERE status = "ACTIVE"', (self.decay_rate,))
                conn.execute('UPDATE research_insights SET status = "DECAYED" WHERE weight < 0.1 AND status = "ACTIVE"')
        except Exception as e:
            print(f"[PLANNER ERROR] Decay failed: {e}")

    def report_failure(self, insight_id: int):
        """Penalize an insight if the suggested research target failed."""
        from storm_modules.db_safety import get_db_connection
        try:
            with get_db_connection(self.db_path) as conn:
                conn.execute('UPDATE research_insights SET weight = weight * ? WHERE id = ?', (self.failure_penalty, insight_id))
        except Exception as e:
            print(f"[PLANNER ERROR] Failure report failed: {e}")

    def report_success(self, insight_id: int):
        """Reinforce an insight if it led to successful data retrieval."""
        from storm_modules.db_safety import get_db_connection
        try:
            with get_db_connection(self.db_path) as conn:
                # Boost weight but cap at 1.5
                conn.execute('UPDATE research_insights SET weight = MIN(1.5, weight * 1.2) WHERE id = ?', (insight_id,))
        except Exception as e:
            print(f"[PLANNER ERROR] Success report failed: {e}")

    def parse_llm_insight(self, raw_text: str) -> Optional[Dict]:
        """
        Heuristic parser for LLM output if it's not strictly JSON.
        Expects a structured response with 'Question', 'Target', 'Rationale'.
        """
        try:
            # Try JSON first
            return json.loads(raw_text)
        except:
            # Heuristic regex extract
            question = re.search(r'Question:\s*(.*)', raw_text)
            target = re.search(r'Target:\s*(.*)', raw_text)
            if question and target:
                return {
                    "question": question.group(1).strip(),
                    "target": target.group(1).strip(),
                    "rationale": "Extracted via heuristic regex"
                }
        return None
