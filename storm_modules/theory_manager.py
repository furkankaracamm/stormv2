"""
Theory Version Manager - History and Rollback System

Features:
- Version control for theories
- Change tracking
- Rollback capability
- Atomic updates via SafeDBContext
"""

import json
from typing import Dict, List, Optional, Tuple
from storm_modules.config import get_academic_brain_db_path
from storm_modules.db_safety import get_db_connection, DatabaseError

class TheoryVersionManager:
    def __init__(self):
        self.db_path = str(get_academic_brain_db_path())

    def create_or_update_theory(self, name: str, data: Dict, change_desc: str = "Automated update") -> bool:
        """
        Create new theory or update existing one with versioning.
        Updates 'theories' table (HEAD) and 'theory_versions' table (History).
        """
        try:
            with get_db_connection(self.db_path) as conn:
                # 1. Check if theory exists
                cursor = conn.execute("SELECT id FROM theories WHERE name = ?", (name,))
                row = cursor.fetchone()
                
                if row:
                    theory_id = row[0]
                    # Get next version number
                    cursor = conn.execute(
                        "SELECT MAX(version_num) FROM theory_versions WHERE theory_id = ?", 
                        (theory_id,)
                    )
                    max_ver = cursor.fetchone()[0]
                    next_ver = (max_ver or 0) + 1
                    
                    # Update HEAD
                    conn.execute("""
                        UPDATE theories 
                        SET core_propositions = ?, key_concepts = ?, typical_hypotheses = ?,
                            typical_methods = ?, boundary_conditions = ?, digital_application = ?
                        WHERE id = ?
                    """, (
                        json.dumps(data.get('core_propositions', [])),
                        json.dumps(data.get('key_concepts', [])),
                        json.dumps(data.get('typical_hypotheses', [])),
                        json.dumps(data.get('typical_methods', {})),
                        json.dumps(data.get('boundary_conditions', [])),
                        data.get('digital_application', ''),
                        theory_id
                    ))
                    
                    # Add to History
                    self._add_version(conn, theory_id, next_ver, data, change_desc)
                    print(f"[THEORY MANAGER] Updated '{name}' to version {next_ver}")
                    
                else:
                    # Create new theory
                    cursor = conn.execute("""
                        INSERT INTO theories 
                        (name, core_propositions, key_concepts, typical_hypotheses, 
                         typical_methods, boundary_conditions, digital_application)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        name,
                        json.dumps(data.get('core_propositions', [])),
                        json.dumps(data.get('key_concepts', [])),
                        json.dumps(data.get('typical_hypotheses', [])),
                        json.dumps(data.get('typical_methods', {})),
                        json.dumps(data.get('boundary_conditions', [])),
                        data.get('digital_application', '')
                    ))
                    theory_id = cursor.lastrowid
                    
                    # Add initial version
                    self._add_version(conn, theory_id, 1, data, "Initial creation")
                    print(f"[THEORY MANAGER] Created '{name}' version 1")
                    
            return True
            
        except DatabaseError as e:
            print(f"[THEORY MANAGER ERROR] {e}")
            return False

    def _add_version(self, conn, theory_id: int, version: int, data: Dict, desc: str):
        """Helper to insert version record."""
        conn.execute("""
            INSERT INTO theory_versions 
            (theory_id, version_num, change_description, content_snapshot)
            VALUES (?, ?, ?, ?)
        """, (
            theory_id,
            version,
            desc,
            json.dumps(data)
        ))

    def get_history(self, name: str) -> List[Dict]:
        """Get version history for a theory."""
        history = []
        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT v.version_num, v.change_description, v.created_at
                    FROM theory_versions v
                    JOIN theories t ON v.theory_id = t.id
                    WHERE t.name = ?
                    ORDER BY v.version_num DESC
                """, (name,))
                
                for row in cursor.fetchall():
                    history.append({
                        "version": row[0],
                        "description": row[1],
                        "date": row[2]
                    })
        except Exception as e:
            print(f"[THEORY MANAGER] History fetch failed: {e}")
        return history

    def rollback(self, name: str, target_version: int) -> bool:
        """Rollback theory to a specific version."""
        try:
            with get_db_connection(self.db_path) as conn:
                # Get snapshot
                cursor = conn.execute("""
                    SELECT v.content_snapshot, t.id
                    FROM theory_versions v
                    JOIN theories t ON v.theory_id = t.id
                    WHERE t.name = ? AND v.version_num = ?
                """, (name, target_version))
                
                row = cursor.fetchone()
                if not row:
                    print(f"[ROLLBACK FAIL] Version {target_version} not found for '{name}'")
                    return False
                
                snapshot = json.loads(row[0])
                theory_id = row[1]
                
                # Restore HEAD
                conn.execute("""
                    UPDATE theories 
                    SET core_propositions = ?, key_concepts = ?, typical_hypotheses = ?,
                        typical_methods = ?, boundary_conditions = ?, digital_application = ?
                    WHERE id = ?
                """, (
                    json.dumps(snapshot.get('core_propositions', [])),
                    json.dumps(snapshot.get('key_concepts', [])),
                    json.dumps(snapshot.get('typical_hypotheses', [])),
                    json.dumps(snapshot.get('typical_methods', {})),
                    json.dumps(snapshot.get('boundary_conditions', [])),
                    snapshot.get('digital_application', ''),
                    theory_id
                ))
                
                # Add "Rollback" version entry
                cursor = conn.execute("SELECT MAX(version_num) FROM theory_versions WHERE theory_id = ?", (theory_id,))
                next_ver = (cursor.fetchone()[0] or 0) + 1
                
                self._add_version(conn, theory_id, next_ver, snapshot, f"Rollback to v{target_version}")
                print(f"[THEORY MANAGER] Rolled back '{name}' to v{target_version} (as v{next_ver})")
                
            return True
        except Exception as e:
            print(f"[ROLLBACK ERROR] {e}")
            return False
