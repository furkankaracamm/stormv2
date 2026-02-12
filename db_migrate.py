"""
STORM DATABASE MIGRATION SCRIPT
Zero-Error Architecture: Create missing tables
"""
import sqlite3
import os
from pathlib import Path

def get_db_path():
    return Path(__file__).parent / "academic_brain.db"

def migrate():
    db_path = get_db_path()
    print(f"[MIGRATE] Database: {db_path}")
    
    if not db_path.exists():
        print("[ERROR] Database does not exist!")
        return False
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}
    print(f"[INFO] Existing tables: {len(existing_tables)}")
    
    # Define required tables
    migrations = {
        "metadata": """
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                content TEXT,
                title TEXT,
                authors TEXT,
                year INTEGER,
                doi TEXT,
                source TEXT,
                file_path TEXT,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER,
                page_count INTEGER
            )
        """,
        "embeddings": """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk_id INTEGER,
                chunk_text TEXT,
                embedding BLOB,
                section TEXT,
                priority TEXT DEFAULT 'normal',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "citations": """
            CREATE TABLE IF NOT EXISTS citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_paper TEXT,
                cited_paper TEXT,
                citation_context TEXT,
                citation_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "gaps": """
            CREATE TABLE IF NOT EXISTS gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                gap_type TEXT,
                geometric_score REAL DEFAULT 0,
                epistemic_score REAL DEFAULT 0,
                status TEXT DEFAULT 'PENDING',
                evidence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "hypotheses": """
            CREATE TABLE IF NOT EXISTS hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gap_id INTEGER,
                theory_id INTEGER,
                hypothesis_text TEXT NOT NULL,
                hypothesis_type TEXT,
                variables TEXT,
                expected_effect_size REAL,
                status TEXT DEFAULT 'PENDING',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (theory_id) REFERENCES theories(id)
            )
        """,
        "study_designs": """
            CREATE TABLE IF NOT EXISTS study_designs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gap_id INTEGER,
                theory_id INTEGER,
                sample_size INTEGER,
                design_type TEXT,
                variables TEXT,
                measures TEXT,
                analysis_plan TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (theory_id) REFERENCES theories(id)
            )
        """
    }
    
    created = 0
    for table_name, sql in migrations.items():
        if table_name not in existing_tables:
            try:
                cursor.execute(sql)
                print(f"  ✓ Created: {table_name}")
                created += 1
            except sqlite3.Error as e:
                print(f"  ✗ Failed: {table_name} - {e}")
        else:
            print(f"  - Exists: {table_name}")
    
    conn.commit()
    
    # Verify final table count
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    final_tables = {row[0] for row in cursor.fetchall()}
    print(f"\n[RESULT] Tables after migration: {len(final_tables)}")
    
    conn.close()
    return True


def verify():
    """Verify all tables exist and have correct schema."""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    required_tables = ['theories', 'metadata', 'embeddings', 'paper_claims', 
                       'citations', 'gaps', 'hypotheses', 'study_designs']
    
    print("\n[VERIFY] Checking tables...")
    all_ok = True
    for table in required_tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  ✓ {table}: {count} rows")
        except sqlite3.Error as e:
            print(f"  ✗ {table}: MISSING")
            all_ok = False
    
    conn.close()
    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("STORM DATABASE MIGRATION")
    print("=" * 60)
    
    if migrate():
        verify()
    
    print("\n" + "=" * 60)
    print("[COMPLETE]")
