from __future__ import annotations

import sqlite3
from storm_modules.config import get_academic_brain_db_path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paper_methods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    sample_size INTEGER,
    statistical_tests TEXT,
    measures TEXT,
    design_type TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS theories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    core_propositions TEXT,
    key_concepts TEXT,
    typical_hypotheses TEXT,
    typical_methods TEXT,
    boundary_conditions TEXT,
    digital_application TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gap_id INTEGER,
    theory_id INTEGER,
    hypothesis_text TEXT,
    hypothesis_type TEXT,
    variables TEXT,
    expected_effect_size REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS study_designs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gap_id INTEGER,
    theory_id INTEGER,
    sample_size INTEGER,
    design_type TEXT,
    variables TEXT,
    measures TEXT,
    analysis_plan TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS generated_theses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    gap_description TEXT,
    theory_used TEXT,
    thesis_sections TEXT,
    pdf_path TEXT,
    word_count INTEGER,
    citation_count INTEGER,
    q1_score INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    page INTEGER,
    rows INTEGER,
    columns INTEGER,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_figures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    page INTEGER,
    figure_id TEXT,
    caption TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    author TEXT,
    year INTEGER,
    raw_text TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- GROBID Full Extraction Tables
CREATE TABLE IF NOT EXISTS paper_authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    author_name TEXT,
    email TEXT,
    affiliation TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_structured_refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    ref_author TEXT,
    ref_year INTEGER,
    ref_title TEXT,
    ref_journal TEXT,
    ref_doi TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    keyword TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_dates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    date_type TEXT,
    date_value TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_abstracts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    abstract_text TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS research_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    insight_type TEXT NOT NULL,
    content TEXT NOT NULL,
    target_query TEXT,
    weight REAL DEFAULT 1.0,
    status TEXT DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def apply_schema(db_path: str | None = None) -> None:
    print("Applying database schema...")
    target = db_path if db_path else str(get_academic_brain_db_path())
    print(f"Target Database: {target}")
    conn = sqlite3.connect(target)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        print("Schema applied successfully! ✅")
    except Exception as e:
        print(f"Error applying schema: {e} ❌")
    finally:
        conn.close()


if __name__ == "__main__":
    apply_schema()
