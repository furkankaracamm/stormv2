from __future__ import annotations

import sqlite3
from typing import Iterable, Optional

from storm_modules.config import get_academic_brain_db_path


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    target = db_path if db_path else str(get_academic_brain_db_path())
    return sqlite3.connect(target)


def save_table_results(
    filename: str,
    results: Iterable[object],
    db_path: Optional[str] = None,
) -> None:
    conn = _connect(db_path)
    try:
        cursor = conn.cursor()
        for result in results:
            page = getattr(result, "page", None)
            rows = getattr(result, "rows", None)
            columns = getattr(result, "columns", None)
            cursor.execute(
                """
                INSERT INTO paper_tables (filename, page, rows, columns)
                VALUES (?, ?, ?, ?)
                """,
                (filename, page, rows, columns),
            )
        conn.commit()
    finally:
        conn.close()


def save_figure_results(
    filename: str,
    results: Iterable[object],
    db_path: Optional[str] = None,
) -> None:
    conn = _connect(db_path)
    try:
        cursor = conn.cursor()
        for result in results:
            page = getattr(result, "page", None)
            figure_id = getattr(result, "figure_id", None)
            caption = getattr(result, "caption", None)
            cursor.execute(
                """
                INSERT INTO paper_figures (filename, page, figure_id, caption)
                VALUES (?, ?, ?, ?)
                """,
                (filename, page, figure_id, caption),
            )
        conn.commit()
    finally:
        conn.close()


def save_citations(
    filename: str,
    citations: Iterable[dict],
    db_path: Optional[str] = None,
) -> None:
    conn = _connect(db_path)
    try:
        cursor = conn.cursor()
        for citation in citations:
            cursor.execute(
                """
                INSERT INTO paper_citations (filename, author, year, raw_text)
                VALUES (?, ?, ?, ?)
                """,
                (
                    filename,
                    citation.get("author"),
                    citation.get("year"),
                    citation.get("raw_text"),
                ),
            )
        conn.commit()
    finally:
        conn.close()
