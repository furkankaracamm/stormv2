import os
import pandas as pd
from pathlib import Path
from typing import List, Optional

class TableExtractor:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "brain" / "tables"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        try:
            import tabula  # noqa: F401
            return True
        except ImportError:
            return False

    def detect_table_pages(self, pdf_path: str) -> str:
        """
        Detects pages containing tables. 
        For now, defaults to 'all' to ensure comprehensive extraction.
        """
        return "all"

    def extract_tables_from_pdf(self, pdf_path: str, paper_id: str, pages: str = "all") -> List[str]:
        """
        Extracts tables from PDF and saves them as CSV files.
        Returns a list of paths to the saved CSV files.
        """
        if not self.is_available():
            print("[TableExtractor] tabula-py not found. Skipping table extraction.")
            return []

        import tabula
        saved_paths = []
        try:
            dfs = tabula.read_pdf(pdf_path, pages=pages, multiple_tables=True)
            for i, df in enumerate(dfs):
                if df.empty: continue
                
                safe_id = "".join([c if c.isalnum() else "_" for c in paper_id])
                csv_filename = f"{safe_id}_table_{i+1}.csv"
                csv_path = self.output_dir / csv_filename
                
                df.to_csv(csv_path, index=False)
                saved_paths.append(str(csv_path))
                print(f"  ✓ Saved table to {csv_filename}")
        except Exception as e:
            print(f"[TableExtractor Error] {e}")

        return saved_paths

    def dry_run(self) -> List[str]:
        return ["example_table_1.csv"]
