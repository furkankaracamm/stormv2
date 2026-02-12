from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


from storm_modules.config import get_academic_brain_db_path


class GROBIDMethodsExtractor:
    def __init__(self, grobid_url: str = "http://localhost:8070", db_path: str | None = None):
        self.grobid_url = grobid_url
        self.db_path = db_path or str(get_academic_brain_db_path())

    def extract_methods_from_pdf(self, pdf_path: str) -> Optional[Dict]:
        try:
            xml_response = self._call_grobid_api(pdf_path)
            if not xml_response:
                return None

            methods_text = self._extract_methods_section(xml_response)
            if not methods_text:
                return None

            return {
                "sample_size": self._extract_sample_size(methods_text),
                "statistical_tests": self._extract_statistical_tests(methods_text),
                "measures": self._extract_measures(methods_text),
                "design_type": self._classify_design(methods_text),
                "raw_methods_text": methods_text[:2000],
            }
        except Exception as exc:
            print(f"[METHODS EXTRACTOR ERROR] {pdf_path}: {exc}")
            return None

    def _call_grobid_api(self, pdf_path: str) -> Optional[str]:
        try:
            import requests
        except ImportError as exc:
            print("[GROBID] Missing dependency: requests. Install it to enable extraction.")
            raise RuntimeError("requests is required for GROBID extraction") from exc

        try:
            with open(pdf_path, "rb") as handle:
                response = requests.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files={"input": handle},
                    timeout=60,
                )

            if response.status_code == 200:
                return response.text
            print(f"[GROBID] Status {response.status_code} for {pdf_path}")
            return None
        except requests.exceptions.Timeout:
            print(f"[GROBID] Timeout for {pdf_path}")
            return None
        except Exception as exc:
            print(f"[GROBID] Error: {exc}")
            return None

    def _extract_methods_section(self, xml_data: str) -> Optional[str]:
        try:
            try:
                from bs4 import BeautifulSoup
            except ImportError as exc:
                print("[XML PARSE] Missing dependency: beautifulsoup4. Install it to parse methods.")
                raise RuntimeError("beautifulsoup4 is required for XML parsing") from exc

            soup = BeautifulSoup(xml_data, "xml")
            methods_div = (
                soup.find("div", {"type": "methods"})
                or soup.find("div", {"type": "materials"})
                or soup.find("div", {"type": "methodology"})
            )
            if methods_div:
                return methods_div.get_text(separator=" ", strip=True)

            for div in soup.find_all("div"):
                head = div.find("head")
                if head and "method" in head.get_text().lower():
                    return div.get_text(separator=" ", strip=True)
            return None
        except Exception as exc:
            print(f"[XML PARSE ERROR] {exc}")
            return None

    def _extract_sample_size(self, text: str) -> Optional[int]:
        patterns = [
            r"[Nn]\s*=\s*(\d+)",
            r"(\d+)\s+participants?",
            r"sample\s+(?:of|size)?\s*[:\-]?\s*(\d+)",
            r"respondents?\s*[:\-]?\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                if 10 <= value <= 100000:
                    return value
        return None

    def _extract_statistical_tests(self, text: str) -> List[str]:
        tests: List[str] = []
        text_lower = text.lower()
        test_patterns = {
            "regression": r"regression|linear model|glm",
            "anova": r"anova|analysis of variance|f-test",
            "ttest": r"t-test|t test|paired t",
            "chi_square": r"chi-?square|χ2|chi2",
            "correlation": r"correlation|pearson|spearman",
            "sem": r"structural equation|sem|path analysis",
            "mediation": r"mediation|indirect effect|process|sobel",
            "moderation": r"moderation|interaction effect",
            "multilevel": r"multilevel|hierarchical linear|hlm|mixed model",
            "factor_analysis": r"factor analysis|efa|cfa|principal component",
            "mann_whitney": r"mann.whitney|wilcoxon",
            "kruskal": r"kruskal.wallis",
        }

        for test_name, pattern in test_patterns.items():
            if re.search(pattern, text_lower):
                tests.append(test_name)
        return tests

    def _extract_measures(self, text: str) -> List[Dict]:
        measures: List[Dict] = []
        pattern = r"([A-Z][A-Za-z\s]+(?:Scale|Inventory|Questionnaire|Index|Measure))\s*\(([^)]+)\)"
        matches = re.findall(pattern, text)
        for scale_name, citation in matches:
            measures.append({"name": scale_name.strip(), "citation": citation.strip()})

        likert_match = re.search(r"(\d+)[-\s]point\s+likert", text.lower())
        if likert_match:
            measures.append(
                {"name": f"{likert_match.group(1)}-point Likert Scale", "citation": "Standard"}
            )
        return measures

    def _classify_design(self, text: str) -> str:
        text_lower = text.lower()
        if re.search(r"experiment|random\s+assignment|control\s+group", text_lower):
            return "experiment"
        if re.search(r"survey|questionnaire|self[-\s]report", text_lower):
            return "survey"
        if re.search(r"content\s+analysis|coding|inter[-\s]?coder", text_lower):
            return "content_analysis"
        if re.search(r"interview|qualitative|focus\s+group", text_lower):
            return "qualitative"
        if re.search(r"longitudinal|panel|time[-\s]series", text_lower):
            return "longitudinal"
        if re.search(r"meta[-\s]analysis|systematic\s+review", text_lower):
            return "meta_analysis"
        return "unknown"

    def save_to_database(self, filename: str, methods_data: Dict) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO paper_methods
                (filename, sample_size, statistical_tests, measures, design_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    methods_data.get("sample_size"),
                    json.dumps(methods_data.get("statistical_tests", [])),
                    json.dumps(methods_data.get("measures", [])),
                    methods_data.get("design_type", "unknown"),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as exc:
            print(f"[DB SAVE ERROR] {exc}")
            return False

    def build_methods_norms_database(self) -> Dict:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT sample_size, statistical_tests, design_type FROM paper_methods")
            rows = cursor.fetchall()
            if not rows:
                conn.close()
                return {}

            norms = {
                "survey_sample_sizes": [],
                "experiment_sample_sizes": [],
                "common_tests": {},
                "total_papers": len(rows),
            }

            for sample_size, tests_json, design_type in rows:
                if sample_size:
                    if design_type == "survey":
                        norms["survey_sample_sizes"].append(sample_size)
                    elif design_type == "experiment":
                        norms["experiment_sample_sizes"].append(sample_size)

                if tests_json:
                    tests = json.loads(tests_json)
                    for test in tests:
                        norms["common_tests"][test] = norms["common_tests"].get(test, 0) + 1

            import numpy as np

            if norms["survey_sample_sizes"]:
                norms["survey_median_n"] = int(np.median(norms["survey_sample_sizes"]))
                norms["survey_mean_n"] = int(np.mean(norms["survey_sample_sizes"]))

            if norms["experiment_sample_sizes"]:
                norms["experiment_median_n"] = int(np.median(norms["experiment_sample_sizes"]))
                norms["experiment_mean_n"] = int(np.mean(norms["experiment_sample_sizes"]))

            norms_path = Path(self.db_path).parent / "methods_norms.json"
            with open(norms_path, "w", encoding="utf-8") as handle:
                json.dump(norms, handle, indent=2)

            conn.close()
            return norms
        except Exception as exc:
            print(f"[NORMS BUILD ERROR] {exc}")
            return {}

    # ============== GROBID FULL EXTRACTION ==============
    
    def extract_full_metadata(self, pdf_path: str) -> Optional[Dict]:
        """Extract all available metadata from PDF using GROBID."""
        try:
            xml_response = self._call_grobid_api(pdf_path)
            if not xml_response:
                return None
            
            return {
                "methods": self._extract_methods_section(xml_response),
                "authors": self._extract_authors(xml_response),
                "references": self._extract_references(xml_response),
                "keywords": self._extract_keywords(xml_response),
                "dates": self._extract_dates(xml_response),
                "abstract": self._extract_abstract(xml_response),
                "sample_size": self._extract_sample_size(self._extract_methods_section(xml_response) or ""),
                "statistical_tests": self._extract_statistical_tests(self._extract_methods_section(xml_response) or ""),
                "design_type": self._classify_design(self._extract_methods_section(xml_response) or ""),
            }
        except Exception as exc:
            print(f"[GROBID FULL ERROR] {pdf_path}: {exc}")
            return None

    def _extract_authors(self, xml_data: str) -> List[Dict]:
        """Extract author names, emails, and affiliations."""
        authors = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(xml_data, "xml")
            
            for author in soup.find_all("author"):
                name_parts = []
                forename = author.find("forename")
                surname = author.find("surname")
                if forename:
                    name_parts.append(forename.get_text())
                if surname:
                    name_parts.append(surname.get_text())
                
                email_tag = author.find("email")
                affil_tag = author.find("affiliation")
                
                authors.append({
                    "name": " ".join(name_parts),
                    "email": email_tag.get_text() if email_tag else None,
                    "affiliation": affil_tag.get_text(separator=" ", strip=True) if affil_tag else None
                })
        except Exception as exc:
            print(f"[AUTHOR EXTRACT ERROR] {exc}")
        return authors

    def _extract_references(self, xml_data: str) -> List[Dict]:
        """Extract structured bibliographic references."""
        refs = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(xml_data, "xml")
            
            for bibl in soup.find_all("biblStruct"):
                ref = {}
                
                # Authors
                author_names = []
                for author in bibl.find_all("author"):
                    surname = author.find("surname")
                    if surname:
                        author_names.append(surname.get_text())
                ref["author"] = ", ".join(author_names[:3])
                if len(author_names) > 3:
                    ref["author"] += " et al."
                
                # Year
                date_tag = bibl.find("date")
                if date_tag and date_tag.get("when"):
                    year_str = date_tag.get("when")[:4]
                    ref["year"] = int(year_str) if year_str.isdigit() else None
                else:
                    ref["year"] = None
                
                # Title
                title_tag = bibl.find("title", {"level": "a"}) or bibl.find("title")
                ref["title"] = title_tag.get_text(strip=True) if title_tag else None
                
                # Journal
                journal_tag = bibl.find("title", {"level": "j"})
                ref["journal"] = journal_tag.get_text(strip=True) if journal_tag else None
                
                # DOI
                doi_tag = bibl.find("idno", {"type": "DOI"})
                ref["doi"] = doi_tag.get_text(strip=True) if doi_tag else None
                
                if ref.get("title") or ref.get("author"):
                    refs.append(ref)
        except Exception as exc:
            print(f"[REF EXTRACT ERROR] {exc}")
        return refs

    def _extract_keywords(self, xml_data: str) -> List[str]:
        """Extract keywords from the document."""
        keywords = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(xml_data, "xml")
            
            for kw in soup.find_all("term"):
                text = kw.get_text(strip=True)
                if text and len(text) < 100:
                    keywords.append(text)
        except Exception as exc:
            print(f"[KEYWORD EXTRACT ERROR] {exc}")
        return keywords

    def _extract_dates(self, xml_data: str) -> List[Dict]:
        """Extract submission, acceptance, and publication dates."""
        dates = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(xml_data, "xml")
            
            for date in soup.find_all("date"):
                date_type = date.get("type", "unknown")
                date_value = date.get("when") or date.get_text(strip=True)
                if date_value:
                    dates.append({"type": date_type, "value": date_value})
        except Exception as exc:
            print(f"[DATE EXTRACT ERROR] {exc}")
        return dates

    def _extract_abstract(self, xml_data: str) -> Optional[str]:
        """Extract the abstract text."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(xml_data, "xml")
            
            abstract_div = soup.find("abstract")
            if abstract_div:
                return abstract_div.get_text(separator=" ", strip=True)
        except Exception as exc:
            print(f"[ABSTRACT EXTRACT ERROR] {exc}")
        return None

    def save_full_metadata(self, filename: str, metadata: Dict) -> bool:
        """Save all extracted metadata to database tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Authors
            for author in metadata.get("authors", []):
                cursor.execute(
                    "INSERT INTO paper_authors (filename, author_name, email, affiliation) VALUES (?, ?, ?, ?)",
                    (filename, author.get("name"), author.get("email"), author.get("affiliation"))
                )
            
            # Structured References
            for ref in metadata.get("references", []):
                cursor.execute(
                    "INSERT INTO paper_structured_refs (filename, ref_author, ref_year, ref_title, ref_journal, ref_doi) VALUES (?, ?, ?, ?, ?, ?)",
                    (filename, ref.get("author"), ref.get("year"), ref.get("title"), ref.get("journal"), ref.get("doi"))
                )
            
            # Keywords
            for kw in metadata.get("keywords", []):
                cursor.execute(
                    "INSERT INTO paper_keywords (filename, keyword) VALUES (?, ?)",
                    (filename, kw)
                )
            
            # Dates
            for date in metadata.get("dates", []):
                cursor.execute(
                    "INSERT INTO paper_dates (filename, date_type, date_value) VALUES (?, ?, ?)",
                    (filename, date.get("type"), date.get("value"))
                )
            
            # Abstract
            if metadata.get("abstract"):
                cursor.execute(
                    "INSERT OR REPLACE INTO paper_abstracts (filename, abstract_text) VALUES (?, ?)",
                    (filename, metadata.get("abstract"))
                )
            
            # Methods (existing table)
            cursor.execute(
                "INSERT INTO paper_methods (filename, sample_size, statistical_tests, measures, design_type) VALUES (?, ?, ?, ?, ?)",
                (filename, metadata.get("sample_size"), json.dumps(metadata.get("statistical_tests", [])), "[]", metadata.get("design_type", "unknown"))
            )
            
            conn.commit()
            conn.close()
            print(f"[GROBID] Saved full metadata for {filename}")
            return True
        except Exception as exc:
            print(f"[GROBID DB SAVE ERROR] {exc}")
            return False


if __name__ == "__main__":
    extractor = GROBIDMethodsExtractor()
    extractor.build_methods_norms_database()

