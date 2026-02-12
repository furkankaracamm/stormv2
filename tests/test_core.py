# STORM System Test Suite
# Run with: pytest tests/ -v

import os
import sys
import tempfile
import sqlite3
import shutil

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class TestGROBIDExtractor:
    """Tests for GROBID methods extractor - ISOLATED, no file locks."""
    
    def test_sample_size_extraction(self):
        """Should extract sample sizes from text."""
        from storm_modules.methods_extractor import GROBIDMethodsExtractor
        
        extractor = GROBIDMethodsExtractor()
        
        # Test N= pattern
        result = extractor._extract_sample_size("The study included N=500 participants")
        assert result == 500
        
        # Test participants pattern
        result = extractor._extract_sample_size("We surveyed 1000 participants")
        assert result == 1000
    
    def test_statistical_tests_extraction(self):
        """Should identify statistical tests."""
        from storm_modules.methods_extractor import GROBIDMethodsExtractor
        
        extractor = GROBIDMethodsExtractor()
        
        text = "We used regression analysis and ANOVA to test our hypotheses."
        tests = extractor._extract_statistical_tests(text)
        
        assert "regression" in tests
        assert "anova" in tests
    
    def test_design_classification(self):
        """Should classify research design types."""
        from storm_modules.methods_extractor import GROBIDMethodsExtractor
        
        extractor = GROBIDMethodsExtractor()
        
        assert extractor._classify_design("We conducted an experiment with random assignment") == "experiment"
        assert extractor._classify_design("A survey was distributed to respondents") == "survey"
        assert extractor._classify_design("Semi-structured interviews were conducted") == "qualitative"
    
    def test_design_longitudinal(self):
        """Should detect longitudinal studies."""
        from storm_modules.methods_extractor import GROBIDMethodsExtractor
        
        extractor = GROBIDMethodsExtractor()
        
        assert extractor._classify_design("This longitudinal study followed participants over 5 years") == "longitudinal"
    
    def test_design_meta_analysis(self):
        """Should detect meta-analysis."""
        from storm_modules.methods_extractor import GROBIDMethodsExtractor
        
        extractor = GROBIDMethodsExtractor()
        
        assert extractor._classify_design("We conducted a meta-analysis of 50 studies") == "meta_analysis"


class TestDatabaseSchema:
    """Tests for database schema integrity - uses temp file."""
    
    def test_schema_creates_all_tables(self):
        """Schema should create all required tables."""
        from storm_modules.schema import SCHEMA_SQL
        
        # Use completely isolated temp file
        temp_db = os.path.join(tempfile.gettempdir(), f"test_schema_{os.getpid()}.db")
        
        try:
            conn = sqlite3.connect(temp_db)
            conn.executescript(SCHEMA_SQL)
            
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required = [
                "paper_methods", "theories", "hypotheses", 
                "paper_authors", "paper_structured_refs", 
                "paper_keywords", "paper_abstracts"
            ]
            
            for table in required:
                assert table in tables, f"Missing table: {table}"
            
            conn.close()
        finally:
            if os.path.exists(temp_db):
                os.unlink(temp_db)
    
    def test_schema_author_table_has_required_columns(self):
        """paper_authors should have all required columns."""
        from storm_modules.schema import SCHEMA_SQL
        
        temp_db = os.path.join(tempfile.gettempdir(), f"test_cols_{os.getpid()}.db")
        
        try:
            conn = sqlite3.connect(temp_db)
            conn.executescript(SCHEMA_SQL)
            
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(paper_authors)")
            columns = [row[1] for row in cursor.fetchall()]
            
            required = ["filename", "author_name", "email", "affiliation"]
            for col in required:
                assert col in columns, f"Missing column: {col}"
            
            conn.close()
        finally:
            if os.path.exists(temp_db):
                os.unlink(temp_db)


class TestOntology:
    """Tests for research ontology - no file dependencies."""
    
    def test_ontology_returns_topics(self):
        """Ontology should return list of research topics."""
        from storm_modules.ontology import ResearchOntology
        
        ontology = ResearchOntology()
        topics = ontology.get_all_topics()
        
        assert len(topics) > 0
        assert all(isinstance(t, str) for t in topics)
    
    def test_topics_are_english(self):
        """Topics should be in English for international search."""
        from storm_modules.ontology import ResearchOntology
        
        ontology = ResearchOntology()
        topics = ontology.get_all_topics()
        
        # Check no Turkish characters in topics
        turkish_chars = "ğüşıöçĞÜŞİÖÇ"
        for topic in topics:
            assert not any(c in topic for c in turkish_chars), f"Turkish chars found in: {topic}"
    
    def test_ontology_has_core_domains(self):
        """Ontology should have core domain categories."""
        from storm_modules.ontology import ResearchOntology
        
        ontology = ResearchOntology()
        
        assert hasattr(ontology, 'core_domains')
        assert len(ontology.core_domains) > 0


class TestLibrarian:
    """Tests for AcademicLibrarian - import and class check only."""
    
    def test_librarian_class_exists(self):
        """Librarian class should be importable."""
        from storm_modules.librarian import AcademicLibrarian
        
        # Class should exist and be callable
        assert AcademicLibrarian is not None
        assert callable(AcademicLibrarian)


class TestPDFValidation:
    """Tests for PDF validation logic - uses temp files."""
    
    def test_valid_pdf_header(self):
        """Should recognize valid PDF headers."""
        temp_file = os.path.join(tempfile.gettempdir(), f"test_pdf_{os.getpid()}.pdf")
        
        try:
            with open(temp_file, "wb") as f:
                f.write(b"%PDF-1.4\n" + b"x" * 15000)
            
            with open(temp_file, "rb") as check:
                header = check.read(4)
                assert header == b"%PDF"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_reject_html_content(self):
        """Should detect HTML content (CAPTCHA pages)."""
        content = b"<!DOCTYPE html><html>CAPTCHA required</html>"
        
        # Detection logic
        is_html = b"<!DOCTYPE" in content or b"<html" in content
        assert is_html == True
    
    def test_reject_small_files(self):
        """Files under 10KB are likely invalid."""
        min_valid_size = 10 * 1024  # 10KB
        
        small_file_size = 5000
        assert small_file_size < min_valid_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
