"""
STORM STARTUP DEPENDENCY CHECKER
Zero-Error Architecture: Verify all dependencies before system start
"""
import sys
import os
import sqlite3
import importlib
from pathlib import Path

class DependencyChecker:
    """
    Checks all STORM system dependencies before startup.
    Fails fast with clear error messages if any dependency is missing.
    """
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []
    
    def check_all(self) -> bool:
        """Run all dependency checks. Returns True if all critical checks pass."""
        print("=" * 60)
        print("STORM STARTUP DEPENDENCY CHECK")
        print("=" * 60)
        
        # Critical checks (system won't work without these)
        self._check_python_version()
        self._check_core_packages()
        self._check_database()
        self._check_llm_gateway()
        self._check_storm_modules()
        
        # Non-critical checks (system works with degradation)
        self._check_optional_services()
        
        # Print summary
        self._print_summary()
        
        return len(self.errors) == 0
    
    def _check_python_version(self):
        """Check Python version."""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            self.passed.append(f"Python {version.major}.{version.minor}.{version.micro}")
        else:
            self.errors.append(f"Python 3.9+ required, found {version.major}.{version.minor}")
    
    def _check_core_packages(self):
        """Check required Python packages."""
        required = [
            ('numpy', 'numpy'),
            ('requests', 'requests'),
            ('sentence_transformers', 'sentence-transformers'),
            ('faiss', 'faiss-cpu'),
            ('sklearn', 'scikit-learn'),
            ('networkx', 'networkx'),
            ('pdfplumber', 'pdfplumber'),
            ('bs4', 'beautifulsoup4'),
        ]
        
        optional = [
            ('tabula', 'tabula-py'),
            ('fitz', 'PyMuPDF'),
            ('unpywall', 'unpywall'),
            ('telethon', 'Telethon'),
        ]
        
        for module, package in required:
            try:
                importlib.import_module(module)
                self.passed.append(f"Package: {package}")
            except ImportError:
                self.errors.append(f"Missing package: {package} (pip install {package})")
        
        for module, package in optional:
            try:
                importlib.import_module(module)
                self.passed.append(f"Optional: {package}")
            except ImportError:
                self.warnings.append(f"Optional package not installed: {package}")
    
    def _check_database(self):
        """Check database exists and has required tables."""
        db_path = Path(__file__).parent / "academic_brain.db"
        
        if not db_path.exists():
            self.errors.append(f"Database not found: {db_path}")
            return
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            
            required_tables = ['theories', 'metadata', 'embeddings']
            for table in required_tables:
                if table in tables:
                    self.passed.append(f"Table: {table}")
                else:
                    # Try to run migration
                    self.warnings.append(f"Table missing: {table} (run db_migrate.py)")
            
            conn.close()
        except sqlite3.Error as e:
            self.errors.append(f"Database error: {e}")
    
    def _check_llm_gateway(self):
        """Check LLM Gateway is functional."""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from storm_modules.llm_gateway import get_llm_gateway
            
            llm = get_llm_gateway()
            if llm.provider:
                self.passed.append(f"LLM Gateway: {llm.provider.upper()}")
            else:
                self.errors.append("No LLM provider available (Groq/OpenRouter/Ollama)")
        except Exception as e:
            self.errors.append(f"LLM Gateway error: {e}")
    
    def _check_storm_modules(self):
        """Check all STORM modules can be imported."""
        modules = [
            'storm_modules.thesis_generator',
            'storm_modules.theory_builder',
            'storm_modules.hypothesis_generator',
            'storm_modules.gap_finder',
            'storm_modules.claim_extractor',
            'storm_modules.config',
        ]
        
        sys.path.insert(0, str(Path(__file__).parent))
        
        for module in modules:
            try:
                importlib.import_module(module)
                self.passed.append(f"Module: {module.split('.')[-1]}")
            except Exception as e:
                self.errors.append(f"Module import failed: {module} - {e}")
    
    def _check_optional_services(self):
        """Check optional services (GROBID, Ollama)."""
        import requests
        
        # GROBID
        try:
            r = requests.get('http://localhost:8070/api/isalive', timeout=2)
            if r.status_code == 200:
                self.passed.append("GROBID: Running")
            else:
                self.warnings.append("GROBID: Not responding correctly")
        except:
            self.warnings.append("GROBID: Not running (Docker Desktop may be closed)")
        
        # Ollama
        try:
            r = requests.get('http://localhost:11434/api/tags', timeout=2)
            if r.status_code == 200:
                self.passed.append("Ollama: Running")
            else:
                self.warnings.append("Ollama: Not responding correctly")
        except:
            self.warnings.append("Ollama: Not running")
    
    def _print_summary(self):
        """Print dependency check summary."""
        print("\n" + "-" * 60)
        print("SUMMARY")
        print("-" * 60)
        
        if self.passed:
            print(f"\n✓ PASSED ({len(self.passed)}):")
            for item in self.passed:
                print(f"  ✓ {item}")
        
        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for item in self.warnings:
                print(f"  ⚠ {item}")
        
        if self.errors:
            print(f"\n✗ ERRORS ({len(self.errors)}):")
            for item in self.errors:
                print(f"  ✗ {item}")
        
        print("\n" + "=" * 60)
        if self.errors:
            print("STATUS: ❌ FAILED - Fix errors before starting STORM")
        else:
            print("STATUS: ✅ READY - All critical dependencies available")
        print("=" * 60)


def main():
    checker = DependencyChecker()
    success = checker.check_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
