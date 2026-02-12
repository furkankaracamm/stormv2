"""
GROBID Batch Processor - Process all existing PDFs with GROBID
Run: python grobid_batch.py
"""
import os
import sys
import sqlite3
import time

# Set environment variables
os.environ["STORM_ENABLE_GROBID_FULL"] = "1"
os.environ["STORM_ENABLE_PARS_CIT"] = "1"
os.environ["STORM_ENABLE_TABLE_EXTRACT"] = "1"
os.environ["STORM_ENABLE_FIGURE_EXTRACT"] = "1"

from storm_modules.methods_extractor import GROBIDMethodsExtractor
from storm_modules.pars_cit import ParsCitExtractor
from storm_modules.table_extractor import TableExtractor
from storm_modules.figure_extractor import DeepFiguresExtractor
from storm_modules.extraction_store import save_citations, save_table_results, save_figure_results

# Configuration
PDF_DIRS = [
    "storm_data/pdfs",
    "storm_data/library/Dead Internet Theory"
]
MIN_PDF_SIZE = 50000  # Skip files smaller than 50KB (stubs)

def get_all_pdfs():
    """Find all valid PDFs recursively."""
    pdfs = []
    for base_dir in PDF_DIRS:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for f in files:
                    if f.endswith('.pdf'):
                        path = os.path.join(root, f)
                        if os.path.getsize(path) > MIN_PDF_SIZE:
                            pdfs.append(path)
    return pdfs

def process_pdf(pdf_path, extractor, parscit, table_ext, figure_ext):
    """Process a single PDF with all GROBID extractors."""
    filename = os.path.basename(pdf_path)
    results = {"citations": 0, "tables": 0, "figures": 0, "methods": False}
    
    try:
        # 1. Methods extraction (full GROBID)
        methods = extractor.extract_methods_from_pdf(pdf_path)
        if methods:
            extractor.save_to_database(filename, methods)
            results["methods"] = True
        
        # 2. Citations (ParsCit-style)
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages[:20])
            
            citations = parscit.parse_citations_from_text(text)
            if citations:
                save_citations(filename, citations)
                results["citations"] = len(citations)
        except Exception as e:
            pass
        
        # 3. Tables
        if table_ext.is_available():
            try:
                pages = table_ext.detect_table_pages(pdf_path)
                if pages:
                    tables = table_ext.extract_tables_from_pdf(pdf_path, pages=",".join(str(p) for p in pages))
                    if tables:
                        save_table_results(filename, tables)
                        results["tables"] = len(tables)
            except:
                pass
        
        # 4. Figures
        try:
            figures = figure_ext.extract_figures_from_pdf(pdf_path)
            if figures:
                save_figure_results(filename, figures)
                results["figures"] = len(figures)
        except:
            pass
            
    except Exception as e:
        print(f"  ERROR: {e}")
    
    return results

def main():
    print("=" * 60)
    print("🔬 GROBID BATCH PROCESSOR")
    print("=" * 60)
    
    # Check GROBID
    try:
        import requests
        r = requests.get("http://localhost:8070/api/version", timeout=5)
        if r.status_code != 200:
            print("❌ GROBID not responding! Start Docker first.")
            return
        print(f"✓ GROBID v{r.text} bağlandı")
    except:
        print("❌ GROBID bağlantısı başarısız! Docker çalışıyor mu?")
        return
    
    # Initialize extractors
    print("\n[1] Initializing extractors...")
    extractor = GROBIDMethodsExtractor()
    parscit = ParsCitExtractor()
    table_ext = TableExtractor()
    figure_ext = DeepFiguresExtractor()
    
    # Find PDFs
    print("\n[2] Scanning for PDFs...")
    pdfs = get_all_pdfs()
    print(f"   Found {len(pdfs)} valid PDFs (>{MIN_PDF_SIZE/1024:.0f}KB)")
    
    if not pdfs:
        print("No PDFs found!")
        return
    
    # Process
    print(f"\n[3] Processing {len(pdfs)} PDFs...")
    print("-" * 60)
    
    stats = {"total": 0, "citations": 0, "tables": 0, "figures": 0, "methods": 0}
    
    for i, pdf in enumerate(pdfs):
        filename = os.path.basename(pdf)[:40]
        print(f"[{i+1}/{len(pdfs)}] {filename}...", end=" ", flush=True)
        
        results = process_pdf(pdf, extractor, parscit, table_ext, figure_ext)
        
        stats["total"] += 1
        stats["citations"] += results["citations"]
        stats["tables"] += results["tables"]
        stats["figures"] += results["figures"]
        if results["methods"]:
            stats["methods"] += 1
        
        status = []
        if results["methods"]:
            status.append("M")
        if results["citations"]:
            status.append(f"C:{results['citations']}")
        if results["tables"]:
            status.append(f"T:{results['tables']}")
        if results["figures"]:
            status.append(f"F:{results['figures']}")
        
        print(" | ".join(status) if status else "skip")
        
        # Small delay to avoid overwhelming GROBID
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SONUÇ")
    print("=" * 60)
    print(f"İşlenen PDF: {stats['total']}")
    print(f"Methods çıkarılan: {stats['methods']}")
    print(f"Citations çıkarılan: {stats['citations']}")
    print(f"Tables çıkarılan: {stats['tables']}")
    print(f"Figures çıkarılan: {stats['figures']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
