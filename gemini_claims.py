"""
Gemini Claim Extractor - Process all PDFs with Gemini LLM
Uses ONLY Gemini API for high-quality claim extraction
"""
import os
import sys
import sqlite3
import time
import json
import pdfplumber
from pathlib import Path

# Add storm to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storm_modules.claim_extractor import extract_claims_llm, save_claims_to_db, extract_claims
from storm_modules.config import get_academic_brain_db_path

# Configuration
PDF_DIRS = [
    "storm_data/pdfs",
    "storm_data/library/Dead Internet Theory"
]
MIN_PDF_SIZE = 50000  # Skip stubs

def get_all_pdfs():
    """Find all valid PDFs."""
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

def extract_text_from_pdf(pdf_path, max_pages=30):
    """Extract text from PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages[:max_pages]
            text = "\n".join(page.extract_text() or "" for page in pages)
            return text
    except Exception as e:
        print(f"  PDF Error: {e}")
        return ""

def main():
    print("=" * 60)
    print("🔬 GEMINI CLAIM EXTRACTOR")
    print("=" * 60)
    
    # Get PDFs
    pdfs = get_all_pdfs()
    print(f"Found {len(pdfs)} PDFs to process")
    
    if not pdfs:
        print("No PDFs found!")
        return
    
    # Clear existing claims (optional - fresh extraction)
    db_path = str(get_academic_brain_db_path())
    print(f"Database: {db_path}")
    
    # Process
    stats = {"processed": 0, "claims": 0, "failed": 0, "llm_success": 0, "regex_fallback": 0}
    
    for i, pdf in enumerate(pdfs):
        filename = os.path.basename(pdf)[:45]
        print(f"\n[{i+1}/{len(pdfs)}] {filename}")
        
        # Check if already processed (optional)
        # ...
        
        # Extract text
        text = extract_text_from_pdf(pdf)
        if len(text) < 500:
            print("  ⚠️ Too short, skipping")
            continue
        
        # Extract claims with Gemini
        try:
            # Try LLM first
            claims = extract_claims_llm(text, max_claims=12)
            source = "LLM"
            
            # Verify if LLM actually worked (check for confidence > 0.8)
            high_conf = [c for c in claims if c.get("confidence", 0) > 0.8]
            if not high_conf:
                # If only weak claims, maybe LLM failed. Try Regex as augment
                regex_claims = extract_claims(text, max_claims=5)
                # Merge unique
                existing_texts = {c["text"] for c in claims}
                for rc in regex_claims:
                    if rc["text"] not in existing_texts:
                        claims.append(rc)
                        source += "+Regex"
            
            if claims:
                # Save to DB
                save_claims_to_db(os.path.basename(pdf), claims, db_path)
                stats["claims"] += len(claims)
                print(f"  ✅ {len(claims)} claims extracted ({source})")
                
                # Show first claim
                if claims:
                    first = claims[0]
                    print(f"     → {first.get('type', 'CLAIM')}: {first.get('text', '')[:60]}...")
                
                if "LLM" in source:
                    stats["llm_success"] += 1
                else:
                    stats["regex_fallback"] += 1
            else:
                print("  ⚠️ No claims found")
                
            stats["processed"] += 1
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            stats["failed"] += 1
        
        # Rate limit: 15 req/min = 4 seconds/req
        # We add buffer to 5s to be safe
        time.sleep(5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SONUÇ")
    print("=" * 60)
    print(f"İşlenen PDF: {stats['processed']}")
    print(f"Toplam Claim: {stats['claims']}")
    print(f"LLM Başarılı: {stats['llm_success']}")
    print(f"Regex Fallback: {stats['regex_fallback']}")
    print(f"Hatalı: {stats['failed']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
