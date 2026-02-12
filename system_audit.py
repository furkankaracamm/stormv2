import sys
import os
import sqlite3
sys.path.insert(0, 'C:\\Users\\Enes\\.gemini\\antigravity\\scratch\\storm')

print("=" * 60)
print("ÖLÜMCÜL OBJEKTİFLİK SİSTEM DENETİMİ")
print("=" * 60)

# 1. DATABASE
print("\n[1] DATABASE DENETIMI")
from storm_modules.config import get_academic_brain_db_path
db_path = str(get_academic_brain_db_path())
if os.path.exists(db_path):
    sz = os.path.getsize(db_path) / 1024 / 1024
    print(f"  ✓ DB Mevcut: {sz:.2f} MB")
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    print(f"  ✓ Tablo sayısı: {len(tables)}")
    
    for t in ['theories', 'metadata', 'embeddings', 'paper_claims']:
        try:
            c.execute(f"SELECT COUNT(*) FROM {t}")
            cnt = c.fetchone()[0]
            status = "✓" if cnt > 0 else "⚠"
            print(f"    {status} {t}: {cnt}")
        except Exception as e:
            print(f"    ✗ {t}: {e}")
    
    conn.close()
else:
    print(f"  ✗ DB BULUNAMADI: {db_path}")

# 2. DOWNLOADER IMPORTS
print("\n[2] DOWNLOADER MODULLERI")
downloaders = [
    ("ArxivDownloader", "downloaders.arxiv_downloader", "ArxivDownloader"),
    ("SemanticScholar", "downloaders.semantic_scholar", "SemanticScholarDownloader"),
    ("TelegramBot", "downloaders.telegram_bot_integration", "TelegramBotIntegration"),
    ("AnnasArchive", "downloaders.annas_archive", "AnnasArchiveDownloader"),
]
for name, mod, cls in downloaders:
    try:
        exec(f"from {mod} import {cls}")
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {str(e)[:40]}")

# 3. STORM MODULES
print("\n[3] STORM MODULLERI")
modules = [
    ("LLMGateway", "storm_modules.llm_gateway", "get_llm_gateway"),
    ("TheoryBuilder", "storm_modules.theory_builder", "TheoryDatabaseBuilder"),
    ("ThesisGenerator", "storm_modules.thesis_generator", "ThesisGenerator"),
    ("HypothesisGen", "storm_modules.hypothesis_generator", "HypothesisGenerator"),
    ("GapFinder", "storm_modules.gap_finder", "GapFinder"),
    ("ClaimExtractor", "storm_modules.claim_extractor", "extract_claims"),
]
for name, mod, cls in modules:
    try:
        exec(f"from {mod} import {cls}")
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {str(e)[:40]}")

# 4. API CONNECTIVITY
print("\n[4] API BAĞLANTILARI")
import requests

# Groq
try:
    r = requests.get('https://api.groq.com/openai/v1/models', 
        headers={'Authorization': 'Bearer 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'}, timeout=5)
    print(f"  ✓ Groq API: {r.status_code}")
except Exception as e:
    print(f"  ✗ Groq API: {e}")

# GROBID
try:
    r = requests.get('http://localhost:8070/api/isalive', timeout=2)
    print(f"  ✓ GROBID: {r.status_code}")
except:
    print("  ✗ GROBID: NOT RUNNING (Docker Desktop kapalı)")

# Ollama
try:
    r = requests.get('http://localhost:11434/api/tags', timeout=2)
    print(f"  ✓ Ollama: {r.status_code}")
except:
    print("  ✗ Ollama: NOT RUNNING")

print("\n" + "=" * 60)
print("SONUÇ: Test tamamlandı")
print("=" * 60)
