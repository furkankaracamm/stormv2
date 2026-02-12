import sqlite3
import os

print("=" * 50)
print("STORM SYSTEM HEALTH CHECK")
print("=" * 50)

# 1. Database Check
print("\n[1] DATABASE STATUS")
try:
    c = sqlite3.connect('storm_persistent_brain/metadata.db')
    cur = c.cursor()
    cur.execute('SELECT COUNT(*) FROM metadata')
    m = cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM processed_files')
    p = cur.fetchone()[0]
    sync_status = "✓ SYNCED" if m == p else f"✗ DESYNC (gap: {m-p})"
    print(f"   Metadata Rows: {m}")
    print(f"   Processed Files: {p}")
    print(f"   Status: {sync_status}")
    c.close()
except Exception as e:
    print(f"   ERROR: {e}")

# 2. Import Check
print("\n[2] CRITICAL IMPORTS")
imports = [
    ("sentence_transformers", "SentenceTransformer"),
    ("faiss", "FAISS"),
    ("telethon", "TelegramClient"),
    ("pdfplumber", "pdfplumber"),
    ("scidownl", "scihub_download"),
    ("requests", "requests"),
    ("bs4", "BeautifulSoup"),
    ("networkx", "NetworkX"),
]
for module, name in imports:
    try:
        __import__(module)
        print(f"   {name}: ✓")
    except Exception as e:
        print(f"   {name}: ✗ ({e})")

# 3. Module Check
print("\n[3] STORM MODULES")
modules = [
    "storm_modules.methods_extractor",
    "storm_modules.openalex_client",
    "storm_modules.pars_cit",
    "storm_modules.table_extractor",
    "storm_modules.figure_extractor",
    "storm_modules.ontology",
    "storm_modules.gap_finder",
    "storm_modules.llm_gateway",
    "storm_modules.claim_extractor",
    "storm_modules.librarian",
]
for mod in modules:
    try:
        __import__(mod)
        print(f"   {mod.split('.')[-1]}: ✓")
    except Exception as e:
        print(f"   {mod.split('.')[-1]}: ✗ ({str(e)[:40]})")

# 4. Environment Variables
print("\n[4] ENVIRONMENT VARIABLES")
env_vars = [
    "STORM_ENABLE_GROBID_FULL",
    "STORM_ENABLE_DEEP_ANALYSIS",
    "STORM_ENABLE_LLM",
    "GROQ_API_KEY",
]
for var in env_vars:
    val = os.environ.get(var)
    if val:
        display = val[:10] + "..." if len(val) > 10 else val
        print(f"   {var}: {display}")
    else:
        print(f"   {var}: NOT SET (will be set by run_storm.bat)")

# 5. File System Check
print("\n[5] FILE SYSTEM")
paths = [
    ("PDFs Inbox", "storm_data/pdfs"),
    ("Library", "storm_data/library"),
    ("Brain Index", "storm_persistent_brain/brain.faiss"),
    ("Metadata DB", "storm_persistent_brain/metadata.db"),
]
for name, path in paths:
    if os.path.exists(path):
        if os.path.isdir(path):
            count = len([f for f in os.listdir(path) if not os.path.isdir(os.path.join(path, f))])
            print(f"   {name}: ✓ ({count} files)")
        else:
            size = os.path.getsize(path)
            print(f"   {name}: ✓ ({size/1024:.1f} KB)")
    else:
        print(f"   {name}: ✗ MISSING")

print("\n" + "=" * 50)
print("CHECK COMPLETE")
print("=" * 50)
