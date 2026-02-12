"""
STORM Comprehensive Problem Detector
Identifies all potential system issues
"""
import os
import sys
import sqlite3
import importlib

print("=" * 60)
print("🔍 STORM PROBLEM DETECTOR")
print("=" * 60)

issues = []
warnings = []

# 1. Check for stale lock file
LOCK_FILE = ".storm_lock"
if os.path.exists(LOCK_FILE):
    try:
        with open(LOCK_FILE, 'r') as f:
            pid = int(f.read().strip())
        import psutil
        if psutil.pid_exists(pid):
            issues.append(f"⛔ STORM zaten çalışıyor (PID: {pid})")
        else:
            warnings.append(f"⚠️ Eski lock dosyası bulundu (temizlenecek)")
            os.remove(LOCK_FILE)
    except:
        warnings.append("⚠️ Lock dosyası okunamadı")

# 2. Database integrity
print("\n[1] Database Integrity...")
dbs = [
    ("storm_persistent_brain/metadata.db", ["metadata", "processed_files"]),
    ("storm_data/academic_brain.db", ["theories", "paper_claims", "detected_gaps"]),
]
for db_path, tables in dbs:
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check")
            result = cur.fetchone()[0]
            if result != "ok":
                issues.append(f"⛔ {db_path}: Veritabanı bozuk!")
            else:
                print(f"   ✓ {db_path}: OK")
            
            # Check tables exist
            for table in tables:
                cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if not cur.fetchone():
                    warnings.append(f"⚠️ {db_path}: '{table}' tablosu yok")
            conn.close()
        except Exception as e:
            issues.append(f"⛔ {db_path}: {e}")
    else:
        warnings.append(f"⚠️ {db_path} bulunamadı")

# 3. Database sync check
print("\n[2] Database Sync...")
try:
    conn = sqlite3.connect("storm_persistent_brain/metadata.db")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM metadata")
    m = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM processed_files")
    p = cur.fetchone()[0]
    if m != p:
        issues.append(f"⛔ Database desync: metadata={m}, processed={p}")
    else:
        print(f"   ✓ Synced: {m} records")
    conn.close()
except Exception as e:
    warnings.append(f"⚠️ Sync check failed: {e}")

# 4. Critical imports
print("\n[3] Critical Imports...")
imports = [
    ("sentence_transformers", "SentenceTransformer"),
    ("faiss", "FAISS"),
    ("telethon", "Telethon"),
    ("pdfplumber", "pdfplumber"),
    ("scidownl", "scidownl"),
    ("requests", "requests"),
    ("bs4", "BeautifulSoup"),
    ("networkx", "NetworkX"),
    ("psutil", "psutil"),
]
for module, name in imports:
    try:
        importlib.import_module(module)
        print(f"   ✓ {name}")
    except ImportError as e:
        issues.append(f"⛔ {name} import hatası: {e}")

# 5. scidownl API check
print("\n[4] scidownl API Check...")
try:
    from scidownl import scihub_download
    print("   ✓ scihub_download fonksiyonu mevcut")
except ImportError:
    try:
        from scidownl import SciHub
        issues.append("⛔ ESKİ scidownl API! Güncelleme gerekli: pip install scidownl --upgrade")
    except:
        issues.append("⛔ scidownl tamamen bozuk")

# 6. Storm modules
print("\n[5] Storm Modules...")
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
        importlib.import_module(mod)
        print(f"   ✓ {mod.split('.')[-1]}")
    except Exception as e:
        issues.append(f"⛔ {mod}: {str(e)[:50]}")

# 7. File system
print("\n[6] File System...")
paths = [
    ("storm_data/pdfs", "dir"),
    ("storm_data/library", "dir"),
    ("storm_persistent_brain", "dir"),
    ("run_storm.bat", "file"),
    ("storm_commander.py", "file"),
]
for path, ptype in paths:
    if ptype == "dir":
        if not os.path.isdir(path):
            issues.append(f"⛔ Klasör yok: {path}")
        else:
            print(f"   ✓ {path}/")
    else:
        if not os.path.isfile(path):
            issues.append(f"⛔ Dosya yok: {path}")
        else:
            print(f"   ✓ {path}")

# 8. Duplicate PDF detection
print("\n[7] Duplicate PDF Detection...")
pdf_dir = "storm_data/pdfs"
if os.path.isdir(pdf_dir):
    files = os.listdir(pdf_dir)
    seen = {}
    duplicates = 0
    for f in files:
        base = f[:30]  # First 30 chars
        if base in seen:
            duplicates += 1
        seen[base] = True
    if duplicates > 10:
        warnings.append(f"⚠️ {duplicates} olası duplicate PDF tespit edildi")
    else:
        print(f"   ✓ Duplicate check passed")

# 9. Stub file detection (broken downloads)
print("\n[8] Broken Download Detection...")
stub_count = 0
if os.path.isdir(pdf_dir):
    for f in os.listdir(pdf_dir):
        fpath = os.path.join(pdf_dir, f)
        if os.path.isfile(fpath) and os.path.getsize(fpath) < 40000:
            stub_count += 1
    if stub_count > 20:
        warnings.append(f"⚠️ {stub_count} bozuk indirme (stub file) tespit edildi")
    else:
        print(f"   ✓ Stub file check passed ({stub_count} small files)")

# 10. Environment variables (info only)
print("\n[9] Environment Variables...")
env_vars = ["STORM_ENABLE_GROBID_FULL", "STORM_ENABLE_DEEP_ANALYSIS", "STORM_ENABLE_LLM", "GROQ_API_KEY"]
for var in env_vars:
    val = os.environ.get(var)
    if val:
        print(f"   ✓ {var} = {val[:10]}..." if len(str(val)) > 10 else f"   ✓ {var} = {val}")
    else:
        print(f"   ⓘ {var} = (run_storm.bat ile ayarlanacak)")

# REPORT
print("\n" + "=" * 60)
print("📋 SONUÇ RAPORU")
print("=" * 60)

if issues:
    print(f"\n🔴 KRİTİK SORUNLAR ({len(issues)}):")
    for i in issues:
        print(f"   {i}")

if warnings:
    print(f"\n🟡 UYARILAR ({len(warnings)}):")
    for w in warnings:
        print(f"   {w}")

if not issues and not warnings:
    print("\n🟢 TÜM KONTROLLER GEÇTİ! Sistem hazır.")
elif not issues:
    print(f"\n🟢 Kritik sorun yok, {len(warnings)} uyarı var.")
else:
    print(f"\n🔴 {len(issues)} kritik sorun düzeltilmeli!")

print("=" * 60)
