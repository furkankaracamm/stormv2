
import sqlite3
import os

DB_PATH = "academic_brain.db"
if not os.path.exists(DB_PATH):
    print("No database found.")
    exit()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check for keywords that should have been purged
keywords = ["nucleon", "galaxy", "physics", "meteorite"]
polluted_count = 0

print(f"Checking {DB_PATH} for pollution...")

for kw in keywords:
    cursor.execute(f"SELECT COUNT(*) FROM metadata WHERE content LIKE '%{kw}%'")
    count = cursor.fetchone()[0]
    if count > 0:
        print(f"[POLLUTION] Found {count} rows containing '{kw}'")
        polluted_count += count
    else:
        print(f"[CLEAN] No trace of '{kw}'")

print("-" * 30)
if polluted_count == 0:
    print("VERDICT: BRAIN IS CLEAN.")
else:
    print(f"VERDICT: {polluted_count} POLLUTED ROWS FOUND.")

conn.close()
