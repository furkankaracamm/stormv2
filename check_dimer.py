
import sqlite3
import os

DB_PATH = "storm_persistent_brain/metadata.db"
if not os.path.exists(DB_PATH):
    print("No database found.")
    exit()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check specifically for the questionable paper
keyword = "Dimer"
cursor.execute(f"SELECT COUNT(*) FROM metadata WHERE content LIKE '%{keyword}%'")
count = cursor.fetchone()[0]

if count > 0:
    print(f"[FOUND] {count} rows containing '{keyword}'. It was saved.")
    # Check if it has any redeeming qualities (e.g., simulation, model)
    cursor.execute(f"SELECT content FROM metadata WHERE content LIKE '%{keyword}%' LIMIT 1")
    sample = cursor.fetchone()[0]
    print(f"Sample content: {sample[:200]}...")
else:
    print(f"[CLEAN] No trace of '{keyword}' in the permanent brain.")

conn.close()
