import sqlite3
import os

print("=" * 50)
print("GROBID VERİTABANI KONTROLÜ")
print("=" * 50)

dbs = [
    "academic_brain.db",
    "storm_data/academic_brain.db"
]

for db in dbs:
    print(f"\n📁 {db}:")
    if os.path.exists(db):
        try:
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            
            # List tables
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            print(f"   Tablolar: {tables}")
            
            # Check specific tables
            checks = [
                ("paper_citations", "Citations"),
                ("paper_tables", "Tables"),
                ("paper_figures", "Figures"),
                ("paper_claims", "Claims"),
            ]
            
            for table, name in checks:
                if table in tables:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    print(f"   ✓ {name}: {count} kayıt")
                else:
                    print(f"   ✗ {name}: Tablo yok")
            
            conn.close()
        except Exception as e:
            print(f"   HATA: {e}")
    else:
        print("   ✗ Dosya bulunamadı")

print("\n" + "=" * 50)
