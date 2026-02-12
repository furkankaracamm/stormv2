
import os
import sqlite3
import faiss
from storm_commander import Colors

BRAIN_DIR = os.path.join(os.getcwd(), "storm_persistent_brain")
DB_PATH = os.path.join(BRAIN_DIR, "metadata.db")
INDEX_PATH = os.path.join(BRAIN_DIR, "brain.faiss")

def inspect_brain():
    print(f"{Colors.HEADER}>>> STORM MEMORY INSPECTOR{Colors.ENDC}")
    
    # 1. Check FAISS Vector Count
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        vector_count = index.ntotal
        print(f"{Colors.CYAN}[FAISS]{Colors.ENDC} Total Vectors (Memories): {vector_count}")
    else:
        print(f"{Colors.FAIL}[FAISS]{Colors.ENDC} Index not found!")
        return

    # 2. Check SQLite Metadata
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Count unique files
        cursor.execute("SELECT COUNT(DISTINCT filename) FROM metadata")
        file_count = cursor.fetchone()[0]
        
        # Count total chunks
        cursor.execute("SELECT COUNT(*) FROM metadata")
        chunk_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"{Colors.CYAN}[SQLITE]{Colors.ENDC} Total Chunks: {chunk_count}")
        print(f"{Colors.CYAN}[SQLITE]{Colors.ENDC} Total Unique Files: {file_count}")
        
        # Analysis
        if file_count > 0:
            avg_chunks = chunk_count / file_count
            print(f"\n{Colors.BOLD}ANALYSIS:{Colors.ENDC}")
            print(f"Average chunks per file: {avg_chunks:.2f}")
            print(f"Reasonable range for academic PDFs: 50 - 150 chunks")
            
            if 50 <= avg_chunks <= 150:
                print(f"{Colors.GREEN}✔ STATUS: NORMAL. The vector count reflects paragraph-level granularity.{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}⚠ STATUS: ABNORMAL. Chunk density is unexpected.{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}[SQLITE]{Colors.ENDC} Database not found!")

if __name__ == "__main__":
    inspect_brain()
