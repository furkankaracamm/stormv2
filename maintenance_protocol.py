
import os
import sys
import shutil
import sqlite3
import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from storm_commander import PersistentBrain, Colors
from storm_modules.ontology import ResearchOntology

# Configuration
LIBRARY_PATH = r"C:\Users\Enes\.gemini\antigravity\scratch\storm\storm_data\library"
DATA_DIR = r"C:\Users\Enes\.gemini\antigravity\scratch\storm\storm_data"
THEORY_TOPIC = "Dead Internet Theory artificial intelligence digital simulation botnet social media algorithm"

def setup_logger():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except: pass

def get_all_pdfs(root_dir):
    pdf_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def clean_library(model):
    print(f"{Colors.HEADER}=== PHASE 1: LIBRARY AUDIT ==={Colors.ENDC}")
    pdfs = get_all_pdfs(LIBRARY_PATH)
    kept = 0
    deleted = 0
    
    topic_embedding = model.encode(THEORY_TOPIC)
    
    for pdf_path in pdfs:
        filename = os.path.basename(pdf_path)
        # Quick check for known physics/math terms in filename/path
        blocklist = ['quantum', 'galaxy', 'stellar', 'neutron', 'bubble', 'fluid', 'coronal', 
                     'topology', 'manifold', 'theorem', 'equation', 'algebra', 'finite', 'infinite', 
                     'kernel', 'spectrum', 'homotopy', 'Lie group', 'cobordism']
        
        if any(x in filename.lower() for x in blocklist):
             # Double check "Filter Bubble" or "Social"
             if "filter bubble" in filename.lower() or "social" in filename.lower():
                 pass # Keep
             else:
                 print(f"{Colors.FAIL}[DELETE - KW] {filename} (Blocklist){Colors.ENDC}")
                 try: os.remove(pdf_path)
                 except: pass
                 deleted += 1
                 continue

        # Semantic Check
        # Embed filename + folder name
        folder = os.path.basename(os.path.dirname(pdf_path))
        text = f"{filename} {folder}"
        file_embedding = model.encode(text)
        
        sim = util.cos_sim(topic_embedding, file_embedding).item()
        
        if sim < 0.25: # Increased threshold for stricter filtering
             print(f"{Colors.WARNING}[DELETE - SEMANTIC] ({sim:.2f}) {filename}{Colors.ENDC}")
             try: os.remove(pdf_path)
             except: pass
             deleted += 1
        else:
             print(f"{Colors.GREEN}[KEEP] ({sim:.2f}) {filename}{Colors.ENDC}")
             kept += 1
             
    print(f"Audit Complete. Kept: {kept}, Deleted: {deleted}")
    return kept

def rebuild_brain(model):
    print(f"\n{Colors.HEADER}=== PHASE 2: BRAIN LOBOTOMY & REBUILD ==={Colors.ENDC}")
    
    # 1. Wipe Brain
    brain_path = os.path.join(DATA_DIR, "brain.faiss")
    db_path = os.path.join(DATA_DIR, "metadata.db")
    processed_path = os.path.join(DATA_DIR, "processed_files") # If it exists separately? No, inside DB.
    
    if os.path.exists(brain_path):
        os.remove(brain_path)
        print("Deleted old FAISS index.")
        
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Deleted old Metadata DB.")
        
    # 2. Re-Initialize Brain (Creates fresh DB/Index)
    brain = PersistentBrain(DATA_DIR)
    
    # 3. Re-Ingest Valid Files
    pdfs = get_all_pdfs(LIBRARY_PATH)
    print(f"Re-learning {len(pdfs)} documents...")
    
    documents = []
    embeddings = []
    metadatas = []
    
    for pdf_path in pdfs:
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:2]: # First 2 pages for indexing
                    text += (page.extract_text() or "") + " "
            
            if len(text) < 100: continue
            
            emb = model.encode(text[:1000]) # Embed first 1000 chars
            
            documents.append(text[:500]) # Store snippet
            embeddings.append(emb)
            metadatas.append({'source': os.path.basename(pdf_path)})
            
            print(f"  > Indexed: {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"  ! Error reading {os.path.basename(pdf_path)}: {e}")
            
    if documents:
        brain.add_batch(documents, embeddings, metadatas)
        # Mark as processed
        for meta in metadatas:
            brain.mark_processed(meta['source'])
            
    print(f"{Colors.GREEN}Brain Rebuilt Successfully!{Colors.ENDC}")

if __name__ == "__main__":
    setup_logger()
    print("Loading AI Models...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    clean_library(model)
    rebuild_brain(model)
