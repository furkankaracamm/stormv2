import os
import sys
import time
import json
import requests
import re
import faiss
import numpy as np
import sqlite3
import random
import networkx as nx
import asyncio
from sentence_transformers import SentenceTransformer
from telethon import TelegramClient
import pdfplumber
import pickle
import unpywall
from bs4 import BeautifulSoup
from unpywall.utils import UnpywallCredentials
from storm_modules.methods_extractor import GROBIDMethodsExtractor
from storm_modules.openalex_client import OpenAlexClient
from storm_modules.pars_cit import ParsCitExtractor
from storm_modules.table_extractor import TableExtractor
from storm_modules.figure_extractor import DeepFiguresExtractor
from storm_modules.extraction_store import save_citations, save_table_results, save_figure_results
from storm_modules.ontology import ResearchOntology
from storm_modules.gap_finder import GapFinder
from storm_modules.llm_gateway import LLMGateway
from storm_modules.gap_finder import GapFinder
from storm_modules.llm_gateway import LLMGateway
from storm_modules.claim_extractor import extract_claims, save_claims_to_db
from storm_modules.research_planner import ResearchPlanner # [NEW] Strategic Mediator
import storm_modules.schema  # [CRITICAL] Schema Enforcement
from storm_modules.scope_guard import ScopeGuard  # [CRITICAL] Topic Enforcement
from storm_modules.semantic_gate import SemanticQualityGate  # [QUALITY] Pre-download Filter

UnpywallCredentials('researcher@storm.io') 

# --- IMPORT LOGGER ---
try:
    from professor_logger import logger
except ImportError:
    class MockLogger:
        def log(self, *args, **kwargs): pass
    logger = MockLogger()

# --- SINGLE INSTANCE LOCK ---
LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".storm_lock")

def acquire_lock():
    """Prevent multiple STORM instances from running."""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Check if process is still running
            import psutil
            if psutil.pid_exists(pid):
                print(f"\n{'='*60}")
                print(f"⛔ HATA: STORM zaten çalışıyor! (PID: {pid})")
                print(f"{'='*60}")
                print("Birden fazla STORM başlatmak veritabanını bozar!")
                print("Önce mevcut pencereyi kapatın veya şu komutu çalıştırın:")
                print(f"  taskkill /F /PID {pid}")
                print(f"{'='*60}\n")
                sys.exit(1)
        except:
            pass  # Stale lock file, continue
    
    # Create lock file with current PID
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))
    
    # Register cleanup
    import atexit
    atexit.register(release_lock)

def release_lock():
    """Remove lock file on exit."""
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except:
        pass

# UTF-8 Output Fix
sys.stdout.reconfigure(encoding='utf-8')

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ITALIC = '\033[3m'
    MAGENTA = '\033[95m'

def log_action(message):
    """Writes a message to the professor_live.log file and the Dashboard."""
    timestamp = time.strftime("%H:%M:%S")
    clean_msg = re.sub(r'\033\[[0-9;]*m', '', message) 
    
    # Log to dashboard
    logger.log("DOWNLOADER", "INFO", clean_msg)
    
    try:
        with open("professor_live.log", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {clean_msg}\n")
    except: pass
    print(message, flush=True)

def init_log():
    """Wipe the log file to ensure clean UTF-8 start."""
    try:
        with open("professor_live.log", "w", encoding="utf-8") as f:
            f.write(f"--- Professor Live Activity Stream [START {time.ctime()}] ---\n")
    except: pass

class PersistentBrain:
    """Production-grade persistent vector store using FAISS + SQLite."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "brain.faiss")
        self.db_path = os.path.join(data_dir, "metadata.db")
        self.dim = 384  # Dimension for all-MiniLM-L6-v2
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # 1. Initialize SQLite Metadata Store with WAL mode for resilience
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Enable Write-Ahead Logging
        self.conn.execute("PRAGMA busy_timeout=30000")  # 30 second wait on locks
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS metadata 
                             (id INTEGER PRIMARY KEY, filename TEXT, content TEXT)''')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS processed_files (filename TEXT PRIMARY KEY)')
        
        # [SELF-HEALING] Sync processed_files with metadata on startup
        self.cursor.execute("INSERT OR IGNORE INTO processed_files (filename) SELECT filename FROM metadata")
        self.conn.commit()

        # 2. Initialize Persistent FAISS Index
        if os.path.exists(self.index_path):
            print(f"{Colors.GREEN}[BRAIN] Loading persistent FAISS index...{Colors.ENDC}", flush=True)
            self.index = faiss.read_index(self.index_path)
            if os.path.exists(os.path.join(data_dir, "knowledge_graph.pkl")):
                with open(os.path.join(data_dir, "knowledge_graph.pkl"), 'rb') as f:
                    self.graph = pickle.load(f)
            else:
                self.graph = nx.DiGraph()
        else:
            print(f"{Colors.WARNING}[BRAIN] Initializing new FAISS index on disk...{Colors.ENDC}", flush=True)
            self.index = faiss.IndexFlatL2(self.dim)
            faiss.write_index(self.index, self.index_path)
            self.graph = nx.DiGraph()

    def is_processed(self, filename):
        self.cursor.execute('SELECT 1 FROM processed_files WHERE filename = ?', (filename,))
        return self.cursor.fetchone() is not None

    def is_title_processed(self, title_fragment):
        clean = "".join(x for x in title_fragment if x.isalnum() or x in " ")
        parts = clean.split()
        if len(parts) < 2: return False
        
        keyword = parts[0] + "%" + parts[-1]
        self.cursor.execute("SELECT 1 FROM processed_files WHERE filename LIKE ?", (f"%{keyword}%",))
        return self.cursor.fetchone() is not None

    def add_batch(self, documents, embeddings, metadatas):
        if not documents: return
        
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        faiss.write_index(self.index, self.index_path)
        
        for doc, meta in zip(documents, metadatas):
            self.cursor.execute('INSERT INTO metadata (filename, content) VALUES (?, ?)', 
                              (meta['source'], doc))
            self.graph.add_node(meta['source'], type='document')
            # [FIX] Mark processed atomically within the same transaction to prevent ghost vectors
            self.cursor.execute('INSERT OR IGNORE INTO processed_files (filename) VALUES (?)', (meta['source'],))
            
        self.conn.commit()
        with open(os.path.join(self.data_dir, "knowledge_graph.pkl"), 'wb') as f:
            pickle.dump(self.graph, f)
            
    # Deprecated standalone method (kept for compatibility)
    def mark_processed(self, filename):
        self.cursor.execute('INSERT OR IGNORE INTO processed_files (filename) VALUES (?)', (filename,))
        self.conn.commit()

    def mark_processed(self, filename):
        self.cursor.execute('INSERT OR IGNORE INTO processed_files (filename) VALUES (?)', (filename,))
        self.conn.commit()

    def get_stats(self):
        self.cursor.execute("SELECT COUNT(*) FROM processed_files WHERE filename LIKE 'BOOK_%'")
        books = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT COUNT(*) FROM processed_files WHERE filename LIKE 'PAPER_%'")
        papers = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT COUNT(*) FROM processed_files")
        total = self.cursor.fetchone()[0]
        return {"processed": total, "books": books, "papers": papers}

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        results = []
        for idx in indices[0]:
            if idx != -1:
                self.cursor.execute('SELECT filename, content FROM metadata WHERE id = ?', (int(idx) + 1,))
                row = self.cursor.fetchone()
                if row:
                    results.append({"filename": row[0], "content": row[1]})
        return results

class StormCommander:
    def __init__(self):
        # Prevent multiple instances
        acquire_lock()
        
        print(f"{Colors.BOLD}{Colors.HEADER}>>> STORM COMMANDER: PERSISTENT BRAIN & MIGRATION ENGINE{Colors.ENDC}", flush=True)
        
        # [CRITICAL] Enforce Database Integrity on Startup
        storm_modules.schema.apply_schema()
        self.base_dir = os.getcwd()
        self.data_dir = os.path.join(self.base_dir, "storm_persistent_brain")
        self.source_pdfs = os.path.join(self.base_dir, "storm_data", "pdfs")
        self.library_dir = os.path.join(self.base_dir, "storm_data", "library") # [NEW] User's Classified Library
        self.pdfs_dir = self.source_pdfs
        
        # Telegram Config
        self.API_ID = int(os.environ.get("STORM_TELEGRAM_API_ID", 0))
        self.API_HASH = os.environ.get("STORM_TELEGRAM_API_HASH", "")
        self.BOT_USERNAME = 'scihubot'
        
        # [FEATURE] Initialize Telegram Client if credentials exist
        if self.API_ID and self.API_HASH:
             self.client = TelegramClient('storm_session', self.API_ID, self.API_HASH)
             print(f"{Colors.GREEN}[TELEGRAM] Client configured (ID: {self.API_ID}){Colors.ENDC}")
        else:
             self.client = None
             print(f"{Colors.WARNING}[TELEGRAM] Credentials missing. Messaging disabled.{Colors.ENDC}")
        
        # User Agents for Stealth
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        # New Modules
        self.openalex_client = OpenAlexClient()
        self.parscit = ParsCitExtractor()
        self.table_extractor = TableExtractor()
        self.figure_extractor = DeepFiguresExtractor()
        self.ontology = ResearchOntology()
        # [FIX] Point to the correct root database (academic_brain.db) instead of split metadata.db
        self.gap_finder = GapFinder(os.path.join(self.base_dir, "academic_brain.db"), self.data_dir)
        self.llm = LLMGateway()
        self.research_planner = ResearchPlanner() # [NEW] Strategic Intelligence Module
        
        print(f"{Colors.CYAN}[MODULES] Gap Finder: READY | LLM: {'GROQ' if self.llm.provider == 'groq' else 'OLLAMA'}{Colors.ENDC}")
        
        # Load theories from DB
        self.theory_context = {}
        try:
            db_path = os.path.join(self.base_dir, "academic_brain.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name, core_propositions FROM theories WHERE name='Dead Internet Theory'")
                row = cursor.fetchone()
                if row:
                    self.theory_context['name'] = row[0]
                    self.theory_context['propositions'] = json.loads(row[1])
                    print(f"{Colors.GREEN}[THEORY LOADED] {row[0]} context active.{Colors.ENDC}")
                conn.close()
        except Exception as e:
            print(f"{Colors.WARNING}[THEORY DB ERROR] {e}{Colors.ENDC}")
        
        # Researcher Persona Settings
        self.academic_qualifiers = [
            "stochastic modeling", "epistemological collapse", "network topology",
            "algorithmic entropy", "critical communication theory", "statistical significance",
            "semantic decay metrics", "post-human discourse topology", "empirical phenomenology"
        ]
        
        # PRIORITY READING LIST
        self.priority_books = [
            "Simulacra and Simulation by Jean Baudrillard", "The Age of Surveillance Capitalism by Shoshana Zuboff",
            "Algorithms of Oppression by Safiya Umoja Noble", "The Filter Bubble by Eli Pariser",
            "Weapons of Math Destruction by Cathy O'Neil", "The Platform Society by Van Dijck",
            "The Black Box Society by Frank Pasquale", "Network Propaganda by Yochai Benkler",
            "The Attention Merchants by Tim Wu", "Alone Together by Sherry Turkle",
            "Life 3.0 by Max Tegmark", "You Are Not a Gadget by Jaron Lanier",
            "Automating Inequality by Virginia Eubanks", "Ghost Work by Gray & Suri",
            "The Shallows by Nicholas Carr", "Surveillance Valley by Yasha Levine",
            "Post-Truth by Lee McIntyre", "Amusing Ourselves to Death by Neil Postman",
            "The Internet Is Not What You Think It Is by Justin Smith", "Fake Accounts by Lauren Oyler"
        ]
        
        # 1. Initialize Brain
        self.brain = PersistentBrain(self.data_dir)
        self.missing_log_path = os.path.join(self.base_dir, "MISSING_PAPERS.txt")
        
        # Filter priority list
        original_count = len(self.priority_books)
        self.priority_books = [b for b in self.priority_books if not self.brain.is_title_processed(b)]
        skipped = original_count - len(self.priority_books)
        if skipped > 0:
            print(f"{Colors.HEADER}[RESUME] Skipped {skipped} already processed books from priority list.{Colors.ENDC}")
        
        # Load initial stats
        brain_stats = self.brain.get_stats()
        self.session_download_limit = 60
        self.session_download_count = 0
        self.stats = {
            "downloaded": 0, 
            "ingested_total": brain_stats["processed"],
            "ingested_books": brain_stats["books"],
            "ingested_papers": brain_stats["papers"],
            "concepts": 0, 
            "validated": 0, 
            "rejected": 0
        }
        self.failed_targets = []
        
        
        # 2. Load Neural Model
        print(f"{Colors.CYAN}Loading Transformer Model (all-MiniLM-L6-v2)...{Colors.ENDC}", flush=True)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # [SCOPE ENFORCEMENT] Initialize Guard with trusted static lists
        print(f"{Colors.HEADER}[SCOPE GUARD] Initializing protocol...{Colors.ENDC}")
        # [FIX] Include the main theory name in the trusted terms
        theory_name = self.theory_context.get('name', 'Dead Internet Theory')
        trusted_terms = [theory_name] + self.priority_books + self.academic_qualifiers
        self.scope_guard = ScopeGuard(additional_terms=trusted_terms)
        
        # [QUALITY GATE] Initialize Semantic Scholar Filter
        self.semantic_gate = SemanticQualityGate()
        
        # [PLURALISM] Control Flow Guardrails
        self.pluralism_threshold = 0.2  # 20% chance to bypass LLM for diversity
        
        os.makedirs(self.pdfs_dir, exist_ok=True)

    def ingest_batch(self):
        """Rapid processing pipeline for all newly downloaded documents."""
        if not os.path.exists(self.source_pdfs):
            print(f"{Colors.FAIL}Source PDF directory not found.{Colors.ENDC}")
            return

        # [RECURSIVE SCAN] Look in both standard PDFs dir and user Library
        all_pdf_paths = []
        search_dirs = [self.source_pdfs, self.library_dir]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, _, files in os.walk(search_dir):
                    for file in files:
                        if file.lower().endswith('.pdf') or file.lower().endswith('.vectorized'):
                            all_pdf_paths.append(os.path.join(root, file))

        # Filter unprocessed (Check DB by filename)
        unprocessed_tuples = [] # (filename, full_path)
        for p in all_pdf_paths:
            fname = os.path.basename(p)
            if not self.brain.is_processed(fname):
                unprocessed_tuples.append((fname, p))

        if not unprocessed_tuples:
            print(f"{Colors.GREEN}[MIGRATION] All {len(all_pdf_paths)} files are already persistent in Brain.{Colors.ENDC}", flush=True)
            return

        print(f"{Colors.BOLD}{Colors.WARNING}[MIGRATION] Found {len(unprocessed_tuples)} files for re-embedding.{Colors.ENDC}", flush=True)
        
        batch_size = 20
        
        for i in range(0, len(unprocessed_tuples), batch_size):
            current_batch = unprocessed_tuples[i:i+batch_size]
            log_action(f"{Colors.BOLD}{Colors.GREEN}[BRAIN INGESTION]{Colors.ENDC} Processing Batch {i//batch_size + 1} ({len(current_batch)} academic files)...")
            
            all_chunks = []
            all_embeddings = []
            all_meta = []
            
            for f_name, f_path in current_batch:
                # [SCOPE GUARD PHASE 1] Filename Check
                # Use abspath for reliable folder detection
                is_auto_download = os.path.abspath(self.source_pdfs) in os.path.abspath(f_path)
                
                if is_auto_download:
                    is_safe_file = self.scope_guard.is_safe(f_name)
                    if f_name.startswith("PAPER_") and not is_safe_file:
                        try:
                            log_action(f"    {Colors.FAIL}[SCOPE PURGE]{Colors.ENDC} Deleting irrelevant file (Filename): {f_name}")
                            os.remove(f_path)
                            continue
                        except: pass

                # [FIX] Do not mark processed here. It creates race conditions.
                # self.brain.mark_processed(f) 
                try:
                    path = f_path # Path is now absolute from the tuple
                    if not os.path.exists(path): continue
                    
                    if os.path.getsize(path) > 50 * 1024 * 1024:
                        log_action(f"    {Colors.WARNING}[SKIP]{Colors.ENDC} {f_name} too large (>50MB)")
                        continue

                    if os.path.getsize(path) < 10 * 1024:
                        log_action(f"    {Colors.FAIL}[DELETE]{Colors.ENDC} {f_name} too small (<10KB), likely broken.")
                        try: os.remove(path)
                        except: pass
                        continue

                    log_action(f"  > Reading: {f_name[:50]}...")
                    text = ""
                    try:
                        with pdfplumber.open(path) as pdf:
                            for i, page in enumerate(pdf.pages[:200]): # Limit to 200 pages
                                extracted = page.extract_text(x_tolerance=2, y_tolerance=2)
                                if extracted:
                                    extracted = extracted.replace('-\n', '')
                                    text += extracted + " "
                                    
                                    if len(extracted) > 100:
                                        import random
                                        sentences = [s.strip() for s in extracted.split('. ') if len(s) > 50]
                                        if sentences:
                                            focus_sentence = random.choice(sentences)
                                            log_action(f"[FOCUS] \"{focus_sentence[:120]}...\"")
                    except Exception as e:
                        log_action(f"    {Colors.FAIL}[Read Error]{Colors.ENDC} {str(e)[:50]}")
                        continue
                    
                    if not text.strip():
                        log_action(f"    {Colors.WARNING}[Warn]{Colors.ENDC} Empty text in {f_name}")
                        continue
                        
                    # [SCOPE GUARD PHASE 2] Deep Content Check
                    # Even if filename passed, if the TEXT is about physics/meteorites, kill it.
                    # We check the first 500 characters + random parts for keywords.
                    if is_auto_download and "PAPER_" in f_name:
                         # Construct a sample for checking
                         sample_text = text[:1000] + text[-1000:]
                         if not self.scope_guard.is_safe(sample_text):
                             log_action(f"    {Colors.FAIL}[SCOPE PURGE]{Colors.ENDC} Content is off-topic (Physics/Bio etc). Deleting: {f_name}")
                             try: os.remove(path)
                             except: pass
                             continue

                    # --- STORM MODULES EXTRACTION ---
                    if os.environ.get("STORM_ENABLE_METHODS_EXTRACT") == "1" or os.environ.get("STORM_ENABLE_GROBID_FULL") == "1":
                        try:
                            extractor = GROBIDMethodsExtractor()
                            methods = extractor.extract_methods_from_pdf(path)
                            if methods:
                                extractor.save_to_database(f_name, methods)
                                log_action(f"    {Colors.CYAN}[METHODS]{Colors.ENDC} N={methods.get('sample_size')} | Design={methods.get('design_type')}")
                        except Exception as e:
                            log_action(f"    {Colors.WARNING}[METHODS FAIL]{Colors.ENDC} {e}")

                    if os.environ.get("STORM_ENABLE_PARS_CIT") == "1":
                        try:
                            citations_parsed = self.parscit.parse_citations_from_text(text)
                            if citations_parsed:
                                save_citations(f_name, citations_parsed)
                                log_action(f"    {Colors.MAGENTA}[PARS CIT]{Colors.ENDC} {len(citations_parsed)} citations extracted.")
                        except Exception as e:
                            log_action(f"    {Colors.WARNING}[PARS FAIL]{Colors.ENDC} {e}")

                    if os.environ.get("STORM_ENABLE_TABLE_EXTRACT") == "1":
                        if self.table_extractor.is_available():
                            try:
                                pages = self.table_extractor.detect_table_pages(path)
                                pages_str = ",".join(str(p) for p in pages)
                                tables = self.table_extractor.extract_tables_from_pdf(path, f_name, pages=pages_str)
                                if tables:
                                    save_table_results(f_name, tables)
                                    log_action(f"    {Colors.CYAN}[TABLES]{Colors.ENDC} {len(tables)} tables extracted.")
                            except Exception as e:
                                log_action(f"    {Colors.WARNING}[TABLES]{Colors.ENDC} {e}")
                        else:
                            log_action(f"    {Colors.WARNING}[TABLES]{Colors.ENDC} tabula-py not available")

                    if os.environ.get("STORM_ENABLE_FIGURE_EXTRACT") == "1":
                        try:
                            figures = self.figure_extractor.extract_figures_from_pdf(path)
                            if figures:
                                save_figure_results(f_name, figures)
                                log_action(f"    {Colors.CYAN}[FIGURES]{Colors.ENDC} {len(figures)} figures extracted.")
                        except Exception as e:
                            log_action(f"    {Colors.WARNING}[FIGURES]{Colors.ENDC} {e}")

                    # --- DEEP ANALYSIS (CLAIMS) ---
                    if os.environ.get("STORM_ENABLE_DEEP_ANALYSIS") == "1":
                        try:
                            claims = extract_claims(text)
                            if claims:
                                save_claims_to_db(f_name, claims, os.path.join(self.data_dir, "metadata.db"))
                                log_action(f"    {Colors.CYAN}[DEEP ANALYSIS]{Colors.ENDC} Extracted {len(claims)} key claims.")
                        except Exception as e:
                            log_action(f"    {Colors.WARNING}[DEEP ANALYSIS FAIL]{Colors.ENDC} {e}")

                    # --- GRAPH MIND ---
                    citations = re.findall(r'\[(\d+)\]', text) 
                    years = re.findall(r'\((\w+ et al\., \d{4})\)', text) 
                    
                    if citations or years:
                        log_action(f"      {Colors.MAGENTA}[GRAPH MIND]{Colors.ENDC} Extracted {len(citations) + len(years)} citation connections.")
                        if hasattr(self.brain, 'graph'):
                            self.brain.graph.add_node(f_name)
                            for c in citations[:10]:
                                self.brain.graph.add_edge(f_name, f"Ref_{c}")
                            for y in years[:10]:
                                self.brain.graph.add_edge(f_name, f"Ref_{y}")

                    # Section-Aware Chunking
                    log_action(f"    * Analyzing structural sections (Methodology/Findings)...")
                    sections_found = {}
                    
                    parts = re.split(r'\n\s*(?:(?:[I|V|X\d]+\.?\s+)?(?:Methodology|Methods|Results|Findings|Discussion|Conclusion|Abstract))\b', text, flags=re.IGNORECASE)
                    found_headers = re.findall(r'\n\s*((?:[I|V|X\d]+\.?\s+)?(?:Methodology|Methods|Results|Findings|Discussion|Conclusion|Abstract))\b', text, flags=re.IGNORECASE)
                    
                    section_blocks = []
                    section_blocks.append(("Introduction/General", parts[0]))
                    for i, header in enumerate(found_headers):
                        if i+1 < len(parts):
                            section_blocks.append((header.strip(), parts[i+1]))
                    
                    for section_name, block_text in section_blocks:
                        if not block_text.strip(): continue
                        
                        # FAZ H: Overlap Chunking with sentence boundary awareness
                        CHUNK_SIZE = 1200
                        STEP_SIZE = 1000  # 200 char overlap
                        s_chunks = []
                        
                        start = 0
                        while start < len(block_text):
                            end = min(start + CHUNK_SIZE, len(block_text))
                            chunk = block_text[start:end]
                            
                            # Try to break at sentence boundary (last period)
                            if end < len(block_text) and len(chunk) > 600:
                                last_period = chunk.rfind('. ')
                                if last_period > 600:  # Only if period is in second half
                                    chunk = chunk[:last_period + 1]
                            
                            s_chunks.append(chunk.strip())
                            start += STEP_SIZE
                        
                        if not s_chunks:
                            continue
                            
                        s_embeddings = self.model.encode(s_chunks)
                        
                        all_chunks.extend(s_chunks)
                        all_embeddings.extend(s_embeddings)
                        
                        for _ in s_chunks:
                            is_high_value = any(k in section_name.lower() for k in ["method", "result", "finding"])
                            all_meta.append({
                                "source": f_name,
                                "section": section_name,
                                "priority": "high" if is_high_value else "normal",
                                "timestamp": time.time()
                            })
                        
                        if any(k in section_name.lower() for k in ["method", "finding"]):
                            sections_found[section_name] = len(block_text)

                        if sections_found or text.strip(): # [FIX] Increment stats if text was extracted, even if sections weren't mapped
                            self.stats['ingested_total'] += 1
                            if f_name.startswith("BOOK_"): self.stats['ingested_books'] += 1
                            elif f_name.startswith("PAPER_"): self.stats['ingested_papers'] += 1
                
                except Exception as e:
                    log_action(f"    {Colors.FAIL}[Migration Error]{Colors.ENDC} {f_name}: {str(e)[:100]}")

            # [RESET] Reset session counter after batch is processed
            self.session_download_count = 0
            
            if all_chunks:
                self.brain.add_batch(all_chunks, all_embeddings, all_meta)
                log_action(f"    {Colors.CYAN}[BRAIN SAVED]{Colors.ENDC} Persistent Index expanded to {self.brain.index.ntotal} chunks.")
                log_action(f"    {Colors.BOLD}[METRICS]{Colors.ENDC} Total: {self.stats['ingested_total']} | Books: {self.stats['ingested_books']} | Papers: {self.stats['ingested_papers']}")

    def verify_persistence(self):
        print(f"\n{Colors.BOLD}{Colors.BLUE}*** VERIFYING PERSISTENT DATA INTEGRITY ***{Colors.ENDC}", flush=True)
        test_query = "dead internet theory simulated consensus"
        print(f"  > Testing search for: '{test_query}'", flush=True)
        emb = self.model.encode([test_query])[0]
        results = self.brain.search(emb, k=2)
        
        if results:
            print(f"  {Colors.GREEN}[SUCCESS]{Colors.ENDC} Found {len(results)} relevant persistent chunks.")
            for r in results:
                print(f"    - Source: {r['filename']} | Snippet: {r['content'][:60]}...")
        else:
            print(f"  {Colors.WARNING}[PENDING]{Colors.ENDC} No data found yet (Migration running).")
            
    def validate_pdf(self, file_path):
        try:
            if not os.path.exists(file_path):
                return False, "File missing"
            
            size = os.path.getsize(file_path)
            if size < 20480: # 20KB minimum
                return False, f"File too small ({size} bytes). Likely a landing page or error response."
            
            # 1. Magic Byte Validation
            with open(file_path, 'rb') as f:
                header = f.read(5)
                if header != b'%PDF-':
                    return False, f"Invalid PDF Header: Expected %PDF-, got {header[:5]!r}. This is likely an HTML landing page."

            # 2. Structural Validation
            with pdfplumber.open(file_path) as pdf:
                if len(pdf.pages) > 0:
                    text_sample = pdf.pages[0].extract_text() or ""
                    if len(text_sample) > 50:
                        return True, "Valid PDF"
                    else:
                        return False, "Unreadable text (Possible image-only scan or corruption)"
                else:
                    return False, "Empty PDF (0 pages)"
        except Exception as e:
            return False, f"Structural Validation Error: {e}"

    async def run_scihub_layer(self, seed):
        # [SCOPE CHECK]
        if not self.scope_guard.is_safe(seed):
            log_action(f"  {Colors.FAIL}[SCOPE BLOCKED]{Colors.ENDC} SciHub Query rejected: '{seed}'")
            return

        log_action(f"{Colors.BOLD}{Colors.WARNING}[RESEARCHER ACTION]{Colors.ENDC} Initiating Peer-Review Discovery for: '{seed}'")
        try:
            query = f"{seed} journal academic study peer-reviewed"
            url = f"https://api.crossref.org/works?query={query}&filter=type:journal-article&rows=40&sort=relevance"
            log_action(f"  > Accessing Meta-Catalog: api.crossref.org (Query: {query})")
            r = requests.get(url, timeout=12)
            items = r.json().get('message', {}).get('items', [])
            
            tasks = []
            for item in items[:15]: 
                doi = item.get('DOI')
                journal = item.get('container-title', [None])[0]
                title = item.get('title', ["Unknown"])[0]
                if doi:
                    log_action(f"    - Found promising Journal DOI: {doi} in '{journal}'")
                    tasks.append(self.retrieve_item(doi, f"[{journal}] {title}" if journal else title, "paper"))
            await asyncio.gather(*tasks)
        except Exception as e:
            log_action(f"  {Colors.FAIL}[DISCOVERY ERROR]{Colors.ENDC} SciHub Layer: {e}")

    async def run_arxiv_layer(self, seed):
        # [SCOPE CHECK]
        if not self.scope_guard.is_safe(seed):
            log_action(f"  {Colors.FAIL}[SCOPE BLOCKED]{Colors.ENDC} Arxiv Query rejected: '{seed}'")
            return

        log_action(f"{Colors.BOLD}{Colors.BLUE}[RESEARCHER ACTION]{Colors.ENDC} Scanning arXiv Pre-prints: '{seed}'")
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            def search_arxiv():
                import arxiv
                client = arxiv.Client()
                # [OPTIMIZATION] Filter by category to reduce noise (CS, Stat, Econ, Quant Fin)
                # Exclude Physics/Math to avoid "Dead Internet" -> "Internet of Things" physics papers
                base_query = f"{seed} AND (abs:thesis OR abs:dissertation)"
                cat_filter = " AND (cat:cs.* OR cat:stat.* OR cat:econ.* OR cat:q-fin.*)"
                
                search = arxiv.Search(
                    query=base_query + cat_filter,
                    max_results=20,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                return list(client.results(search))

            log_action(f"  > Connecting to export.arxiv.org API...")
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(search_arxiv)
                try:
                    results = future.result(timeout=45)
                except Exception as e:
                    if 'timeout' in str(e).lower() or isinstance(e, TimeoutError):
                        log_action(f"    {Colors.WARNING}[TIMEOUT]{Colors.ENDC} arXiv API unresponsive after 45s.")
                        return
                    raise
            
            from difflib import SequenceMatcher
            skipped_count = 0
            found_count = 0
            
            for result in results:
                # [STRICT FILTER] Check if title is relevant
                similarity = SequenceMatcher(None, seed.lower(), result.title.lower()).ratio()
                
                # Check for Scope Safety
                is_safe = self.scope_guard.is_safe(result.title)
                
                if similarity < 0.6 and not is_safe:
                    skipped_count += 1
                    continue
                    
                # [QUALITY CHECK]
                if not self.semantic_gate.check_quality(result.title):
                    skipped_count += 1
                    continue

                log_action(f"    {Colors.GREEN}[HIT]{Colors.ENDC} Relevancy {similarity:.2f}: {result.title[:60]}...")
                
                # Download
                if await self.retrieve_item(result.entry_id, result.title, "paper"):
                    found_count += 1
            
            if skipped_count > 0:
                log_action(f"    {Colors.WARNING}[FILTER]{Colors.ENDC} Filtered {skipped_count} irrelevant/off-topic results.")
            if found_count == 0:
                 log_action(f"    {Colors.WARNING}[MISS]{Colors.ENDC} No sufficiently relevant results found in this batch.")

        except Exception as e:
            log_action(f"  {Colors.FAIL}[DISCOVERY ERROR]{Colors.ENDC} arXiv Layer: {e}")

    async def run_semantic_scholar_layer(self, seed):
        log_action(f"{Colors.BOLD}{Colors.CYAN}[RESEARCHER ACTION]{Colors.ENDC} Querying Semantic Scholar (S2): '{seed}'")
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            def search_s2():
                from semanticscholar import SemanticScholar
                s2 = SemanticScholar()
                return s2.search_paper(seed, limit=50, open_access_pdf=True)

            log_action(f"  > Accessing Semantic Scholar API (Impact Analysis)...")
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(search_s2)
                try:
                    results = future.result(timeout=45)
                except Exception as e:
                    if 'timeout' in str(e).lower() or isinstance(e, TimeoutError):
                        log_action(f"    {Colors.WARNING}[TIMEOUT]{Colors.ENDC} Semantic Scholar API unresponsive after 45s.")
                        return
                    raise

            if not results: return

            papers = []
            for p in results:
                if p.openAccessPdf:
                    papers.append({
                        'obj': p,
                        'cites': p.citationCount if p.citationCount else 0,
                        'title': p.title,
                        'url': p.openAccessPdf.get('url')
                    })
            
            papers.sort(key=lambda x: x['cites'], reverse=True)
            
            tiers = {
                'Q1 (Top Impact)': [p for p in papers if p['cites'] >= 100],
                'Q2 (High Impact)': [p for p in papers if 30 <= p['cites'] < 100],
                'Q3 (Standard)': [p for p in papers if 5 <= p['cites'] < 30],
                'Q4 (Emerging)': [p for p in papers if p['cites'] < 5]
            }

            log_action(f"  > Impact Distribution: Q1:{len(tiers['Q1 (Top Impact)'])} | Q2:{len(tiers['Q2 (High Impact)'])} | Q3:{len(tiers['Q3 (Standard)'])} | Q4:{len(tiers['Q4 (Emerging)'])}")

            processed_count = 0
            for tier_name, tier_papers in tiers.items():
                if not tier_papers: continue
                log_action(f"    {Colors.BOLD}{Colors.GREEN}[QUALITY TIER]{Colors.ENDC} Scanning {tier_name} papers...")
                for p_meta in tier_papers:
                    if tier_name.startswith("Q4") and processed_count > 10:
                        log_action(f"      [SKIP] Skipping remaining Q4 papers to focus on Quality.")
                        break

                    url = p_meta['url']
                    title = p_meta['title']
                    if url: 
                        log_action(f"    - Found {tier_name} Paper ({p_meta['cites']} cites): '{title[:40]}...'")
                        await self.retrieve_item(url, f"[{tier_name[:2]}] {title}", "paper")
                        processed_count += 1

        except Exception as e:
            log_action(f"  {Colors.FAIL}[DISCOVERY ERROR]{Colors.ENDC} S2 Layer: {e}")

    async def run_openalex_layer(self, seed):
        """Discovery Layer: Uses OpenAlex for additional metadata + OA links."""
        if not self.openalex_client.is_available():
            log_action(f"{Colors.WARNING}[OPENALEX]{Colors.ENDC} requests not available.")
            return

        log_action(f"{Colors.BOLD}{Colors.BLUE}[RESEARCHER ACTION]{Colors.ENDC} Querying OpenAlex: '{seed}'")
        try:
            works = self.openalex_client.fetch_works(seed, per_page=10)
        except Exception as e:
            log_action(f"{Colors.FAIL}[OPENALEX ERROR]{Colors.ENDC} {e}")
            return

        tasks = []
        for work in works:
            doi = work.get("doi")
            title = work.get("title") or "Unknown"
            open_access = work.get("open_access") or {}
            oa_url = open_access.get("oa_url")
            target = doi or oa_url
            if target:
                tasks.append(self.retrieve_item(target, f"[OA] {title}", "paper"))

        if tasks:
            print(f"    [OPENALEX] Dispatching {len(tasks)} retrieval tasks...")
            await asyncio.gather(*tasks)

    async def run_annas_layer(self, seed):
        log_action(f"{Colors.BOLD}{Colors.MAGENTA}[RESEARCHER ACTION]{Colors.ENDC} Hunting for Academic Books (V2): '{seed}'")
        try:
            await self.fetch_from_annas_v2(seed, os.path.join(self.pdfs_dir, f"BOOK_{seed[:30].replace(' ', '_')}.pdf"))
        except Exception as e:
            log_action(f"  {Colors.FAIL}[DISCOVERY ERROR]{Colors.ENDC} Book Layer: {e}")

    async def retrieve_item(self, url_or_doi, title, item_type):
        # [SCOPE CHECK] Gatekeeper for titles
        if title and not self.scope_guard.is_safe(title):
             log_action(f"  {Colors.FAIL}[SCOPE BLOCKED]{Colors.ENDC} Download rejected for title: '{title}'")
             return

        # [QUALITY CHECK] Advisory Semantic Gate (Post-Scope, Pre-Download)
        if title and not self.semantic_gate.check_quality(title):
             log_action(f"  {Colors.FAIL}[QUALITY BLOCKED]{Colors.ENDC} Irrelevant field of study: '{title}'")
             return
            
        if not url_or_doi: return
        # [FIX] Removed "Source" from character class to prevent mangling vowels
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:60].strip()
        filename = f"{item_type.upper()}_{safe_title}.pdf"
        file_path = os.path.join(self.pdfs_dir, filename)
        
        if os.path.exists(file_path): 
            log_action(f"      [SKIP] '{title[:30]}...' already exists in library.")
            return

        # [THRESHOLD CHECK] Block if session limit reached
        if self.session_download_count >= self.session_download_limit:
            log_action(f"      {Colors.WARNING}[BATCH LIMIT]{Colors.ENDC} Session limit ({self.session_download_limit}) reached. Deferring to Reading Mode.")
            return False

        async def _attempt_download():
            # TIER 0: DIRECT PDF LINK
            if item_type == "paper" and url_or_doi and url_or_doi.startswith("http") and url_or_doi.lower().endswith(".pdf"):
                 log_action(f"      [ACTION] Attempting Direct PDF Download: {url_or_doi[:40]}...")
                 if await self.fetch_direct(url_or_doi, file_path):
                     is_ok, msg = self.validate_pdf(file_path)
                     if is_ok:
                         log_action(f"      {Colors.GREEN}[VERIFIED]{Colors.ENDC} Direct Source Success.")
                         self.stats['downloaded'] += 1; return True

            # TIER 1: LEGAL SPEED LAYER
            if item_type == "paper" and url_or_doi and isinstance(url_or_doi, str) and url_or_doi.startswith("10."):
                if await self.run_legal_speed_layer(url_or_doi, file_path):
                     is_ok, msg = self.validate_pdf(file_path)
                     if is_ok:
                         log_action(f"      {Colors.GREEN}[VERIFIED]{Colors.ENDC} Legal Speed Layer Success.")
                         self.stats['downloaded'] += 1; return True

            # TIER 2: ANNA'S ARCHIVE V2
            if await self.fetch_from_annas_v2(url_or_doi or title, file_path):
                 is_ok, msg = self.validate_pdf(file_path)
                 if is_ok:
                     log_action(f"      {Colors.GREEN}[VERIFIED]{Colors.ENDC} Anna's Archive V2 Success.")
                     self.stats['downloaded'] += 1; return True

            # TIER 3: TELEGRAM
            if item_type == "paper" and url_or_doi and isinstance(url_or_doi, str) and url_or_doi.startswith("10."):
                if await self.fetch_via_telegram(url_or_doi, file_path):
                    is_ok, msg = self.validate_pdf(file_path)
                    if is_ok: 
                        log_action(f"      {Colors.GREEN}[VERIFIED]{Colors.ENDC} Telegram Success: '{title[:30]}...'")
                        self.stats['downloaded'] += 1; return True

            # TIER 4: DIRECT SCI-HUB MIRRORS
            if item_type == "paper" and url_or_doi and isinstance(url_or_doi, str) and ("/" in url_or_doi or "10." in url_or_doi):
                log_action(f"      [ACTION] Exhausting direct Sci-Hub infrastructure for: {url_or_doi[:40]}")
                if await self.fetch_direct(url_or_doi, file_path):
                    is_ok, msg = self.validate_pdf(file_path)
                    if is_ok: 
                        log_action(f"      {Colors.GREEN}[VERIFIED]{Colors.ENDC} Sci-Hub Direct Success.")
                        self.stats['downloaded'] += 1; return True
                    else: 
                        log_action(f"      {Colors.FAIL}[REJECTED]{Colors.ENDC} {msg}. Deleting local copy.")
                        try: os.remove(file_path) 
                        except: pass
            
            # TIER 5: SCIDOWNL FALLBACK (DOI ONLY)
            if item_type == "paper" and url_or_doi and isinstance(url_or_doi, str) and "10." in url_or_doi:
                 log_action(f"      [ACTION] Attempting Scidownl DOI Resolution fallback...")
                 if await self.fetch_direct(url_or_doi, file_path):
                      is_ok, msg = self.validate_pdf(file_path)
                      if is_ok: 
                          log_action(f"      {Colors.GREEN}[VERIFIED]{Colors.ENDC} Scidownl Success.")
                          self.stats['downloaded'] += 1; return True
            
            return False

        try:
            if await _attempt_download():
                self.session_download_count += 1
                return True
        except Exception as e:
            log_action(f"      [ERROR] Retrieval failed: {e}")

        # [FINAL FAILURE]
        self.log_missing_target(title, url_or_doi)
        log_action(f"      {Colors.FAIL}[MISSING]{Colors.ENDC} All layers exhausted. Added to MISSING_PAPERS.txt")
        return False

    def log_missing_target(self, title, doi=None):
        try:
            entry = f"[{time.strftime('%Y-%m-%d %H:%M')}] TITLE: {title} | DOI: {doi or 'N/A'}\n"
            with open(self.missing_log_path, "a", encoding="utf-8") as f:
                f.write(entry)
            
            # [FEATURE] Send direct alert to User via Telegram
            if self.client:
                async def _send_alert():
                    try:
                        await self.client.connect()
                        if await self.client.is_user_authorized():
                            msg = f"⚠️ **MISSING PAPER ALERT**\n\n**Title:** {title}\n**DOI:** {doi}\n\n_Please upload PDF to library._"
                            await self.client.send_message('me', msg)
                    except Exception as e:
                        print(f"{Colors.WARNING}[TELEGRAM ERROR] Could not send alert: {e}{Colors.ENDC}")
                
                # Fire and forget (schedule correctly)
                loop = asyncio.get_running_loop()
                loop.create_task(_send_alert())

        except: pass

    async def run_legal_speed_layer(self, doi, file_path):
        if not doi: return False
        try:
             log_action(f"      [SPEED LAYER] Checking Unpaywall for legal FullText...")
             loop = asyncio.get_running_loop()
             def check_unpaywall():
                 return unpywall.Unpaywall.doi(doi)
             paper = await asyncio.wait_for(loop.run_in_executor(None, check_unpaywall), timeout=10)
             if paper and paper.is_oa and paper.best_oa_location:
                 url = paper.best_oa_location['url']
                 log_action(f"      {Colors.GREEN}[SPEED LAYER]{Colors.ENDC} HIT! Legal PDF found: {url[:50]}...")
                 return await self.fetch_direct(url, file_path)
             return False
        except Exception: 
            return False

    async def fetch_from_annas_v2(self, query, file_path):
        mirrors = ["https://annas-archive.li", "https://annas-archive.se", "https://annas-archive.pm", "https://annas-archive.org"]
        
        if not hasattr(self, '_mirror_blacklist'):
            self._mirror_blacklist = {}
        
        current_time = time.time()
        healthy_mirrors = [m for m in mirrors if current_time - self._mirror_blacklist.get(m, 0) > 300]
        
        if not healthy_mirrors:
            log_action(f"      [ANNAS ARCHIVE] All mirrors blacklisted. Resetting...")
            self._mirror_blacklist = {}
            healthy_mirrors = mirrors
        
        log_action(f"      [ANNAS ARCHIVE] Searching {len(healthy_mirrors)} healthy mirrors for: '{query[:40]}...'")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }

        async def try_mirror(base_url):
            try:
                search_url = f"{base_url}/search?q={requests.utils.quote(query)}"
                loop = asyncio.get_running_loop()
                r = await loop.run_in_executor(None, lambda: requests.get(search_url, headers=headers, timeout=10))
                if r.status_code != 200: 
                    self._mirror_blacklist[base_url] = time.time()
                    return None
                
                soup = BeautifulSoup(r.text, 'html.parser')
                links = soup.find_all('a', href=True)
                for l in links:
                    if "/md5/" in l['href']:
                        book_page = base_url + l['href']
                        r2 = await loop.run_in_executor(None, lambda: requests.get(book_page, headers=headers, timeout=10))
                        soup2 = BeautifulSoup(r2.text, 'html.parser')
                        down_links = soup2.find_all('a', href=True)
                        for dl in down_links:
                            href = dl['href']
                            if ("ipfs" in href or "scihub" in href or "annas-archive" in href) and "libgen" not in href:
                                return href
                return None
            except Exception as e:
                self._mirror_blacklist[base_url] = time.time()
                return None

        for mirror in healthy_mirrors:
            download_link = await try_mirror(mirror)
            if download_link:
                log_action(f"      [ANNAS HIT] Using Mirror {mirror}: {download_link[:50]}...")
                return await self.fetch_direct(download_link, file_path)
            
        return False

    async def fetch_from_annas_v2(self, query, file_path):
        filename = os.path.basename(file_path)
        if os.path.exists(file_path) or self.brain.is_processed(filename):
            log_action(f"      [SKIP] '{filename[:30]}...' already processed.")
            return True

        # [SCOPE CHECK]
        # Only check text queries. Trust DOIs/MD5s as they likely came from validated upstream steps.
        is_identifier = "10." in query or len(query) == 32
        if not is_identifier and not self.scope_guard.is_safe(str(query)):
             log_action(f"      {Colors.FAIL}[SCOPE BLOCKED]{Colors.ENDC} Anna's Query rejected: '{query}'")
             return False

        mirrors = ["https://annas-archive.li", "https://annas-archive.se", "https://annas-archive.pm", "https://annas-archive.org"]
        
        if not hasattr(self, '_mirror_blacklist'):
            self._mirror_blacklist = {}
        
        current_time = time.time()
        healthy_mirrors = [m for m in mirrors if current_time - self._mirror_blacklist.get(m, 0) > 300]
        
        if not healthy_mirrors:
            log_action(f"      [ANNAS ARCHIVE] All mirrors blacklisted. Resetting...")
            self._mirror_blacklist = {}
            healthy_mirrors = mirrors
        
        log_action(f"      [ANNAS ARCHIVE] Searching {len(healthy_mirrors)} healthy mirrors for: '{query[:40]}...'")
        
        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }

        async def try_mirror(base_url):
            try:
                # [STEALTH] Add Delay to avoid Blacklist
                await asyncio.sleep(random.uniform(2, 5))
                search_url = f"{base_url}/search?q={requests.utils.quote(query)}"
                loop = asyncio.get_running_loop()
                r = await loop.run_in_executor(None, lambda: requests.get(search_url, headers=headers, timeout=10))
                if r.status_code != 200: 
                    self._mirror_blacklist[base_url] = time.time()
                    return None
                
                soup = BeautifulSoup(r.text, 'html.parser')
                links = soup.find_all('a', href=True)
                for l in links:
                    if "/md5/" in l['href']:
                        book_page = base_url + l['href']
                        r2 = await loop.run_in_executor(None, lambda: requests.get(book_page, headers=headers, timeout=10))
                        soup2 = BeautifulSoup(r2.text, 'html.parser')
                        down_links = soup2.find_all('a', href=True)
                        for dl in down_links:
                            href = dl['href']
                            # [PROTOCOL UPDATE] Allow LibGen loops as they are often cleaner than Anna's direct
                            if ("ipfs" in href or "scihub" in href or "annas-archive" in href or "library.lol" in href or "libgen" in href):
                                return href
                return None
            except Exception as e:
                self._mirror_blacklist[base_url] = time.time()
                return None

        for mirror in healthy_mirrors:
            download_link = await try_mirror(mirror)
            if download_link:
                log_action(f"      [ANNAS HIT] Using Mirror {mirror}: {download_link[:50]}...")
                if await self.fetch_direct(download_link, file_path):
                    return True
                else:
                    log_action(f"      [ANNAS WARN] Download failed from {mirror}. Trying next...")
                    continue
            
        return False

    async def fetch_from_scihub_mirrors(self, doi, file_path):
        mirrors = [
            "https://sci-hub.se",
            "https://sci-hub.st",
            "https://sci-hub.ru",
            "https://sci-hub.hkvisa.net"
        ]
        
        loop = asyncio.get_running_loop()
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

        for mirror in mirrors:
            try:
                target_url = f"{mirror}/{doi}"
                log_action(f"        * Trying Sci-Hub Mirror: {mirror}...")
                
                # 1. Get the page to find the PDF link
                r = await loop.run_in_executor(None, lambda: requests.get(target_url, headers=headers, timeout=15))
                if r.status_code != 200: continue

                # [FIX] Handle potential non-UTF8 mirror responses
                try: 
                    response_text = r.text
                except:
                    response_text = r.content.decode('utf-8', errors='ignore')

                # 2. Parse PDF link (simple heuristics)
                if "location.href='" in response_text:
                    pdf_url = response_text.split("location.href='")[1].split("'")[0]
                    if pdf_url.startswith("//"): pdf_url = "https:" + pdf_url
                    
                    # 3. Download the actual PDF
                    log_action(f"          > Found PDF link: {pdf_url[:40]}...")
                    return await self.fetch_direct(pdf_url, file_path)
            except Exception as e:
                continue
        return False

    async def fetch_direct(self, url_or_doi, file_path):
        try:
            loop = asyncio.get_running_loop()
            if url_or_doi.startswith("http"):
                # [SMART FIX] Convert arXiv 'abs' links to 'pdf'
                if "arxiv.org/abs/" in url_or_doi:
                    url_or_doi = url_or_doi.replace("arxiv.org/abs/", "arxiv.org/pdf/")
                    if not url_or_doi.endswith(".pdf"): url_or_doi += ".pdf"
                
                log_action(f"        * Establishing Secure Stream to: {url_or_doi[:50]}...")
                headers = {'User-Agent': random.choice(self.user_agents)}
                
                # Use stream=True to check headers before full download
                r = await loop.run_in_executor(None, lambda: requests.get(url_or_doi, headers=headers, timeout=30, stream=True))
                
                if r.status_code == 200:
                    # [ROBUSTNESS] Verify Content-Type
                    content_type = r.headers.get('Content-Type', '').lower()
                    if 'application/pdf' not in content_type and 'octet-stream' not in content_type:
                        log_action(f"        {Colors.FAIL}[REJECTED]{Colors.ENDC} Content-Type is '{content_type}', not PDF.")
                        return False

                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=32768): 
                            if chunk: f.write(chunk)
                    
                    # Immediate structural check
                    is_ok, msg = self.validate_pdf(file_path)
                    if not is_ok:
                        log_action(f"        {Colors.FAIL}[BROKEN]{Colors.ENDC} {msg}. Deleting.")
                        try: os.remove(file_path)
                        except: pass
                        return False

                    return True
                else:
                    log_action(f"        {Colors.FAIL}[DENIED]{Colors.ENDC} Status: {r.status_code}")

            if isinstance(url_or_doi, str) and ("/" in url_or_doi or "10." in url_or_doi):
                # [STRATEGY 1] Try Custom Mirror Cycler First (Faster, more control)
                if await self.fetch_from_scihub_mirrors(url_or_doi, file_path):
                    return True

                # [STRATEGY 2] Fallback to Scidownl library
                from scidownl import scihub_download
                log_action(f"        * Routing DOI via Sci-Hub Global Proxies: {url_or_doi}")
                def run_scidownl():
                    try:
                        scihub_download(url_or_doi, paper_type="doi", out=file_path)
                        return os.path.exists(file_path) and os.path.getsize(file_path) > 10000
                    except: return False
                return await loop.run_in_executor(None, run_scidownl)
            return False
        except Exception as e:
            log_action(f"        [FETCH ERROR] {str(e)[:50]}")
            return False

    async def fetch_via_telegram(self, query, file_path):
        try:
            chat = await self.client.get_entity(self.BOT_USERNAME)
            async with self.client.conversation(chat, timeout=30) as conv:
                await conv.send_message(query)
                response = await conv.get_response()
                if (not response.file) and ("found" in response.text.lower() or "have" in response.text.lower()):
                    try: response = await conv.get_response()
                    except: pass
                if response.file:
                    await self.client.download_media(response.media, file_path)
                    return True
            return False
        except: return False

    def simulate_thought_stream(self):
        thoughts = [
            "Analyzing epistemic consistency...",
            "Checking for theoretical drift...",
            "Synthesizing new contradictions...",
            "Mapping discursive topology...",
            "Evaluating source credibility...",
            "Detecting algorithmic bias...",
            "Calculating semantic entropy...",
            "Refactoring knowledge graph..."
        ]
        
        # Inject Dynamic Theory Thoughts
        if self.theory_context and 'propositions' in self.theory_context:
            prop = random.choice(self.theory_context['propositions'])
            thoughts.append(f"Reflecting on axiom: '{prop[:60]}...'")
        print(f"  {Colors.ITALIC}{Colors.CYAN}[THOUGHT] {random.choice(thoughts)}{Colors.ENDC}", flush=True)

    async def run_autonomous_loop(self):
        seeds = [
            'simulacra and simulation pdf', 'surveillance capitalism shoshana zuboff pdf',
            'algorithms of oppression pdf', 'neural networks critical theory',
            'AI generated semantic hollows', 'epistemology of fake consensus',
            'social media bot detection PhD thesis', 'theory of non-human discourse dissertation'
        ]
        
        while True:
            self.cycle_count = getattr(self, 'cycle_count', 0) + 1
            print(f"\n{Colors.BOLD}{Colors.BLUE}>>> PROFESSOR STATUS (Cycle {self.cycle_count}): {self.stats['ingested_total']} Read | {self.brain.index.ntotal} Memories{Colors.ENDC}", flush=True)

            # GAP FINDER TRIGGER (Every 10 Cycles)
            if self.cycle_count % 10 == 0:
                print(f"{Colors.BOLD}{Colors.MAGENTA}>>> GAP FINDER: Analyzing Research Landscape...{Colors.ENDC}")
                try:
                    gaps = self.gap_finder.get_top_gaps(limit=3)
                    if gaps:
                        # [SEMANTIC FIX] Archive gaps for future research phases
                        self.gap_finder.save_gaps_to_db(gaps)
                        
                        top_gap = gaps[0]
                        suggestion = top_gap.get('suggestion', 'No suggestion')
                        log_action(f"  {Colors.MAGENTA}[GAP ARCHIVED] {suggestion} (Saved to DB){Colors.ENDC}")
                        
                        # Reflexive Action: Use gap to drive IMMEDIATE reading
                        await self.retrieve_item(None, suggestion + " research paper", "paper")
                except Exception as e:
                    log_action(f"  [GAP ERROR] {e}")

            # STRATEGIC LLM REFLECTION (Every 5 Cycles)
            if self.cycle_count % 5 == 0 and self.llm.is_available():
                prompt = f"""Synthesize a profound research question following 'Dead Internet Theory' logic.
                Return JSON only: {{"question": "...", "target_query": "specific search terms", "rationale": "..."}}"""
                raw_insight = self.llm.generate(prompt, max_tokens=200, json_mode=True)
                
                if raw_insight:
                    insight_data = self.research_planner.parse_llm_insight(raw_insight)
                    if insight_data and "target_query" in insight_data:
                        self.research_planner.add_insight("STRATEGIC_RESEARCH", insight_data, insight_data["target_query"])
                        log_action(f"  {Colors.CYAN}[STRATEGIC INSIGHT] {insight_data.get('question')[:80]}...{Colors.ENDC}")
            
            # PHASE 0: STRATEGIC STEERING (LLM DRIVEN)
            strategic_targets = self.research_planner.get_top_strategic_targets(limit=1)
            active_strategic_id = None
            
            # [PLURALISMRefactor] Check for bypass roll
            pluralism_bypass = random.random() < self.pluralism_threshold
            
            if strategic_targets and not pluralism_bypass:
                target_meta = strategic_targets[0]
                active_strategic_id = target_meta["id"]
                target = target_meta["query"]
                log_action(f"  {Colors.BOLD}{Colors.CYAN}[STRATEGIC STEER]{Colors.ENDC} LLM Weight ({target_meta['weight']:.2f}) -> '{target}'")
                
                # Execute strategic discovery
                success = await self.run_scihub_layer(target)
                success = success or await self.run_annas_layer(target)
                
                if success:
                    self.research_planner.report_success(active_strategic_id)
                else:
                    self.research_planner.report_failure(active_strategic_id)

            # PHASE 1: PRIORITY RESEARCH (DIRECTED)
            elif self.priority_books:
                if strategic_targets and pluralism_bypass:
                    log_action(f"  {Colors.YELLOW}[PLURALISM BYPASS]{Colors.ENDC} Diverting from LLM to User Priority List")
                target_book = self.priority_books.pop(0)
                print(f"  {Colors.HEADER}[PRIORITY RESEARCH]{Colors.ENDC} Targeting: {Colors.BOLD}{target_book}{Colors.ENDC}", flush=True)
                await self.run_annas_layer(target_book)
                await self.run_arxiv_layer(f"Critical analysis of {target_book}")
            
            # PHASE 2: DISCOVERY (GAP FINDING / ONTOLOGY DRIVEN)
            else:
                if strategic_targets and pluralism_bypass:
                    log_action(f"  {Colors.YELLOW}[PLURALISM BYPASS]{Colors.ENDC} Diverting from LLM to Ontology Layer")
                # 80% chance to derive topic from Ontology (Structured Research)
                if random.random() < 0.8:
                    sub_topic = random.choice(self.ontology.get_all_topics())
                    context = self.ontology.get_domain_context(sub_topic)
                    
                    # Construct a sophisticated academic query
                    target = f"{sub_topic} {random.choice(self.academic_qualifiers)} \"Dead Internet Theory\""
                    log_action(f"  {Colors.MAGENTA}[ONTOLOGY FOCUS]{Colors.ENDC} {context} -> '{sub_topic}'")
                    
                # 20% chance for Wildcard Exploration
                else:
                    base = random.choice(seeds)
                    qualifier = random.choice(self.academic_qualifiers)
                    target = f"{base} {qualifier}"
                    log_action(f"  {Colors.CYAN}[WILDCARD EXPLORATION]{Colors.ENDC} '{target}'")
                
                # [BATCH CHECK] Skip discovery if session limit reached
                if self.session_download_count >= self.session_download_limit:
                    log_action(f"  {Colors.WARNING}[BATCH LIMIT]{Colors.ENDC} 60 files gathered. Skipping discovery until ingested.")
                else:
                    discovery_tasks = [
                        self.run_scihub_layer(target),
                        self.run_arxiv_layer(target),
                        self.run_semantic_scholar_layer(target),
                        self.run_annas_layer(target),
                    ]
                    if os.environ.get("STORM_ENABLE_OPENALEX") == "1":
                        discovery_tasks.append(self.run_openalex_layer(target))
                    await asyncio.gather(*discovery_tasks)
            
            # PROCESSING
            self.ingest_batch()
            self.research_planner.apply_cycle_decay() # [SAFETY] Decay LLM influence
            self.simulate_thought_stream()
            await asyncio.sleep(2)

if __name__ == "__main__":
    init_log()
    commander = StormCommander()
    commander.ingest_batch()
    commander.verify_persistence()
    asyncio.run(commander.run_autonomous_loop())
