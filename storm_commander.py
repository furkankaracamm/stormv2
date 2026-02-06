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
import asyncio
from sentence_transformers import SentenceTransformer
from telethon import TelegramClient
import pdfplumber
import pickle
import unpywall # Changed from 'from unpywall import Unpaywall'
from bs4 import BeautifulSoup
from unpywall.utils import UnpywallCredentials
UnpywallCredentials('researcher@storm.io') 

# --- IMPORT LOGGER ---
try:
    from professor_logger import logger
except ImportError:
    class MockLogger:
        def log(self, *args, **kwargs): pass
    logger = MockLogger()

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
        # Fuzzy check if a book with this title has been processed
        # Remove special chars for safer matching
        clean = "".join(x for x in title_fragment if x.isalnum() or x in " ")
        parts = clean.split()
        if len(parts) < 2: return False
        
        # Check if any processed file contains significant parts of the title
        # This is a heuristic to prevent re-downloading "Simulacra and Simulation" 
        # if "BOOK_Simulacra_and_Simulation_Jean_Baudrillard.pdf" exists.
        keyword = parts[0] + "%" + parts[-1]
        self.cursor.execute("SELECT 1 FROM processed_files WHERE filename LIKE ?", (f"%{keyword}%",))
        return self.cursor.fetchone() is not None

    def add_batch(self, documents, embeddings, metadatas):
        if not documents: return
        
        # FAISS Persistent Write
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        faiss.write_index(self.index, self.index_path)
        
        # SQLite Metadata Persistent Write
        for doc, meta in zip(documents, metadatas):
            self.cursor.execute('INSERT INTO metadata (filename, content) VALUES (?, ?)', 
                              (meta['source'], doc))
            # Graph Mind: Add Node
            self.graph.add_node(meta['source'], type='document')
            
        self.conn.commit()
        # Persist Graph
        with open(os.path.join(self.data_dir, "knowledge_graph.pkl"), 'wb') as f:
            pickle.dump(self.graph, f)

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
        print(f"{Colors.BOLD}{Colors.HEADER}>>> STORM COMMANDER: PERSISTENT BRAIN & MIGRATION ENGINE{Colors.ENDC}", flush=True)
        self.base_dir = os.getcwd()
        self.data_dir = os.path.join(self.base_dir, "storm_persistent_brain")
        self.source_pdfs = os.path.join(self.base_dir, "storm_data", "pdfs")
        self.pdfs_dir = self.source_pdfs
        
        # Telegram Config
        self.API_ID = 30881934
        self.API_HASH = 'f21730701d0b1da80764c094c73effdb'
        self.BOT_USERNAME = 'scihubot'
        self.client = TelegramClient('storm_session', self.API_ID, self.API_HASH)
        
        # Researcher Persona Settings
        self.academic_qualifiers = [
            "stochastic modeling", "epistemological collapse", "network topology",
            "algorithmic entropy", "critical communication theory", "statistical significance",
            "semantic decay metrics", "post-human discourse topology", "empirical phenomenology"
        ]
        
        # PRIORITY READING LIST (User Curated - Top Priority)
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
        
        # Filter priority list against already processed items
        original_count = len(self.priority_books)
        self.priority_books = [b for b in self.priority_books if not self.brain.is_title_processed(b)]
        skipped = original_count - len(self.priority_books)
        if skipped > 0:
            print(f"{Colors.HEADER}[RESUME] Skipped {skipped} already processed books from priority list.{Colors.ENDC}")
        
        # Load initial stats from persistence
        brain_stats = self.brain.get_stats()
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
        
        os.makedirs(self.pdfs_dir, exist_ok=True)

    def ingest_batch(self):
        """Rapid processing pipeline for all newly downloaded documents."""
        if not os.path.exists(self.source_pdfs):
            print(f"{Colors.FAIL}Source PDF directory not found.{Colors.ENDC}")
            return

        files = [f for f in os.listdir(self.source_pdfs) if f.endswith('.pdf') or f.endswith('.vectorized')]
        unprocessed = [f for f in files if not self.brain.is_processed(f)]
        
        if not unprocessed:
            print(f"{Colors.GREEN}[MIGRATION] All {len(files)} files are already persistent in Brain.{Colors.ENDC}", flush=True)
            return

        print(f"{Colors.BOLD}{Colors.WARNING}[MIGRATION] Found {len(unprocessed)} files for re-embedding.{Colors.ENDC}", flush=True)
        
        batch_size = 20
        # Replaced pypdf with pdfplumber globally
        
        for i in range(0, len(unprocessed), batch_size):
            current_batch = unprocessed[i:i+batch_size]
            log_action(f"{Colors.BOLD}{Colors.GREEN}[BRAIN INGESTION]{Colors.ENDC} Processing Batch {i//batch_size + 1} ({len(current_batch)} academic files)...")
            
            all_chunks = []
            all_embeddings = []
            all_meta = []
            
            for f in current_batch:
                # Mark as processed immediately to prevent retry-loops on crash/hang
                self.brain.mark_processed(f)
                try:
                    path = os.path.join(self.source_pdfs, f)
                    if not os.path.exists(path): continue
                    
                    # Basic Size Filter
                    if os.path.getsize(path) > 50 * 1024 * 1024:
                        log_action(f"    {Colors.WARNING}[SKIP]{Colors.ENDC} {f} too large (>50MB)")
                        continue

                    log_action(f"  > Reading: {f[:50]}...")
                    text = ""
                    try:
                        with pdfplumber.open(path) as pdf:
                            for i, page in enumerate(pdf.pages[:200]): # Limit to 200 pages
                                extracted = page.extract_text(x_tolerance=2, y_tolerance=2)
                                if extracted:
                                    # Fix de-hyphenation (common in papers: "net- work" -> "network")
                                    extracted = extracted.replace('-\n', '')
                                    text += extracted + " "
                                    
                                    # MICRO-FOCUS: Broadcast a sentence to the terminal
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
                        log_action(f"    {Colors.WARNING}[Warn]{Colors.ENDC} Empty text in {f}")
                        continue

                    # GRAPH MIND: Extract Citations for Network Analysis
                    citations = re.findall(r'\[(\d+)\]', text) # Matches [1], [12]
                    years = re.findall(r'\((\w+ et al\., \d{4})\)', text) # Matches (Smith et al., 2020)
                    
                    if citations or years:
                        log_action(f"      {Colors.MAGENTA}[GRAPH MIND]{Colors.ENDC} Extracted {len(citations) + len(years)} citation connections.")
                        # Add to Brain Graph
                        # Note: We need to access the graph directly or via a method
                        if hasattr(self.brain, 'graph'):
                            self.brain.graph.add_node(f)
                            for c in citations[:10]: # Limit to avoiding graph explosion
                                self.brain.graph.add_edge(f, f"Ref_{c}")
                            for y in years[:10]:
                                self.brain.graph.add_edge(f, f"Ref_{y}")

                    # Section-Aware Chunking (Methodology & Findings Focus)
                    log_action(f"    * Analyzing structural sections (Methodology/Findings)...")
                    sections_found = {}
                    current_section = "General"
                    
                    # Split text by common section headers to identify blocks
                    parts = re.split(r'\n\s*(?:(?:[I|V|X\d]+\.?\s+)?(?:Methodology|Methods|Results|Findings|Discussion|Conclusion|Abstract))\b', text, flags=re.IGNORECASE)
                    
                    # Heuristic mapping of headers to the split parts
                    found_headers = re.findall(r'\n\s*((?:[I|V|X\d]+\.?\s+)?(?:Methodology|Methods|Results|Findings|Discussion|Conclusion|Abstract))\b', text, flags=re.IGNORECASE)
                    
                    section_blocks = []
                    section_blocks.append(("Introduction/General", parts[0]))
                    for i, header in enumerate(found_headers):
                        if i+1 < len(parts):
                            section_blocks.append((header.strip(), parts[i+1]))
                    
                    for section_name, block_text in section_blocks:
                        if not block_text.strip(): continue
                        
                        # High-speed chunking per section
                        s_chunks = [block_text[j:j+1200] for j in range(0, len(block_text), 1200)]
                        s_embeddings = self.model.encode(s_chunks)
                        
                        all_chunks.extend(s_chunks)
                        all_embeddings.extend(s_embeddings)
                        
                        for _ in s_chunks:
                            # Metadata enrichment with section affinity
                            is_high_value = any(k in section_name.lower() for k in ["method", "result", "finding"])
                            all_meta.append({
                                "source": f,
                                "section": section_name,
                                "priority": "high" if is_high_value else "normal",
                                "timestamp": time.time()
                            })
                        
                        if any(k in section_name.lower() for k in ["method", "finding"]):
                            sections_found[section_name] = len(block_text)

                    if sections_found:
                        log_action(f"      {Colors.CYAN}[CORE SECTIONS]{Colors.ENDC} Mapped: {', '.join(sections_found.keys())}")
                        
                        # Increment session and long-term ingest count
                        self.stats['ingested_total'] += 1
                        if f.startswith("BOOK_"): self.stats['ingested_books'] += 1
                        elif f.startswith("PAPER_"): self.stats['ingested_papers'] += 1
                
                except Exception as e:
                    log_action(f"    {Colors.FAIL}[Migration Error]{Colors.ENDC} {f}: {str(e)[:100]}")

            # Commit batch to persistence
            if all_chunks:
                self.brain.add_batch(all_chunks, all_embeddings, all_meta)
                log_action(f"    {Colors.CYAN}[BRAIN SAVED]{Colors.ENDC} Persistent Index expanded to {self.brain.index.ntotal} chunks.")
                # Log metrics for Overlord dashboard
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
        """Professor-level PDF validation: checks for corruption, size, and text density."""
        try:
            if not os.path.exists(file_path) or os.path.getsize(file_path) < 20480:
                return False, "File too small or missing"
            
            with pdfplumber.open(file_path) as pdf:
                if len(pdf.pages) > 0 and len(pdf.pages[0].extract_text() or "") > 50:
                    return True, "Valid PDF"
                elif len(pdf.pages) == 0:
                     return False, "Empty PDF"
                else:
                    return False, "Unreadable text"
        except Exception as e:
            return False, f"Validation Error: {e}"

    async def run_scihub_layer(self, seed):
        """Discovery Layer: Filters specifically for International Peer-Reviewed Journals."""
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
        """Discovery Layer: Specialized for arXiv pre-prints and technical theses."""
        log_action(f"{Colors.BOLD}{Colors.BLUE}[RESEARCHER ACTION]{Colors.ENDC} Scanning arXiv Pre-prints: '{seed}'")
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            def search_arxiv():
                import arxiv
                client = arxiv.Client()
                search = arxiv.Search(
                    query=f"{seed} AND (abs:thesis OR abs:dissertation)",
                    max_results=20,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                return list(client.results(search))

            log_action(f"  > Connecting to export.arxiv.org API...")
            
            # Use explicit executor to avoid task context issues
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(search_arxiv)
                try:
                    results = future.result(timeout=45)
                except Exception as e:
                    if 'timeout' in str(e).lower() or isinstance(e, TimeoutError):
                        log_action(f"    {Colors.WARNING}[TIMEOUT]{Colors.ENDC} arXiv API unresponsive after 45s.")
                        return
                    raise
            
            for result in results:
                safe_title = re.sub(r'[\\/*?"<>|]', "", result.title)[:60].strip()
                filename = f"PAPER_ARXIV_{safe_title}.pdf"
                file_path = os.path.join(self.pdfs_dir, filename)
                
                if not os.path.exists(file_path):
                    log_action(f"    - Targeting arXiv ID: {result.entry_id.split('/')[-1]} ('{result.title[:30]}')") 
                    try:
                        log_action(f"      * Visiting {result.pdf_url} to fetch stream...")
                        result.download_pdf(dirpath=self.pdfs_dir, filename=filename)
                        self.stats['downloaded'] += 1
                    except Exception as dl_e:
                        log_action(f"      {Colors.WARNING}[DL FAIL]{Colors.ENDC} {dl_e}")
        except Exception as e:
            log_action(f"  {Colors.FAIL}[DISCOVERY ERROR]{Colors.ENDC} arXiv Layer: {e}")

    async def run_semantic_scholar_layer(self, seed):
        """Discovery Layer: Uses Semantic Scholar with Q1-Q4 Quality Tiering."""
        log_action(f"{Colors.BOLD}{Colors.CYAN}[RESEARCHER ACTION]{Colors.ENDC} Querying Semantic Scholar (S2): '{seed}'")
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            def search_s2():
                from semanticscholar import SemanticScholar
                s2 = SemanticScholar()
                # Fetch a larger pool to allow for quality filtering (limit=50)
                return s2.search_paper(seed, limit=50, open_access_pdf=True)

            log_action(f"  > Accessing Semantic Scholar API (Impact Analysis)...")
            
            # Use explicit executor to avoid task context issues
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

            # QUALITY CONTROL LOGIC (Simulating Q1-Q4 based on Citation Impact)
            # We assume high citation count correlates with Q1/Q2 Journals.
            
            # 1. Convert to list and extract metadata
            papers = []
            for p in results:
                if p.openAccessPdf:
                    papers.append({
                        'obj': p,
                        'cites': p.citationCount if p.citationCount else 0,
                        'title': p.title,
                        'url': p.openAccessPdf.get('url')
                    })
            
            # 2. Start Processing by Tiers
            # Tier 1 (Q1 Proxy): Highly Cited (>100 citations or Top 10%)
            papers.sort(key=lambda x: x['cites'], reverse=True)
            
            tiers = {
                'Q1 (Top Impact)': [p for p in papers if p['cites'] >= 100],
                'Q2 (High Impact)': [p for p in papers if 30 <= p['cites'] < 100],
                'Q3 (Standard)': [p for p in papers if 5 <= p['cites'] < 30],
                'Q4 (Emerging)': [p for p in papers if p['cites'] < 5]
            }

            log_action(f"  > Impact Distribution: Q1:{len(tiers['Q1 (Top Impact)'])} | Q2:{len(tiers['Q2 (High Impact)'])} | Q3:{len(tiers['Q3 (Standard)'])} | Q4:{len(tiers['Q4 (Emerging)'])}")

            # 3. Aggressive Retrieval Sequence (Best First)
            processed_count = 0
            for tier_name, tier_papers in tiers.items():
                if not tier_papers: continue
                
                log_action(f"    {Colors.BOLD}{Colors.GREEN}[QUALITY TIER]{Colors.ENDC} Scanning {tier_name} papers...")
                
                for p_meta in tier_papers:
                    # HEURISTIC: If we have enough high quality stuff, skip the junk (Q4)
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
            
        # CIRCUIT BREAKER: If fail, spend time reading what we have.
        # This prevents getting stuck in a "Hunt Loop" while books wait.
    async def run_annas_layer(self, seed):
        """Book Hunter Layer: Uses the new V2 Scraper for robustness."""
        log_action(f"{Colors.BOLD}{Colors.MAGENTA}[RESEARCHER ACTION]{Colors.ENDC} Hunting for Academic Books (V2): '{seed}'")
        try:
            # Skip Google Books API for speed, search directly on Anna's
            await self.fetch_from_annas_v2(seed, os.path.join(self.pdfs_dir, f"BOOK_{seed[:30].replace(' ', '_')}.pdf"))
        except Exception as e:
            log_action(f"  {Colors.FAIL}[DISCOVERY ERROR]{Colors.ENDC} Book Layer: {e}")

    # DEPRECATED: Old Libgen method removed to prevent timeouts
    # async def fetch_from_annas(self, query, file_path): ...

    async def retrieve_item(self, id_val, title, type_val):
        safe_title = re.sub(r'[\\/*?Source:"<>|]', "", title)[:60].strip()
        filename = f"{type_val.upper()}_{safe_title}.pdf"
        file_path = os.path.join(self.pdfs_dir, filename)
        
        if os.path.exists(file_path): 
            log_action(f"      [SKIP] '{title[:30]}...' already exists in library.")
            return

        async def _search_sequence():
            
            # --- TIER 1: LEGAL SPEED LAYER (Unpaywall / OpenAlex) ---
            if type_val == "paper" and id_val and isinstance(id_val, str) and id_val.startswith("10."):
                if await self.run_legal_speed_layer(id_val, file_path):
                     is_ok, msg = self.validate_pdf(file_path)
                     if is_ok:
                         log_action(f"      {Colors.GREEN}[VERIFIED]{Colors.ENDC} Legal Speed Layer Success.")
                         self.stats['downloaded'] += 1; return True

            # --- TIER 2: ANNA'S ARCHIVE V2 (The Tank) ---
            if await self.fetch_from_annas_v2(id_val or title, file_path):
                 is_ok, msg = self.validate_pdf(file_path)
                 if is_ok:
                     log_action(f"      {Colors.GREEN}[VERIFIED]{Colors.ENDC} Anna's Archive V2 Success.")
                     self.stats['downloaded'] += 1; return True

            # --- TIER 3: TELEGRAM NEGOTIATION ---
            if type_val == "paper" and id_val and isinstance(id_val, str) and id_val.startswith("10."):
                log_action(f"      [ACTION] Negotiating with Telegram Sci-Hub Bot for DOI: {id_val}")
                if await self.fetch_via_telegram(id_val, file_path):
                    is_ok, msg = self.validate_pdf(file_path)
                    if is_ok: 
                        log_action(f"      {Colors.GREEN}[VERIFIED]{Colors.ENDC} Telegram Download Success: '{title[:30]}...'")
                        self.stats['downloaded'] += 1; return True
                    else: 
                        log_action(f"      {Colors.FAIL}[REJECTED]{Colors.ENDC} {msg}. Deleting local copy.")
                        os.remove(file_path); self.stats['rejected'] += 1

            # 3. Direct Sci-Hub Fallback (scidownl)
            if type_val == "paper" and id_val and isinstance(id_val, str) and ("/" in id_val or "10." in id_val):
                log_action(f"      [ACTION] Exhausting direct Sci-Hub infrastructure for DOI: {id_val}")
                try:
                    if await self.fetch_direct(id_val, file_path):
                        is_ok, msg = self.validate_pdf(file_path)
                        if is_ok: 
                            log_action(f"      {Colors.GREEN}[VERIFIED]{Colors.ENDC} Sci-Hub Direct Success: '{title[:30]}...'")
                            self.stats['downloaded'] += 1; return True
                        else: 
                            log_action(f"      {Colors.FAIL}[REJECTED]{Colors.ENDC} {msg}. Deleting local copy.")
                            os.remove(file_path); self.stats['rejected'] += 1
                except Exception as e:
                    # Catch the "Timeout inside task" error from scidownl and ignore it
                    log_action(f"      [SCIHUB INTERNAL ERROR] Lib skipped: {str(e)[:50]}")
            return False

        try:
            # RETRIEVAL SEQUENCE
            # (Global timeout removed to prevent task conflicts; layers have own timeouts)
            await _search_sequence()
        except Exception as e:
            log_action(f"      [ERROR] Retrieval failed: {e}")
        
        # 3. Scidownl Fallback (DOI only)
        if type_val == "paper" and id_val and isinstance(id_val, str) and "10." in id_val:
             log_action(f"      [ACTION] Attempting Scidownl DOI Resolution fallback for {id_val}")
             if await self.fetch_direct(id_val, file_path):
                  is_ok, msg = self.validate_pdf(file_path)
                  if is_ok: 
                      log_action(f"      {Colors.GREEN}[VERIFIED]{Colors.ENDC} Scidownl Success: '{title[:30]}...'")
                      self.stats['downloaded'] += 1; return
                  else: 
                      log_action(f"      {Colors.FAIL}[REJECTED]{Colors.ENDC} {msg}. Deleting local copy.")
                      os.remove(file_path); self.stats['rejected'] += 1

    async def run_legal_speed_layer(self, doi, file_path):
        """Tier 1: Legal Speed Layer (Unpaywall + OpenAlex)"""
        if not doi: return False
        try:
             log_action(f"      [SPEED LAYER] Checking Unpaywall for legal FullText...")
             # Run in executor to avoid blocking
             loop = asyncio.get_running_loop()
             def check_unpaywall():
                 # Enforce internal timeout if library supports it, else wrapper handles it
                 return unpywall.Unpaywall.doi(doi)
             
             # HARD TIMEOUT: Unpaywall library is known to hang on Windows/IPv6
             paper = await asyncio.wait_for(loop.run_in_executor(None, check_unpaywall), timeout=10)
             if paper and paper.is_oa and paper.best_oa_location:
                 url = paper.best_oa_location['url']
                 log_action(f"      {Colors.GREEN}[SPEED LAYER]{Colors.ENDC} HIT! Legal PDF found: {url[:50]}...")
                 return await self.fetch_direct(url, file_path)
             return False
        except Exception: 
            return False

    async def fetch_from_annas_v2(self, query, file_path):
        """Tier 2: Anna's Archive Scraper (Custom Multi-Mirror with Health-Check)"""
        mirrors = ["https://annas-archive.li", "https://annas-archive.se", "https://annas-archive.pm", "https://annas-archive.org"]
        
        # MIRROR HEALTH-CHECK: Skip mirrors that failed in the last 5 minutes
        if not hasattr(self, '_mirror_blacklist'):
            self._mirror_blacklist = {}  # {mirror_url: failure_timestamp}
        
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
                    self._mirror_blacklist[base_url] = time.time()  # Mark as failed
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
                self._mirror_blacklist[base_url] = time.time()  # Mark as failed
                return None

        for mirror in healthy_mirrors:
            download_link = await try_mirror(mirror)
            if download_link:
                log_action(f"      [ANNAS HIT] Using Mirror {mirror}: {download_link[:50]}...")
                return await self.fetch_direct(download_link, file_path)
            
        return False

    async def fetch_from_annas(self, query, file_path):
        """Libgen-based retrieval with mirror exhaustion logic."""
        log_action(f"      [MIRROR ROTATION] Resolving mirrors for: '{query.split('/')[-1] if '/' in str(query) else query}'")
        try:
            loop = asyncio.get_running_loop()
            def search_exhausted():
                from libgen_api import LibgenSearch
                s = LibgenSearch()
                log_action(f"        > Navigating to Libgen Metadata Engine...")
                try:
                    results = s.search_title(query)
                except Exception as e:
                     log_action(f"        {Colors.WARNING}[MIRROR FAIL]{Colors.ENDC} Libgen main mirror unreachable: {e}")
                     return None
                
                if not results: 
                    log_action(f"        {Colors.WARNING}[WARN]{Colors.ENDC} No matches in Libgen Index.")
                    return None
                
                for i, r in enumerate(results[:3]):
                    log_action(f"        > Analyzing Entry #{i+1}: {r.get('Author', 'Unknown')} - {r.get('Title', 'Unknown')[:30]}...")
                    try:
                        log_action(f"        > Clicking 'GET' to resolve mirror list...")
                        links = s.resolve_download_links(r)
                        for mirror_key in ['GET', 'Cloudflare', 'IPFS.io', 'Pinata']:
                            if mirror_key in links:
                                log_action(f"        {Colors.GREEN}[HIT]{Colors.ENDC} Selecting Mirror: {mirror_key} -> {links[mirror_key][:60]}...")
                                return links[mirror_key]
                    except: continue
                return None

            try:
                download_link = await asyncio.wait_for(loop.run_in_executor(None, search_exhausted), timeout=45)
            except asyncio.TimeoutError:
                log_action(f"        {Colors.WARNING}[TIMEOUT]{Colors.ENDC} Libgen Search timed out after 45s.")
                download_link = None

            if download_link: 
                log_action(f"      [FETCH] Mirror Link Acquired: {download_link[:60]}...")
                return await self.fetch_direct(download_link, file_path)
            return False
        except Exception as e:
            log_action(f"      {Colors.FAIL}[LIBGEN ERROR]{Colors.ENDC} {e}")
            return False

    async def fetch_direct(self, url_or_doi, file_path):
        """High-Resilience direct downloader for URLs and DOIs."""
        try:
            loop = asyncio.get_running_loop()
            if url_or_doi.startswith("http"):
                log_action(f"        * Establishing Secure Stream to: {url_or_doi[:50]}...")
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
                # Use a larger chunk size for speed and resilience
                r = await loop.run_in_executor(None, lambda: requests.get(url_or_doi, headers=headers, timeout=30, stream=True))
                if r.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=32768): 
                            if chunk: f.write(chunk)
                    return os.path.exists(file_path) and os.path.getsize(file_path) > 10000
                else:
                    log_action(f"        {Colors.FAIL}[DENIED]{Colors.ENDC} Status: {r.status_code}")

            if isinstance(url_or_doi, str) and ("/" in url_or_doi or "10." in url_or_doi):
                from scidownl import SciHub
                log_action(f"        * Routing DOI via Sci-Hub Global Proxies: {url_or_doi}")
                def run_scidownl():
                    try:
                        paper = SciHub(url_or_doi, file_path)
                        paper.download()
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
        """Professor Persona: Synthesizes the intersection of methodology, findings, and quantitative rigor."""
        log_action(f"\n{Colors.BOLD}{Colors.CYAN}[PROFESSOR METHODOLOGICAL REFLECTION]{Colors.ENDC}")
        thoughts = [
            "Critiquing the stochastic sampling methods used in recent bot-identification studies...",
            "Synthesizing the empirical findings of network density vs. semantic variance decay...",
            "Evaluating the validity of Bayesian models applied to synthetic consensus detection...",
            "Linking methodological silos in computational linguistics to the failure of 'Dead Internet' detection...",
            "Analyzing the correlation between data collection biases and reported algorithmic echo-chamber effects...",
            "Re-evaluating post-human discourse topologies through the lens of robust quantitative findings...",
            "Deconstructing the feedback loops between algorithmic recommendation and identity fragmentation...",
            "Measuring the entropy gradient of LLM-generated information silos in academic discourse...",
            "Identifying the shift from observational to performative truth-claims in digital networks...",
            "Assessing the fragility of consensus protocols in the age of high-frequency cognitive automation..."
        ]
        thought = random.choice(thoughts)
        log_action(f"  > {Colors.ITALIC}'{thought}'{Colors.ENDC}")

    async def run_autonomous_loop(self):
        await self.client.start()
        print(f"{Colors.GREEN}[SYSTEM] World-Class Researcher Mode Active.{Colors.ENDC}", flush=True)

        seeds = [
            'dead internet theory', 'algorithmic governance', 'digital simulation of consensus',
            'automated propaganda bots', 'internet traffic fragmentation', 'turing test failure cases',
            'synthetic identity clusters', 'computational linguistics in disinformation',
            'non-organic network growth', 'platform capitalism vs dead internet',
            'AI generated semantic hollows', 'epistemology of fake consensus',
            'social media bot detection PhD thesis', 'theory of non-human discourse dissertation'
        ]
        
        while True:
            # Stats line
            print(f"\n{Colors.BOLD}{Colors.BLUE}>>> PROFESSOR STATUS: {self.stats['ingested_total']} Read ({self.stats['ingested_books']} Books | {self.stats['ingested_papers']} Papers) | {self.brain.index.ntotal} Memories{Colors.ENDC}", flush=True)
            
            # --- PHASE 1: PRIORITY RESEARCH (DIRECTED) ---
            if self.priority_books:
                target_book = self.priority_books.pop(0)
                print(f"  {Colors.HEADER}[PRIORITY RESEARCH]{Colors.ENDC} Targeting: {Colors.BOLD}{target_book}{Colors.ENDC}", flush=True)
                # Try all mirrors for this specific title
                await self.run_annas_layer(target_book)
                # Also try arXiv just in case there are papers/theses about this book
                await self.run_arxiv_layer(f"Critical analysis of {target_book}")
            
            # --- PHASE 2: DISCOVERY (GAP FINDING) ---
            else:
                # Seed Evolution (Gap Finding)
                base = random.choice(seeds)
                qualifier = random.choice(self.academic_qualifiers)
                target = f"{base} {qualifier}"
                
                # --- PARALLEL DISCOVERY ---
                await asyncio.gather(
                    self.run_scihub_layer(target),
                    self.run_arxiv_layer(target),
                    self.run_semantic_scholar_layer(target),
                    self.run_annas_layer(target)
                )
            
            # --- PROCESSING ---
            self.ingest_batch()
            
            # --- PROFESSOR REFLECTION ---
            self.simulate_thought_stream()
            
            await asyncio.sleep(2)

if __name__ == "__main__":
    init_log()
    commander = StormCommander()
    
    # 1. Start Migration & Re-embedding (Sync)
    commander.ingest_batch()
    
    # 2. Verify (Sync)
    commander.verify_persistence()
    
    # 3. Resume (Async Loop)
    asyncio.run(commander.run_autonomous_loop())
