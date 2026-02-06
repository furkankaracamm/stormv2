import os
import sys
import sqlite3
import json
import numpy as np
import logging
import requests
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
WORK_DIR = r"C:\Users\Enes\.gemini\antigravity\scratch"
DB_PATH = os.path.join(WORK_DIR, "academic_brain.db")
INSIGHTS_DB_PATH = os.path.join(WORK_DIR, "insights.db")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.3"

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='[INSIGHT] %(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(WORK_DIR, "logs_academic", "insight_engine.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("INSIGHT")

class ScientificInsightEngine:
    def __init__(self):
        self._init_db()
        
    def _init_db(self):
        """Insight veritabanını ve tablolarını kurar."""
        conn = sqlite3.connect(INSIGHTS_DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS clusters
                     (id INTEGER PRIMARY KEY, topic_label TEXT, size INTEGER, center_vector TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS gaps
                     (id INTEGER PRIMARY KEY, description TEXT, 
                      geometric_score REAL, epistemic_score REAL, status TEXT, found_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()

    def load_vectors(self):
        """Ana beyinden vektörleri çeker."""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            # Not: Eğer knowledge_vectors tablosu yoksa boş döner, sistem çökmez.
            c.execute("SELECT id, vector FROM knowledge_vectors") 
            rows = c.fetchall()
            if not rows: 
                logger.warning("Veritabanında vektör bulunamadı.")
                return [], np.array([])
            
            ids = [r[0] for r in rows]
            vectors = []
            for r in rows:
                v_raw = r[1]
                if isinstance(v_raw, str):
                    try: vectors.append(json.loads(v_raw))
                    except: pass
                else:
                    vectors.append(v_raw)
            
            return ids, np.array(vectors)
        except Exception as e:
            logger.warning(f"Vektör okuma hatası: {e}")
            return [], np.array([])
        finally:
            conn.close()

    def reduce_dimensions(self, vectors, target_dim=50):
        """
        [BOYUT İNDİRGEME]
        768 boyutu 50'ye indirerek kümeleme kalitesini artırır.
        """
        if len(vectors) < target_dim + 5:
            logger.info("Veri seti PCA için küçük, ham vektör kullanılıyor.")
            return vectors
        
        logger.info(f"PCA Başlatılıyor: {vectors.shape[1]} -> {target_dim} boyut")
        pca = PCA(n_components=target_dim)
        reduced = pca.fit_transform(vectors)
        return reduced

    def analyze_gaps(self):
        logger.info(">>> INSIGHT ENGINE ÇALIŞIYOR <<<")
        ids, vectors = self.load_vectors()
        
        if len(vectors) < 10:
            logger.info("Yetersiz veri (En az 10 makale gerekir). Beklemede.")
            return

        # 1. ADIM: Boyut İndirgeme
        reduced_vectors = self.reduce_dimensions(vectors)

        # 2. ADIM: Kümeleme (DBSCAN)
        # eps=0.5 ve min_samples=3 literatür taraması için genelde iyi başlangıçtır
        clustering = DBSCAN(eps=0.5, min_samples=3, metric='cosine').fit(reduced_vectors)
        labels = clustering.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"Analiz Sonucu: {n_clusters} Küme, {n_noise} Outlier (Potansiyel Boşluk).")

        # 3. ADIM: Geometrik Gap -> Epistemik Gap Dönüşümü
        outlier_indices = np.where(labels == -1)[0]
        
        conn = sqlite3.connect(INSIGHTS_DB_PATH)
        c = conn.cursor()

        for idx in outlier_indices:
            if idx >= len(ids): continue
            vector_id = ids[idx]
            paper_text = self._get_paper_text(vector_id)
            
            if not paper_text: continue

            # --- EPİSTEMİK FİLTRE ---
            logger.info(f"🔍 Outlier Analizi (ID: {vector_id})...")
            is_necessary, reasoning = self.evaluate_epistemic_necessity(paper_text)

            if is_necessary:
                logger.info(f"   ✅ GAP ONAYLANDI: {reasoning[:60]}...")
                c.execute("INSERT INTO gaps (description, geometric_score, epistemic_score, status) VALUES (?, ?, ?, ?)",
                          (reasoning, 1.0, 1.0, "VALIDATED"))
            else:
                logger.info(f"   🗑️ GAP REDDEDİLDİ: {reasoning[:60]}...")

        conn.commit()
        conn.close()
        logger.info(">>> ANALİZ TAMAMLANDI <<<")

    def _get_paper_text(self, vector_id):
        """Metadata DB'den makale özetini çeker."""
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT content FROM papers WHERE id=?", (vector_id,))
            res = c.fetchone()
            return res[0][:2000] if res else None # İlk 2000 karakter yeterli
        except: return None
        finally: conn.close()

    def evaluate_epistemic_necessity(self, text_segment):
        """
        [EPİSTEMİK ZORUNLULUK FİLTRESİ]
        Outlier'ın sadece 'kötü veri' mi yoksa 'teorik fırsat' mı olduğunu sorgular.
        """
        prompt = f"""
        ANALYSIS OBJECT: An outlier research paper segment in a vector space.
        CONTENT: {text_segment[:1500]}...

        TASK: Apply the "SO WHAT?" test.
        Is this outlier a "Geometric Gap" (just noise/bad data) or an "Epistemic Gap" (Theoretical Necessity)?
        
        CRITERIA:
        1. Does it contradict established theory? (Anomaly = KEEP)
        2. Does it bridge two disconnected fields? (Synthesis = KEEP)
        3. Is it just a poorly written or irrelevant text? (Noise = DISCARD)

        OUTPUT JSON:
        {{
            "is_epistemic": true/false,
            "reasoning": "Why is this theoretically significant?"
        }}
        """
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json"
            }, timeout=45)
            
            if resp.status_code == 200:
                data = resp.json()
                res_content = data.get("response", "{}")
                if isinstance(res_content, str):
                    try: res_json = json.loads(res_content)
                    except: return False, "JSON Parse Error"
                else:
                    res_json = res_content
                    
                return res_json.get("is_epistemic", False), res_json.get("reasoning", "No reasoning provided.")
        except Exception as e:
            logger.error(f"LLM Bağlantı Hatası: {e}")
            return False, "LLM Error"
        return False, "Unknown Error"

if __name__ == "__main__":
    ScientificInsightEngine().analyze_gaps()
