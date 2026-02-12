import os
import shutil
import sqlite3
import json
import time
from typing import List, Dict
from storm_modules.config import get_academic_brain_db_path

class AcademicLibrarian:
    """
    The Librarian organizes raw PDFs into a semantic folder structure.
    It reads the ontology and moves files from 'storm_data/pdfs' to 'storm_data/library/...'.
    """
    def __init__(self, commander):
        self.commander = commander
        self.brain = commander.brain
        self.base_dir = commander.base_dir
        self.source_dir = commander.pdfs_dir
        self.library_dir = os.path.join(self.base_dir, "storm_data", "library")
        
        # Ensure library root exists
        if not os.path.exists(self.library_dir):
            os.makedirs(self.library_dir)
            
        print(f"  [LIBRARIAN] Initialized. Ready to organize stacks.")

    def organize_stack(self):
        """Main function to classify and move files."""
        print(f"\n======== LIBRARIAN WORKING ========")
        
        # 1. Get all processed files from Brain metadata
        # We need files that are in the source directory
        files = [f for f in os.listdir(self.source_dir) if f.endswith('.pdf')]
        
        if not files:
            print("  [LIBRARIAN] No files in intake folder to organize.")
            return

        organized_count = 0
        
        for filename in files:
            file_path = os.path.join(self.source_dir, filename)
            
            # Skip if file is currently being written (too small or locked)
            if os.path.getsize(file_path) < 100: continue
            
            # 2. Determine Topic
            # We ask the Brain: "What is the best matching topic for this file?"
            # Since we don't have per-file topic tags in metadata yet,
            # we can infer it from the Semantic Search or filename.
            
            # Strategy: Use the filenames/content to find the closest Topic in Ontology.
            topic, theory = self.classify_file(filename)
            
            if topic:
                # 3. Move File
                target_folder = os.path.join(self.library_dir, self.sanitize(theory), self.sanitize(topic))
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    
                target_path = os.path.join(target_folder, filename)
                
                try:
                    shutil.move(file_path, target_path)
                    print(f"  [MOVED] {filename[:30]}... -> {theory}/{topic}")
                    organized_count += 1
                except Exception as e:
                    print(f"  [ERROR] Could not move {filename}: {e}")
            else:
                # If no topic found, maybe move to "Unclassified"
                pass

        print(f"======== LIBRARIAN FINISHED ({organized_count} moved) ========\n")

    def classify_file(self, filename):
        """
        Deduce the best Topic/Theory for a file.
        Simple approach: Check if filename contains keywords from Ontology.
        Better approach: Use the Embedding Model (since we have it loaded in Commander).
        """
        # Get Ontology Topics
        topics = self.commander.ontology.get_all_topics()
        
        # 1. Cheap Check: Filename matching
        clean_name = filename.lower().replace('_', ' ')
        
        best_topic = "General"
        best_theory = "Dead Internet Theory" # Default for now
        
        # We can use the Commander's embedding model to find best similarity
        # between filename and topics.
        try:
            # Topic Embeddings (Cache this ideally)
            topic_embeddings = self.commander.model.encode(topics)
            file_embedding = self.commander.model.encode([clean_name])[0]
            
            import numpy as np
            # Cosine Similarity
            sims = np.dot(topic_embeddings, file_embedding) / (
                np.linalg.norm(topic_embeddings, axis=1) * np.linalg.norm(file_embedding)
            )
            
            best_idx = np.argmax(sims)
            
            # FORCE CLASSIFICATION (User Request: No "Unclassified")
            # We take the best match even if confidence is low.
            best_topic = topics[best_idx]
            
            # Optional: Log if confidence is very low 
            if sims[best_idx] < 0.15:
                print(f"  [LIBRARIAN] Low confidence ({sims[best_idx]:.2f}) for {filename} -> {best_topic}")
                
        except Exception as e:
            print(f"  [LIBRARIAN AI ERROR] {e}")
            return None, None

        return best_topic, best_theory

    def sanitize(self, name):
        return "".join(x for x in name if x.isalnum() or x in " -_").strip()
