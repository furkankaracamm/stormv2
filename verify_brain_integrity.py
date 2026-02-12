
import os
import sys
from storm_commander import StormCommander, Colors, PersistentBrain

def verify_brain_search():
    commander = StormCommander()
    brain = commander.brain
    
    print(f"{Colors.HEADER}>>> BRAIN INTEGRITY CHECK{Colors.ENDC}")
    print(f"FAISS Size: {brain.index.ntotal}")
    
    # Search for a common term
    term = "algorithm"
    print(f"Searching for: '{term}'")
    
    emb = commander.model.encode([term])[0]
    results = brain.search(emb, k=5)
    
    if not results:
        print(f"{Colors.FAIL}[FAIL] No results found (even for common term).{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}[RESULTS FOUND] {len(results)}{Colors.ENDC}")
        for i, r in enumerate(results):
            content = r.get('content', 'MISSING')
            filename = r.get('filename', 'UNKNOWN')
            print(f"Result {i+1}: File={filename}")
            print(f"Snippet: {content[:100]}...")
            
            if content == 'MISSING' or len(content) < 10:
                print(f"{Colors.FAIL}  -> CRITICAL: Text content missing or empty!{Colors.ENDC}")

if __name__ == "__main__":
    verify_brain_search()
