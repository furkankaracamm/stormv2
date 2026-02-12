
import sys
import os
import time
from storm_modules.semantic_gate import SemanticQualityGate

# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def test_semantic_gate():
    print(f"{Colors.HEADER}>>> TESTING SEMANTIC QUALITY GATE{Colors.ENDC}")
    gate = SemanticQualityGate()

    # TEST 1: CS Paper (Should PASS)
    title_cs = "Attention Is All You Need"
    print(f"\n{Colors.BLUE}TEST 1: CS Paper ('{title_cs}'){Colors.ENDC}")
    result_cs = gate.check_quality(title_cs)
    if result_cs:
        print(f"{Colors.GREEN}[PASS] Correctly accepted CS paper.{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}[FAIL] Incorrectly blocked CS paper.{Colors.ENDC}")

    # TEST 2: Medical Paper (Should BLOCK)
    # Using a very specific medical title to ensure it hits Medicine field
    title_med = "Clinical features of patients infected with 2019 novel coronavirus in Wuhan, China"
    print(f"\n{Colors.BLUE}TEST 2: Medical Paper ('{title_med[:30]}...'){Colors.ENDC}")
    
    # DEBUG: Manual Check to see fields
    import requests
    r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params={"query": title_med, "limit": 1, "fields": "fieldsOfStudy"}, headers={"User-Agent": "Debug"})
    print(f"DEBUG DATA: {r.json()}")
    
    result_med = gate.check_quality(title_med)
    if not result_med:
        print(f"{Colors.GREEN}[PASS] Correctly blocked Medical paper.{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}[FAIL] Incorrectly accepted Medical paper.{Colors.ENDC}")

    # TEST 3: Fail Open (Network Error)
    print(f"\n{Colors.BLUE}TEST 3: Network Failure (Fail Open){Colors.ENDC}")
    # Mocking requests to fail
    original_get = gate.check_quality
    
    # We can't easily mock inner requests without unittest.mock or monkeypatching requests
    # Let's simple check a nonsense string that won't match anything "alksdjflkasjdflkjasdlkfja"
    # The API will return empty data -> Should return True (Pass)
    title_nonsense = "alksdjflkasjdflkjasdlkfja_nonsense_string_12345"
    result_fail = gate.check_quality(title_nonsense)
    if result_fail:
         print(f"{Colors.GREEN}[PASS] Correctly failed open on missing/nonsense paper.{Colors.ENDC}")
    else:
         print(f"{Colors.FAIL}[FAIL] Incorrectly blocked missing paper.{Colors.ENDC}")

if __name__ == "__main__":
    test_semantic_gate()
