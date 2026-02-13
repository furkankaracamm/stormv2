"""
Verification Script for Hybrid DOI Resolver Pipeline
Tests:
1. OpenAlex Fallback
2. Unpaywall (Legal Speed Layer)
3. Direct Sci-Hub/Scidownl
4. Anna's/LibGen Direct
"""

import asyncio
import os
import sys
from storm_commander import StormCommander, Colors

# Mock Log Action to avoid cluttering real logs
def mock_log(msg):
    # Strip colors for clean output
    import re
    clean = re.sub(r'\033\[[0-9;]*m', '', msg)
    print(f"[TEST LOG] {clean}")

async def run_verification():
    print(f"{Colors.HEADER}>>> STARTING HYBRID DOI RESOLVER VERIFICATION{Colors.ENDC}")
    
    # Initialize Commander (headless-ish)
    commander = StormCommander()
    
    # [TEST BYPASS] Disable ScopeGuard for infrastructure test
    commander.scope_guard.is_safe = lambda x: True
    print(f"{Colors.WARNING}[TEST] ScopeGuard Disabled for Verification{Colors.ENDC}")
    
    # 1. Test Data (Mix of OA and Paywalled)
    test_cases = [
        # Open Access (Should hit Unpaywall or OpenAlex)
        {"doi": "10.1371/journal.pcbi.1004668", "title": "Ten Simple Rules for Better Figures", "expected": "OA"},
        
        # Paywalled (Should hit Scidownl) - Old canonical example
        {"doi": "10.1038/nature14246", "title": "Deep learning (LeCun, Bengio, Hinton)", "expected": "SCIHUB"},
        
        # Book (Should hit Anna's/LibGen)
        {"query": "Simulacra and Simulation Baudrillard", "type": "book", "expected": "ANNAS"}
    ]
    
    success_count = 0
    
    # Ensure test dir exists
    os.makedirs("test_downloads", exist_ok=True)
    commander.pdfs_dir = os.path.abspath("test_downloads")
    
    for case in test_cases:
        target = case.get("doi") or case.get("query")
        print(f"\n{Colors.BOLD}Testing Target: {case['title']}{Colors.ENDC}")
        print(f"Target Identifier: {target}")
        
        item_type = "paper" if case.get("doi") else "book"
        filename = f"TEST_{case['expected']}_{hash(target)}.pdf"
        file_path = os.path.join(commander.pdfs_dir, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
        success = False
        if item_type == "paper":
             success = await commander.retrieve_item(target, case['title'], "paper")
        else:
             # Manually trigger Anna's layer for book test as retrieve_item logic is complex for books
             success = await commander.fetch_from_annas_v2(target, file_path)
             
        if success and os.path.exists(file_path) and os.path.getsize(file_path) > 10000:
            print(f"{Colors.GREEN}[PASS] Successfully downloaded.{Colors.ENDC}")
            success_count += 1
        else:
            print(f"{Colors.FAIL}[FAIL] Download failed.{Colors.ENDC}")
            
    print(f"\n{Colors.HEADER}>>> VERIFICATION COMPLETE: {success_count}/{len(test_cases)} Passed{Colors.ENDC}")

if __name__ == "__main__":
    # Suppress normal init logs
    import logging
    logging.disable(logging.CRITICAL)
    
    try:
        asyncio.run(run_verification())
    except KeyboardInterrupt:
        pass
