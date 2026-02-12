
import sys
import os
import asyncio
from storm_commander import StormCommander, Colors

# Mocking print to avoid spam, but we want to see logs
def mock_log(msg):
    print(msg)

async def test_scope():
    print(f"{Colors.HEADER}>>> TESTING DOWNLOADER SCOPE ENFORCEMENT{Colors.ENDC}")
    
    commander = StormCommander()
    
    # CASE 1: VIOLATION (Pizza)
    print(f"\n{Colors.BOLD}TEST 1: Out-of-Scope Query ('culinary arts pizza dough'){Colors.ENDC}")
    await commander.run_scihub_layer("culinary arts pizza dough")
    await commander.run_arxiv_layer("culinary arts pizza dough")
    await commander.retrieve_item("http://pizza.com/menu.pdf", "Perfect Pizza Dough", "paper")

    # CASE 2: COMPLIANCE (Dead Internet)
    print(f"\n{Colors.BOLD}TEST 2: In-Scope Query ('dead internet theory'){Colors.ENDC}")
    # We won't actually wait for full download, just check if it passes the gate
    # Check log output for "SCOPE BLOCKED" vs "RESEARCHER ACTION"

if __name__ == "__main__":
    # Redirect stdout to capture logs? No, just run and visually verify or check logs
    asyncio.run(test_scope())
