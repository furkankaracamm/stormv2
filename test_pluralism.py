
import random
import unittest
from unittest.mock import MagicMock, patch

# Mocking Colors for the test environment
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'

def log_action(msg):
    pass

class MockStorm:
    def __init__(self):
        self.pluralism_threshold = 0.2
        self.priority_books = []
        self.research_planner = MagicMock()
        self.run_scihub_layer = MagicMock()
        self.run_annas_layer = MagicMock()
        self.run_arxiv_layer = MagicMock()
        self.ontology = MagicMock()
        self.academic_qualifiers = ["test"]
        self.session_download_count = 0
        self.session_download_limit = 10
        self.brain = MagicMock()

    async def test_loop_logic(self, iterations=100):
        results = {"strategic": 0, "pluralism_bypass": 0, "ontology": 0}
        
        # Mapping the refactored logic into a testable function
        for _ in range(iterations):
            strategic_targets = [{"id": 1, "query": "test", "weight": 1.0}]
            
            # [PLURALISMRefactor] Check for bypass roll
            pluralism_bypass = random.random() < self.pluralism_threshold
            
            if strategic_targets and not pluralism_bypass:
                results["strategic"] += 1
            elif self.priority_books:
                if strategic_targets and pluralism_bypass:
                    results["pluralism_bypass"] += 1
            else:
                if strategic_targets and pluralism_bypass:
                    results["pluralism_bypass"] += 1
                    results["ontology"] += 1
                else:
                    results["ontology"] += 1
        
        return results

class TestPluralism(unittest.IsolatedAsyncioTestCase):
    async def test_distribution(self):
        storm = MockStorm()
        stats = await storm.test_loop_logic(1000)
        
        print(f"\n[TEST] Pluralism Distribution (1000 Cycles):")
        print(f"  > Strategic Steer: {stats['strategic']}")
        print(f"  > Pluralism Bypass: {stats['pluralism_bypass']}")
        
        # Verify 20% +/- 5% Slack
        self.assertGreater(stats['pluralism_bypass'], 150)
        self.assertLess(stats['pluralism_bypass'], 250)
        print("  ✓ SUCCESS: Distribution within pluralistic guardrails (15%-25%).")

if __name__ == "__main__":
    unittest.main()
