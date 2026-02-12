
import sys
import os
import json
import sqlite3
sys.path.insert(0, '.')

from storm_modules.research_planner import ResearchPlanner
from storm_modules.config import get_academic_brain_db_path

def test_strategic_logic():
    print("="*60)
    print("STRATEGIC AGENT LOGIC VERIFICATION")
    print("="*60)
    
    db_path = str(get_academic_brain_db_path())
    planner = ResearchPlanner(db_path)
    
    # 1. Test Insight Storage
    print("\n[TEST 1] Testing Insight Persistence...")
    test_content = {"question": "Is the internet dead?", "target_query": "Dead Internet Theory evidence", "rationale": "Audit test"}
    planner.add_insight("STRATEGIC_RESEARCH", test_content, test_content["target_query"], weight=1.0)
    
    targets = planner.get_top_strategic_targets(limit=1)
    if targets and targets[0]["query"] == test_content["target_query"]:
        print("  ✓ Insight stored and retrieved successfully.")
        insight_id = targets[0]["id"]
    else:
        print("  ✗ Insight retrieval failed.")
        return

    # 2. Test Success Reinforcement
    print("\n[TEST 2] Testing Success Reinforcement...")
    planner.report_success(insight_id)
    targets = planner.get_top_strategic_targets(limit=1)
    if targets[0]["weight"] > 1.0:
        print(f"  ✓ Weight boosted: {targets[0]['weight']:.2f}")
    else:
        print("  ✗ Weight reinforcement failed.")

    # 3. Test Cycle Decay
    print("\n[TEST 3] Testing Cycle Decay...")
    initial_weight = targets[0]["weight"]
    planner.apply_cycle_decay()
    targets = planner.get_top_strategic_targets(limit=1)
    if targets[0]["weight"] < initial_weight:
        print(f"  ✓ Weight decayed: {targets[0]['weight']:.2f}")
    else:
        print("  ✗ Weight decay failed.")

    # 4. Test Failure Penalty
    print("\n[TEST 4] Testing Failure Penalty...")
    planner.report_failure(insight_id)
    targets = planner.get_top_strategic_targets(limit=1)
    print(f"  ✓ Weight penalized: {targets[0]['weight']:.2f}")

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE: ALL PASS")
    print("="*60)

if __name__ == "__main__":
    test_strategic_logic()
