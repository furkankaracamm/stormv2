"""
Verify Downloaders Package
Zero-Error Architecture
"""
import sys
import os
from pathlib import Path

# Add storm directory to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_downloaders():
    print("=" * 60)
    print("DOWNLOADERS PACKAGE VERIFICATION")
    print("=" * 60)
    
    modules = [
        "downloaders.arxiv_downloader",
        "downloaders.semantic_scholar",
        "downloaders.annas_archive",
        "downloaders.telegram_bot_integration"
    ]
    
    passed = 0
    for mod in modules:
        try:
            full_name = mod
            class_name = mod.split('.')[-1].replace('_', ' ').title().replace(' ', '')
            if 'Arxiv' in class_name: class_name = 'ArxivDownloader'
            if 'Semantic' in class_name: class_name = 'SemanticScholarDownloader'
            if 'Annas' in class_name: class_name = 'AnnasArchiveDownloader'
            if 'Telegram' in class_name: class_name = 'TelegramBotIntegration'
            
            exec(f"from {mod} import {class_name}")
            print(f"  ✓ Import {class_name}: OK")
            
            # Instantiate to check init logic
            exec(f"downloader = {class_name}()")
            print(f"  ✓ Init {class_name}: OK")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed {mod}: {e}")
            
    print("-" * 60)
    if passed == 4:
        print("STATUS: ✅ ALL DOWNLOADERS READY")
        return 0
    else:
        print(f"STATUS: ❌ {4-passed} MODULES FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(verify_downloaders())
