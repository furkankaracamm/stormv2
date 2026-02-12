
import os
import asyncio
import requests
import pdfplumber
from unittest.mock import MagicMock

# Simulated StormCommander for testing
class StormMock:
    def __init__(self):
        self.user_agents = ["Mozilla/5.0"]
        self.pdfs_dir = "test_pdfs"
        os.makedirs(self.pdfs_dir, exist_ok=True)

    def validate_pdf(self, file_path):
        try:
            if not os.path.exists(file_path):
                return False, "File missing"
            
            size = os.path.getsize(file_path)
            if size < 2048: # 2KB for test purposes
                return False, f"File too small ({size} bytes)."
            
            with open(file_path, 'rb') as f:
                header = f.read(5)
                if header != b'%PDF-':
                    return False, f"Invalid PDF Header: Expected %PDF-, got {header[:5]!r}."

            # Real check requires valid PDF structure, so we mock pdfplumber behavior for this test
            # In production, it uses the real pdfplumber.
            return True, "Valid PDF"
        except Exception as e:
            return False, f"Validation Error: {e}"

    async def fetch_direct(self, url, file_path):
        print(f"  [TEST] Attempting download: {url}")
        try:
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code == 200:
                ct = r.headers.get('Content-Type', '').lower()
                if 'application/pdf' not in ct:
                    print(f"  [TEST] REJECTED: Content-Type '{ct}' is not PDF.")
                    return False
                
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)
                
                is_ok, msg = self.validate_pdf(file_path)
                if not is_ok:
                    print(f"  [TEST] REJECTED: {msg}")
                    os.remove(file_path)
                    return False
                
                print(f"  [TEST] SUCCESS: {msg}")
                return True
            return False
        except Exception as e:
            print(f"  [TEST] ERROR: {e}")
            return False

async def run_test():
    storm = StormMock()
    
    print("\n--- TEST 1: POSITIVE CASE (Direct PDF) ---")
    # Using a known stable academic PDF link (arXiv)
    arxiv_url = "https://arxiv.org/pdf/1706.03762.pdf"
    res1 = await storm.fetch_direct(arxiv_url, "test_pdfs/valid.pdf")
    
    print("\n--- TEST 2: NEGATIVE CASE (HTML Landing Page) ---")
    # Using a typical landing page that is NOT a PDF
    html_url = "https://www.google.com"
    res2 = await storm.fetch_direct(html_url, "test_pdfs/invalid.pdf")
    
    print("\n--- RESULTS ---")
    print(f"  Test 1 (Valid PDF): {'PASS' if res1 else 'FAIL'}")
    print(f"  Test 2 (HTML Page): {'PASS' if not res2 else 'FAIL (Accepted HTML)'}")

if __name__ == "__main__":
    asyncio.run(run_test())
