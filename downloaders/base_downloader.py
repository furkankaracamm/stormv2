"""
Base Downloader Class
Zero-Error Architecture: Retry logic, Validation, Logging
"""
import os
import time
import requests
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict

# Configure logger if not exists
logger = logging.getLogger("STORM_DOWNLOADER")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class BaseDownloader(ABC):
    """Abstract base class for all downloaders with enforced error handling."""
    
    def __init__(self, download_dir: str = "storm_data/papers"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "STORM-Research-Agent/1.0 (Academic; mailto:researcher@storm.io)"
        })

    def _download_file(self, url: str, filename: str) -> Optional[str]:
        """Download file with retry logic and validation."""
        filepath = self.download_dir / filename
        
        if filepath.exists() and filepath.stat().st_size > 1024:
            logger.info(f"File exists, skipping: {filename}")
            return str(filepath)
            
        for attempt in range(3):
            try:
                response = self.session.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Check message content type if possible, but PDF might have various types
                content_type = response.headers.get('Content-Type', '').lower()
                if 'html' in content_type and 'pdf' not in filename:
                     # Suspicious if we expect binary but get html
                     pass

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify size
                if filepath.stat().st_size < 1000:
                    logger.warning(f"Downloaded file too small ({filepath.stat().st_size} bytes): {filename}")
                    filepath.unlink(missing_ok=True)
                    return None
                
                logger.info(f"Successfully downloaded: {filename}")
                return str(filepath)
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/3 failed for {url}: {e}")
                time.sleep(2 * (attempt + 1))
        
        return None

    @abstractmethod
    def search_and_download(self, query: str, limit: int = 1) -> int:
        """Search and download papers. Returns count of successful downloads."""
        pass
