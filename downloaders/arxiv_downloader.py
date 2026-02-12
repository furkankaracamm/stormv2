"""
Arxiv Downloader
Zero-Error Architecture
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
from typing import List, Dict
from .base_downloader import BaseDownloader, logger

class ArxivDownloader(BaseDownloader):
    """Downloads papers from Arxiv API."""
    
    API_URL = "http://export.arxiv.org/api/query"
    
    def search_and_download(self, query: str, limit: int = 5) -> int:
        """Search Arxiv and download PDFs."""
        logger.info(f"Searching Arxiv for: {query}")
        
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit
        }
        
        try:
            url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            count = 0
            
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                pdf_url = None
                
                for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                    if link.attrib.get('title') == 'pdf':
                        pdf_url = link.attrib['href']
                        break
                
                if pdf_url:
                    # Clean title for filename
                    safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:50]
                    filename = f"ARXIV_{safe_title}.pdf"
                    
                    if self._download_file(pdf_url, filename):
                        count += 1
                        time.sleep(1)  # Polite delay
            
            return count
            
        except Exception as e:
            logger.error(f"Arxiv search failed: {e}")
            return 0
