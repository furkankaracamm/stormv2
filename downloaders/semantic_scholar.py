"""
Semantic Scholar Downloader
Zero-Error Architecture
"""
import time
from typing import List, Dict
from .base_downloader import BaseDownloader, logger

class SemanticScholarDownloader(BaseDownloader):
    """Downloads papers from Semantic Scholar API (Open Access only)."""
    
    API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    def search_and_download(self, query: str, limit: int = 5) -> int:
        """Search Semantic Scholar and download Open Access PDFs."""
        logger.info(f"Searching Semantic Scholar for: {query}")
        
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,url,isOpenAccess,openAccessPdf"
        }
        
        try:
            response = self.session.get(self.API_URL, params=params, timeout=30)
            if response.status_code == 429:
                logger.warning("Semantic Scholar rate limit reached")
                return 0
                
            response.raise_for_status()
            data = response.json()
            
            count = 0
            for paper in data.get('data', []):
                if not paper.get('isOpenAccess'):
                    continue
                    
                pdf_info = paper.get('openAccessPdf')
                if pdf_info and pdf_info.get('url'):
                    pdf_url = pdf_info['url']
                    title = paper.get('title', 'Unknown')
                    
                    # Clean title
                    safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:50]
                    filename = f"S2_{safe_title}.pdf"
                    
                    try:
                        if self._download_file(pdf_url, filename):
                            count += 1
                            time.sleep(1)
                    except Exception as e:
                        logger.warning(f"Failed to download {pdf_url}: {e}")
            
            return count
            
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return 0
