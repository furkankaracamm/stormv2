"""Text Chunker - Split documents into overlapping chunks for embedding."""

from typing import List


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better semantic embedding.
    
    Args:
        text: Full document text
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) > 50:  # Minimum chunk size
            chunks.append(" ".join(chunk_words))
    
    return chunks


def chunk_by_paragraphs(text: str, min_size: int = 100) -> List[str]:
    """
    Split text by paragraphs, merging small ones.
    
    Args:
        text: Full document text
        min_size: Minimum words per chunk
    
    Returns:
        List of paragraph chunks
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_words = len(para.split())
        
        if current_size + para_words < min_size:
            current_chunk.append(para)
            current_size += para_words
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_size = para_words
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
