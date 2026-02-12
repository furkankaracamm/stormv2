"""
Claim Extractor v2.0 - 5-Star LLM-Based Extraction Pipeline

Features:
- Multi-pass extraction (abstract → full text)
- Source validation (anti-hallucination)
- Claim taxonomy (FINDING, HYPOTHESIS, METHOD, LIMITATION)
- Confidence calibration (linguistic markers)
- Robust JSON parsing with repair
- Deduplication
"""

import re
import json
import sqlite3
from typing import List, Dict, Optional, Tuple
from storm_modules.config import get_academic_brain_db_path
from storm_modules.llm_gateway import get_llm_gateway


# ============================================================================
# CLAIM TAXONOMY PATTERNS
# ============================================================================

CLAIM_TAXONOMY = {
    "FINDING": {
        "patterns": [
            r"we (found|discovered|observed|identified|determined) that",
            r"(results|data|analysis|evidence) (show|indicate|demonstrate|reveal|suggest)",
            r"our (findings|results) (indicate|suggest|show)",
            r"(significantly|positively|negatively) (correlated|associated|related)",
        ],
        "keywords": ["found", "discovered", "results show", "data indicates", "significant"],
    },
    "HYPOTHESIS": {
        "patterns": [
            r"we (hypothesize|predict|expect|propose) that",
            r"(hypothesis|h\d+):?\s*",
            r"it is (hypothesized|predicted|expected) that",
        ],
        "keywords": ["hypothesize", "predict", "expect", "hypothesis"],
    },
    "METHOD": {
        "patterns": [
            r"we (developed|designed|implemented|created|proposed) (a|an|the)",
            r"(this|our) (method|approach|framework|algorithm|system) (enables|allows|provides)",
            r"we (introduce|present) (a|an) (novel|new)",
        ],
        "keywords": ["developed", "implemented", "novel method", "framework", "algorithm"],
    },
    "LIMITATION": {
        "patterns": [
            r"(limitation|limitations) (of|include)",
            r"(future work|further research) (should|could|may)",
            r"(we did not|does not) (address|consider|examine)",
            r"beyond the scope of",
        ],
        "keywords": ["limitation", "future work", "not addressed", "further research"],
    },
}

# ============================================================================
# CONFIDENCE CALIBRATION MARKERS
# ============================================================================

CONFIDENCE_MARKERS = {
    # High confidence (0.85-1.0)
    "conclusively": 0.95,
    "definitively": 0.95,
    "strongly suggest": 0.90,
    "clearly demonstrate": 0.90,
    "we conclude that": 0.90,
    "unequivocally": 0.95,
    
    # Medium-high confidence (0.75-0.85)
    "we found that": 0.85,
    "results indicate": 0.82,
    "data show": 0.82,
    "evidence suggests": 0.80,
    "our findings": 0.80,
    "significantly": 0.78,
    
    # Medium confidence (0.65-0.75)
    "suggest that": 0.70,
    "appears to": 0.68,
    "seems to": 0.65,
    "indicates": 0.70,
    
    # Low confidence (0.5-0.65)
    "may indicate": 0.60,
    "might suggest": 0.58,
    "preliminary": 0.55,
    "tentative": 0.52,
    "possible that": 0.55,
}


# ============================================================================
# CORE EXTRACTION FUNCTIONS
# ============================================================================

def extract_claims_pipeline(
    text: str,
    abstract: Optional[str] = None,
    max_claims: int = 15,
    validate: bool = True
) -> List[Dict]:
    """
    5-Star Claim Extraction Pipeline.
    
    Pipeline:
    1. Extract abstract claims (if available)
    2. Extract full-text claims
    3. Classify by taxonomy
    4. Calibrate confidence
    5. Validate against source (anti-hallucination)
    6. Deduplicate
    
    Args:
        text: Full document text
        abstract: Document abstract (optional, for prioritization)
        max_claims: Maximum claims to return
        validate: Whether to validate claims against source
    
    Returns:
        List of validated, classified claim dictionaries
    """
    if not text or len(text) < 100:
        return []
    
    llm = get_llm_gateway()
    if not llm.is_available() and not llm.gemini_available:
        return extract_claims_regex(text, max_claims)
    
    all_claims = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # PASS 1: Extract from Abstract (High Priority)
    # ─────────────────────────────────────────────────────────────────────────
    if abstract:
        abstract_claims = _extract_claims_llm(llm, abstract, max_claims=5, source="abstract")
        for c in abstract_claims:
            c["priority"] = "HIGH"
        all_claims.extend(abstract_claims)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PASS 2: Extract from Full Text
    # ─────────────────────────────────────────────────────────────────────────
    # Use sections: intro, results, discussion, conclusion
    text_sample = _smart_truncate(text, max_chars=12000)
    fulltext_claims = _extract_claims_llm(llm, text_sample, max_claims=max_claims, source="fulltext")
    for c in fulltext_claims:
        c["priority"] = c.get("priority", "MEDIUM")
    all_claims.extend(fulltext_claims)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PASS 3: Classify by Taxonomy
    # ─────────────────────────────────────────────────────────────────────────
    for claim in all_claims:
        claim["type"] = _classify_claim(claim.get("text", ""))
    
    # ─────────────────────────────────────────────────────────────────────────
    # PASS 4: Calibrate Confidence
    # ─────────────────────────────────────────────────────────────────────────
    for claim in all_claims:
        claim["confidence"] = _calibrate_confidence(
            claim.get("text", ""),
            base_confidence=claim.get("confidence", 0.7)
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # PASS 5: Validate Against Source (Anti-Hallucination)
    # ─────────────────────────────────────────────────────────────────────────
    if validate:
        validated_claims = []
        for claim in all_claims:
            is_valid, quote = _validate_claim(claim.get("text", ""), text)
            if is_valid:
                claim["validated"] = True
                claim["source_quote"] = quote
                validated_claims.append(claim)
            else:
                # Penalize but don't discard completely
                claim["validated"] = False
                claim["confidence"] *= 0.5
                if claim["confidence"] >= 0.4:
                    validated_claims.append(claim)
        all_claims = validated_claims
    
    # ─────────────────────────────────────────────────────────────────────────
    # PASS 6: Deduplicate
    # ─────────────────────────────────────────────────────────────────────────
    all_claims = _deduplicate_claims(all_claims)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Sort and Return
    # ─────────────────────────────────────────────────────────────────────────
    # Priority: HIGH > MEDIUM, then by confidence
    def sort_key(c):
        priority_score = 1.0 if c.get("priority") == "HIGH" else 0.5
        return (priority_score, c.get("confidence", 0))
    
    all_claims.sort(key=sort_key, reverse=True)
    return all_claims[:max_claims]


def _extract_claims_llm(llm, text: str, max_claims: int = 10, source: str = "text") -> List[Dict]:
    """Extract claims using LLM with robust JSON parsing."""
    
    prompt = f"""Extract scientific claims from this academic text.

TEXT:
{text}

Return ONLY a JSON array of claims. Each claim must have:
- "text": exact claim text from the document
- "type": one of FINDING, HYPOTHESIS, METHOD, LIMITATION
- "confidence": number between 0.5 and 1.0

Example format:
[
  {{"text": "Social media bots influence political discourse", "type": "FINDING", "confidence": 0.85}},
  {{"text": "We hypothesize that bot activity increases during elections", "type": "HYPOTHESIS", "confidence": 0.75}}
]

Rules:
- Extract up to {max_claims} most important claims
- Use exact wording from the text when possible
- Higher confidence for main findings, lower for hypotheses/limitations
- Return ONLY the JSON array, no other text"""

    response = llm.generate(
        prompt,
        max_tokens=2000,
        temperature=0.2,
        json_mode=True,
        prefer_long_context=True
    )
    
    if not response:
        return []
    
    claims = _parse_json_robust(response)
    for c in claims:
        c["source"] = source
    return claims


def _parse_json_robust(response: str) -> List[Dict]:
    """Parse JSON with multiple fallback strategies."""
    
    if not response:
        return []
    
    clean = response.strip()
    
    # Strategy 1: Direct parse
    try:
        result = json.loads(clean)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "claims" in result:
            return result["claims"]
    except:
        pass
    
    # Strategy 2: Remove markdown code blocks
    if "```" in clean:
        parts = clean.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("[") or part.startswith("{"):
                try:
                    result = json.loads(part)
                    if isinstance(result, list):
                        return result
                    if isinstance(result, dict) and "claims" in result:
                        return result["claims"]
                except:
                    continue
    
    # Strategy 3: Find JSON array
    start = clean.find("[")
    end = clean.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(clean[start:end+1])
        except:
            pass
    
    # Strategy 4: Find JSON object with claims
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(clean[start:end+1])
            if "claims" in obj:
                return obj["claims"]
        except:
            pass
    
    # Strategy 5: Repair common errors
    try:
        # Remove trailing commas
        fixed = re.sub(r',\s*]', ']', clean)
        fixed = re.sub(r',\s*}', '}', fixed)
        # Find array again
        start = fixed.find("[")
        end = fixed.rfind("]")
        if start != -1 and end != -1:
            return json.loads(fixed[start:end+1])
    except:
        pass
    
    return []


def _smart_truncate(text: str, max_chars: int = 12000) -> str:
    """Intelligently truncate text, keeping abstract, intro, results, conclusion."""
    
    if len(text) <= max_chars:
        return text
    
    # Try to find key sections
    sections = {
        "abstract": r"(?i)(abstract|summary)\s*[:.]?\s*\n",
        "introduction": r"(?i)(introduction|1\.\s*introduction)\s*\n",
        "results": r"(?i)(results|findings|4\.\s*results)\s*\n",
        "discussion": r"(?i)(discussion|5\.\s*discussion)\s*\n",
        "conclusion": r"(?i)(conclusion|6\.\s*conclusion)\s*\n",
    }
    
    extracted = []
    
    for name, pattern in sections.items():
        match = re.search(pattern, text)
        if match:
            start = match.start()
            # Get ~2000 chars from this section
            section_text = text[start:start+2000]
            extracted.append(section_text)
    
    if extracted:
        combined = "\n\n".join(extracted)
        if len(combined) <= max_chars:
            return combined
        return combined[:max_chars]
    
    # Fallback: first + last portions
    return text[:max_chars//2] + "\n...\n" + text[-max_chars//2:]


def _classify_claim(claim_text: str) -> str:
    """Classify claim by taxonomy using patterns and keywords."""
    
    claim_lower = claim_text.lower()
    scores = {}
    
    for claim_type, config in CLAIM_TAXONOMY.items():
        score = 0
        
        # Check patterns
        for pattern in config["patterns"]:
            if re.search(pattern, claim_lower):
                score += 2
        
        # Check keywords
        for keyword in config["keywords"]:
            if keyword in claim_lower:
                score += 1
        
        scores[claim_type] = score
    
    # Return type with highest score, default to FINDING
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return "FINDING"


def _calibrate_confidence(claim_text: str, base_confidence: float = 0.7) -> float:
    """Calibrate confidence based on linguistic markers."""
    
    claim_lower = claim_text.lower()
    
    # Find matching marker with highest/lowest confidence
    matched_confidence = None
    
    for marker, conf in CONFIDENCE_MARKERS.items():
        if marker in claim_lower:
            if matched_confidence is None:
                matched_confidence = conf
            else:
                # Average if multiple markers
                matched_confidence = (matched_confidence + conf) / 2
    
    if matched_confidence is not None:
        # Blend with base confidence
        return round((matched_confidence + base_confidence) / 2, 2)
    
    return round(base_confidence, 2)


def _validate_claim(claim_text: str, source_text: str) -> Tuple[bool, str]:
    """Validate that claim exists in source text (anti-hallucination)."""
    
    # Strategy 1: Exact substring match
    if claim_text in source_text:
        return True, claim_text
    
    # Strategy 2: Significant phrase match
    # Extract key phrases (3+ word sequences)
    words = claim_text.split()
    if len(words) >= 5:
        # Check overlapping 4-grams
        for i in range(len(words) - 3):
            phrase = " ".join(words[i:i+4])
            if phrase.lower() in source_text.lower():
                # Find the actual quote
                idx = source_text.lower().find(phrase.lower())
                if idx != -1:
                    # Extract surrounding context
                    start = max(0, idx - 20)
                    end = min(len(source_text), idx + len(phrase) + 20)
                    return True, source_text[start:end]
    
    # Strategy 3: Key term overlap
    claim_terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', claim_text.lower()))
    source_terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', source_text.lower()))
    
    if claim_terms:
        overlap = len(claim_terms & source_terms) / len(claim_terms)
        if overlap >= 0.7:  # 70% term overlap
            return True, "(term-validated)"
    
    return False, ""


def _deduplicate_claims(claims: List[Dict]) -> List[Dict]:
    """Remove duplicate or near-duplicate claims."""
    
    unique = []
    seen_texts = set()
    
    for claim in claims:
        text = claim.get("text", "").lower().strip()
        
        # Exact duplicate
        if text in seen_texts:
            continue
        
        # Near-duplicate (80% word overlap)
        is_duplicate = False
        claim_words = set(text.split())
        
        for seen in seen_texts:
            seen_words = set(seen.split())
            if claim_words and seen_words:
                overlap = len(claim_words & seen_words) / max(len(claim_words), len(seen_words))
                if overlap >= 0.8:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            seen_texts.add(text)
            unique.append(claim)
    
    return unique


# ============================================================================
# LEGACY FUNCTIONS (Backward Compatibility)
# ============================================================================

def extract_claims_llm(text: str, max_claims: int = 15) -> List[Dict]:
    """Legacy wrapper for extract_claims_pipeline."""
    return extract_claims_pipeline(text, max_claims=max_claims, validate=True)


def extract_claims(text: str, max_claims: int = 15) -> List[Dict]:
    """Legacy regex-based extraction (fallback)."""
    return extract_claims_regex(text, max_claims)


def extract_claims_regex(text: str, max_claims: int = 15) -> List[Dict]:
    """Extract claims using regex patterns only (fast fallback)."""
    
    if not text:
        return []
    
    claims = []
    sentences = re.split(r'[.!?]\s+', text)
    
    all_patterns = []
    for claim_type, config in CLAIM_TAXONOMY.items():
        for pattern in config["patterns"]:
            all_patterns.append((pattern, claim_type))
    
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30 or len(sent) > 500:
            continue
        
        for pattern, claim_type in all_patterns:
            if re.search(pattern, sent.lower()):
                confidence = _calibrate_confidence(sent)
                claims.append({
                    "text": sent,
                    "type": claim_type,
                    "confidence": confidence,
                    "validated": False,
                    "source": "regex"
                })
                break
    
    claims.sort(key=lambda x: x["confidence"], reverse=True)
    return claims[:max_claims]


def extract_hypotheses(text: str) -> List[Dict]:
    """Extract formal hypotheses from text."""
    if not text:
        return []
    
    hypotheses = []
    sentences = re.split(r'[.!?]\s+', text)
    
    for sent in sentences:
        sent = sent.strip()
        for pattern in CLAIM_TAXONOMY["HYPOTHESIS"]["patterns"]:
            if re.search(pattern, sent.lower()):
                hypotheses.append({
                    "text": sent,
                    "type": "HYPOTHESIS",
                    "confidence": 0.85,
                    "validated": False
                })
                break
    
    return hypotheses


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def save_claims_to_db(filename: str, claims: List[Dict], db_path: Optional[str] = None) -> bool:
    """Save extracted claims to database with full metadata."""
    try:
        path = db_path or str(get_academic_brain_db_path())
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        # Enhanced table schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                claim_text TEXT,
                claim_type TEXT,
                confidence REAL,
                validated INTEGER DEFAULT 0,
                source_quote TEXT,
                priority TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        for claim in claims:
            cursor.execute('''
                INSERT INTO paper_claims 
                (filename, claim_text, claim_type, confidence, validated, source_quote, priority) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                filename,
                claim.get("text", ""),
                claim.get("type", "FINDING"),
                claim.get("confidence", 0.5),
                1 if claim.get("validated") else 0,
                claim.get("source_quote", ""),
                claim.get("priority", "MEDIUM")
            ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[CLAIM DB ERROR] {e}")
        return False


def get_top_claims(limit: int = 20, db_path: Optional[str] = None) -> List[Dict]:
    """Retrieve highest-confidence validated claims."""
    try:
        path = db_path or str(get_academic_brain_db_path())
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, claim_text, claim_type, confidence, validated, priority 
            FROM paper_claims 
            ORDER BY validated DESC, confidence DESC 
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "filename": row[0],
                "text": row[1],
                "type": row[2],
                "confidence": row[3],
                "validated": bool(row[4]),
                "priority": row[5]
            })
        
        conn.close()
        return results
    except Exception as e:
        print(f"[CLAIM QUERY ERROR] {e}")
        return []


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test with sample text
    sample = """
    Abstract: This study investigates the impact of social media bots on political discourse.
    We found that automated accounts significantly amplify misinformation by 340%.
    Our results demonstrate that bot activity increases during election periods.
    We hypothesize that platform intervention could reduce this effect.
    
    Introduction: The proliferation of bots on social media platforms has become a major concern.
    
    Results: Data analysis shows that bot-generated content reaches 10x more users.
    The findings indicate a strong correlation between bot activity and misinformation spread.
    
    Discussion: These results suggest the need for better detection mechanisms.
    A limitation of this study is the focus on English-language content only.
    Future work should examine multilingual bot behavior.
    
    Conclusion: We conclude that social media bots pose a significant threat to democratic discourse.
    """
    
    print("Testing 5-Star Claim Extraction Pipeline...")
    claims = extract_claims_pipeline(sample, validate=False)
    
    for i, claim in enumerate(claims, 1):
        print(f"\n{i}. [{claim['type']}] (conf: {claim['confidence']:.2f})")
        print(f"   {claim['text'][:80]}...")
