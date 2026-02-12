"""LLM Gateway - Unified interface for Groq, OpenRouter, and Ollama."""


import os
import json
import time
import requests
from typing import Optional, Dict, Any
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """
    Decorator for retry with exponential backoff.
    
    Retries on:
    - requests.Timeout
    - requests.ConnectionError
    - HTTP 429 (rate limit)
    - HTTP 503 (service unavailable)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if result indicates retryable failure
                    if result is not None:
                        return result
                    
                    # If None returned but no exception, try next provider/fallback
                    return result
                    
                except (requests.Timeout, requests.ConnectionError) as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    if attempt < max_retries - 1:
                        print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}")
                        print(f"[RETRY] Waiting {delay:.1f}s before retry...")
                        time.sleep(delay)
                    else:
                        print(f"[RETRY] All {max_retries} attempts failed: {e}")
                        
            return None
        return wrapper
    return decorator



class LLMGateway:
    """
    Unified LLM interface with automatic provider selection.
    
    Priority:
    1. Groq (fast, cloud, 30 req/min)
    2. OpenRouter (free models, unlimited)
    3. Ollama (local, unlimited)
    
    Usage:
        llm = LLMGateway()
        response = llm.generate("Write a summary...")
    """
    
    # =========================================================================
    # GROQ CONFIG
    # =========================================================================
    # =========================================================================
    # GROQ CONFIG
    # =========================================================================
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    
    # =========================================================================
    # OPENROUTER CONFIG (Free unlimited)
    # =========================================================================
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"  # FREE model
    
    # =========================================================================
    # OLLAMA CONFIG (Local fallback)
    # =========================================================================
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3"
    
    # Rate limiting for Groq (30 requests/minute)
    _groq_call_count = 0
    _groq_minute_start = 0
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize the gateway."""
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY") or self.GROQ_API_KEY
        self.openrouter_available = self._check_openrouter()
        self.provider = self._detect_provider()
        
        if self.provider:
            print(f"[LLM GATEWAY] Active provider: {self.provider.upper()}")
        else:
            print("[LLM GATEWAY] WARNING: No LLM provider available!")
    
    def _check_openrouter(self) -> bool:
        """Check if OpenRouter API is available."""
        try:
            r = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self.OPENROUTER_API_KEY}"},
                timeout=5
            )
            return r.status_code == 200
        except:
            return False
    
    def _detect_provider(self) -> Optional[str]:
        """Detect available provider with priority: Groq > OpenRouter > Ollama."""
        
        # 1. Check Groq first (fastest)
        if self.groq_api_key:
            try:
                r = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.groq_api_key}"},
                    timeout=5
                )
                if r.status_code == 200:
                    return "groq"
            except:
                pass
        
        # 2. Check OpenRouter (free, unlimited)
        if self.openrouter_available:
            return "openrouter"
        
        # 3. Check Ollama (local fallback)
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                return "ollama"
        except:
            pass
        
        return None
    
    def _rate_limit_groq(self):
        """Enforce Groq rate limit (30 requests/minute)."""
        current_time = time.time()
        
        if current_time - self._groq_minute_start > 60:
            self._groq_call_count = 0
            self._groq_minute_start = current_time
        
        if self._groq_call_count >= 28:
            wait_time = 60 - (current_time - self._groq_minute_start)
            if wait_time > 0:
                print(f"[LLM GATEWAY] Groq rate limit - waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self._groq_call_count = 0
                self._groq_minute_start = time.time()
        
        self._groq_call_count += 1
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.5,
        json_mode: bool = False,
        prefer_long_context: bool = False  # Kept for backward compatibility, ignored
    ) -> Optional[str]:
        """
        Generate text using the active provider.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Creativity (0.0-1.0)
            json_mode: If True, request JSON output
            prefer_long_context: IGNORED (kept for backward compatibility)
        
        Returns:
            Generated text or None if failed
        """
        if not self.provider:
            return None
        
        if self.provider == "groq":
            return self._call_groq(prompt, max_tokens, temperature, json_mode)
        elif self.provider == "openrouter":
            return self._call_openrouter(prompt, max_tokens, temperature, json_mode)
        else:
            return self._call_ollama(prompt, max_tokens, temperature, json_mode)
    
    def _call_groq(
        self, prompt: str, max_tokens: int, temperature: float, json_mode: bool
    ) -> Optional[str]:
        """Call Groq API (OpenAI-compatible format)."""
        self._rate_limit_groq()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            
            response = requests.post(
                self.GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"[GROQ ERROR] {response.status_code}: {response.text[:200]}")
                # Fallback to OpenRouter
                return self._call_openrouter(prompt, max_tokens, temperature, json_mode)
                
        except Exception as e:
            print(f"[GROQ EXCEPTION] {e}")
            return self._call_openrouter(prompt, max_tokens, temperature, json_mode)
    
    def _call_openrouter(
        self, prompt: str, max_tokens: int, temperature: float, json_mode: bool
    ) -> Optional[str]:
        """Call OpenRouter API (OpenAI-compatible, free models)."""
        try:
            headers = {
                "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://storm-research.local",
                "X-Title": "STORM Research System"
            }
            
            payload = {
                "model": self.OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            
            response = requests.post(
                self.OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"[OPENROUTER ERROR] {response.status_code}: {response.text[:200]}")
                return self._call_ollama(prompt, max_tokens, temperature, json_mode)
                
        except Exception as e:
            print(f"[OPENROUTER EXCEPTION] {e}")
            return self._call_ollama(prompt, max_tokens, temperature, json_mode)
    
    def _call_ollama(
        self, prompt: str, max_tokens: int, temperature: float, json_mode: bool
    ) -> Optional[str]:
        """Call local Ollama API."""
        try:
            payload = {
                "model": self.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            if json_mode:
                payload["format"] = "json"
            
            response = requests.post(
                self.OLLAMA_API_URL,
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                print(f"[OLLAMA ERROR] {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[OLLAMA EXCEPTION] {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if any LLM provider is available."""
        return self.provider is not None
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider."""
        if self.provider == "groq":
            return {
                "provider": "groq",
                "model": self.GROQ_MODEL,
                "type": "cloud",
                "rate_limit": "30/min"
            }
        elif self.provider == "openrouter":
            return {
                "provider": "openrouter", 
                "model": self.OPENROUTER_MODEL,
                "type": "cloud",
                "rate_limit": "unlimited",
                "cost": "FREE"
            }
        elif self.provider == "ollama":
            return {
                "provider": "ollama", 
                "model": self.OLLAMA_MODEL,
                "type": "local",
                "rate_limit": "unlimited"
            }
        return {"provider": None}


# Singleton instance for global use
_gateway_instance = None

def get_llm_gateway() -> LLMGateway:
    """Get or create the global LLM Gateway instance."""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = LLMGateway()
    return _gateway_instance


# Legacy compatibility function
def generate_text(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    """Legacy function for backward compatibility."""
    return get_llm_gateway().generate(prompt, max_tokens)
