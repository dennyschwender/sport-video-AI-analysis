import hashlib
import json
from typing import Dict, Any, Optional
from pathlib import Path
import diskcache


class LLMCache:
    """Disk-based cache for LLM responses."""
    
    def __init__(self, cache_dir: str = ".cache", enabled: bool = True):
        self.enabled = enabled
        if self.enabled:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.cache = diskcache.Cache(str(cache_path))
        else:
            self.cache = None
    
    def _generate_key(self, backend: str, model: str, text: str) -> str:
        """Generate cache key from backend, model, and input text."""
        content = f"{backend}:{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, backend: str, model: str, text: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if available."""
        if not self.enabled or not self.cache:
            return None
        
        key = self._generate_key(backend, model, text)
        result = self.cache.get(key)
        return result
    
    def set(self, backend: str, model: str, text: str, response: Dict[str, Any], expire: Optional[int] = None):
        """Store response in cache."""
        if not self.enabled or not self.cache:
            return
        
        key = self._generate_key(backend, model, text)
        self.cache.set(key, response, expire=expire)
    
    def clear(self):
        """Clear all cached responses."""
        if self.enabled and self.cache:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or not self.cache:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self.cache),
            "volume": self.cache.volume()
        }
