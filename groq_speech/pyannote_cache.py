"""
Global Pyannote.audio Pipeline Cache Manager.

This module provides a singleton pattern for caching Pyannote.audio pipelines
across multiple runs, significantly reducing the time needed to load models.
"""

import os
import time
import threading
from typing import Optional, Dict, Any
import logging

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    Pipeline = None

logger = logging.getLogger(__name__)


class PyannotePipelineCache:
    """
    Global singleton cache for Pyannote.audio pipelines.
    
    This class ensures that Pyannote.audio models are loaded only once
    and reused across multiple runs, dramatically improving performance.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._pipelines: Dict[str, Pipeline] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}
        self._last_used: Dict[str, float] = {}
        self._cache_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_load_time': 0.0,
            'models_loaded': 0
        }
        
        logger.info("🚀 PyannotePipelineCache initialized - Global model caching enabled")
    
    def get_pipeline(
        self, 
        model_name: str = "pyannote/speaker-diarization-3.1",
        use_auth_token: Optional[str] = None,
        force_reload: bool = False
    ) -> Optional[Pipeline]:
        """
        Get a cached Pyannote.audio pipeline or load it if not cached.
        
        Args:
            model_name: Name of the Pyannote.audio model
            use_auth_token: HuggingFace token for model access
            force_reload: Force reload even if cached
            
        Returns:
            Cached or newly loaded Pipeline instance
        """
        if not PYANNOTE_AVAILABLE:
            logger.warning("Pyannote.audio not available - cannot load pipeline")
            return None
        
        self._cache_stats['total_requests'] += 1
        
        # Check if we have a cached pipeline
        if not force_reload and model_name in self._pipelines:
            pipeline = self._pipelines[model_name]
            self._last_used[model_name] = time.time()
            self._cache_stats['cache_hits'] += 1
            
            logger.debug(f"✅ Cache HIT for {model_name} (loaded {self._model_info[model_name]['load_time']:.2f}s ago)")
            return pipeline
        
        # Cache miss - need to load the model
        self._cache_stats['cache_misses'] += 1
        logger.info(f"🔄 Cache MISS for {model_name} - Loading model...")
        
        try:
            start_time = time.time()
            
            # Load the pipeline
            pipeline = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token)
            
            load_time = time.time() - start_time
            self._cache_stats['total_load_time'] += load_time
            self._cache_stats['models_loaded'] += 1
            
            # Cache the pipeline
            self._pipelines[model_name] = pipeline
            self._last_used[model_name] = time.time()
            self._model_info[model_name] = {
                'load_time': load_time,
                'loaded_at': time.time(),
                'model_name': model_name
            }
            
            logger.info(f"✅ Model {model_name} loaded and cached in {load_time:.2f}s")
            logger.info(f"💾 Pipeline cached - Future requests will be instant!")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ Failed to load Pyannote.audio model {model_name}: {e}")
            return None
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear cached pipelines."""
        if model_name:
            if model_name in self._pipelines:
                del self._pipelines[model_name]
                del self._model_info[model_name]
                del self._last_used[model_name]
                logger.info(f"🗑️  Cleared cache for {model_name}")
        else:
            self._pipelines.clear()
            self._model_info.clear()
            self._last_used.clear()
            logger.info("🗑️  Cleared all cached pipelines")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        stats = self._cache_stats.copy()
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / max(stats['total_requests'], 1) * 100
        )
        stats['avg_load_time'] = (
            stats['total_load_time'] / max(stats['models_loaded'], 1)
        )
        stats['cached_models'] = list(self._pipelines.keys())
        stats['cache_size'] = len(self._pipelines)
        
        return stats
    
    def print_cache_stats(self):
        """Print cache performance statistics."""
        stats = self.get_cache_stats()
        
        print("\n📊 Pyannote.audio Pipeline Cache Statistics:")
        print("=" * 50)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"Cache Misses: {stats['cache_misses']}")
        print(f"Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")
        print(f"Models Loaded: {stats['models_loaded']}")
        print(f"Average Load Time: {stats['avg_load_time']:.2f}s")
        print(f"Currently Cached: {stats['cache_size']} models")
        
        if stats['cached_models']:
            print("\nCached Models:")
            for model in stats['cached_models']:
                info = self._model_info[model]
                age = time.time() - info['loaded_at']
                print(f"  • {model} (loaded {age:.1f}s ago, took {info['load_time']:.2f}s)")


# Global instance
_pipeline_cache = None

def get_pipeline_cache() -> PyannotePipelineCache:
    """Get the global pipeline cache instance."""
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = PyannotePipelineCache()
    return _pipeline_cache

def get_cached_pipeline(
    model_name: str = "pyannote/speaker-diarization-3.1",
    use_auth_token: Optional[str] = None,
    force_reload: bool = False
) -> Optional[Pipeline]:
    """Get a cached pipeline from the global cache."""
    cache = get_pipeline_cache()
    return cache.get_pipeline(model_name, use_auth_token, force_reload)

def clear_pipeline_cache(model_name: Optional[str] = None):
    """Clear the global pipeline cache."""
    cache = get_pipeline_cache()
    cache.clear_cache(model_name)

def print_pipeline_cache_stats():
    """Print global pipeline cache statistics."""
    cache = get_pipeline_cache()
    cache.print_cache_stats()
