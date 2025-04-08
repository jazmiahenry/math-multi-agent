"""
Cache Manager for Financial Analysis Multi-Agent System

This module provides disk-based caching to avoid redundant computations.
"""

import os
import json
import hashlib
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages disk-based caching of agent outputs.
    
    This class handles saving and retrieving cached results to avoid
    redundant computation for identical tasks.
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, data: Any) -> str:
        """
        Generate a unique cache key for the given data.
        
        Args:
            data: Input data to hash
            
        Returns:
            str: Unique hash key
        """
        # Convert data to a consistent string representation
        if isinstance(data, dict) or isinstance(data, list):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        # Create hash
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            str: File path for the cache file
        """
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, data: Any) -> Optional[Any]:
        """
        Retrieve cached result for the given data.
        
        Args:
            data: Data to look up in cache
            
        Returns:
            Optional[Any]: Cached result or None if not found
        """
        key = self._get_cache_key(data)
        cache_path = self._get_cache_path(key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                # If cache file is corrupted, return None
                logger.warning(f"Cache read error: {e}")
                return None
        
        return None
    
    def set(self, data: Any, result: Any) -> None:
        """
        Store result in cache.
        
        Args:
            data: Original input data (used for key generation)
            result: Result to cache
        """
        key = self._get_cache_key(data)
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(result, f)
        except IOError as e:
            logger.error(f"Cache write error: {e}")
    
    def clear(self, data: Optional[Any] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            data: If provided, clear only this specific entry;
                 if None, clear all entries
        """
        if data is not None:
            # Clear specific entry
            key = self._get_cache_key(data)
            cache_path = self._get_cache_path(key)
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Cleared cache entry: {key}")
        else:
            # Clear all entries
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("Cleared all cache entries")