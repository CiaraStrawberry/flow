import torch
import hashlib
import os
import json
from pathlib import Path
import shutil

class LatentCache:
    def __init__(self, max_cache_size=None, use_disk=False, cache_dir="cache", clear_existing=False):
        """
        Initialize the latent cache with options for memory or disk storage.
        
        Args:
            max_cache_size: Maximum number of items to store in cache (None for unlimited)
            use_disk: If True, store cache on disk instead of in memory
            cache_dir: Directory to store cached files when use_disk=True
            clear_existing: If True, clear any existing cache files in cache_dir
        """
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.use_disk = use_disk
        
        # Set up disk cache if needed
        if self.use_disk:
            self.cache_dir = Path(cache_dir)
            if clear_existing and self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            
            # Create subdirectories to avoid too many files in one directory
            for i in range(16):
                subdir = self.cache_dir / f"{i:x}"
                subdir.mkdir(exist_ok=True)
                
            # Keep an index file to track metadata
            self.index_path = self.cache_dir / "index.json"
            if self.index_path.exists():
                with open(self.index_path, 'r') as f:
                    self.index = json.load(f)
            else:
                self.index = {"keys": [], "count": 0}
                self._save_index()
        
    def hash_content(self, content):
        """Generate a hash from content for content-based caching"""
        
        if isinstance(content, torch.Tensor):
            # For tensors, use a subset of values to create a hash
            # Using full tensor might be too slow
            if content.numel() > 1000:
                # Sample some values from the tensor for hashing
                indices = torch.linspace(0, content.numel()-1, 1000, dtype=torch.long)
                flat_content = content.flatten().cpu().detach()
                samples = flat_content[indices].numpy().tobytes()
                return hashlib.md5(samples).hexdigest()
            else:
                # Small tensor, use all values
                return hashlib.md5(content.cpu().detach().numpy().tobytes()).hexdigest()
        elif isinstance(content, str):
            # For text content
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        else:
            # For other types, convert to string first
            return hashlib.md5(str(content).encode('utf-8')).hexdigest()
    
    def _save_index(self):
        """Save the index file with metadata (only used when use_disk=True)"""
        if self.use_disk:
            with open(self.index_path, 'w') as f:
                json.dump(self.index, f)
    
    def _get_file_path(self, key):
        """Get the file path for a key (only used when use_disk=True)"""
        # Use first character of hash to determine subdirectory
        subdir = key[:1]
        return self.cache_dir / subdir / f"{key}.pt"
    
    def get(self, key):
        """Get a latent from cache if it exists"""
        if self.use_disk:
            file_path = self._get_file_path(key)
            if file_path.exists():
                self.cache_hits += 1
                return torch.load(file_path)
            self.cache_misses += 1
            return None
        else:
            # In-memory cache
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key]
            self.cache_misses += 1
            return None
    
    def put(self, key, value):
        """Add a latent to cache, potentially evicting old items if at capacity"""
        if self.use_disk:
            # Save to disk
            if self.max_cache_size and self.index["count"] >= self.max_cache_size:
                # Simple LRU implementation - remove oldest item
                if self.index["keys"]:
                    oldest_key = self.index["keys"][0]
                    self.index["keys"].remove(oldest_key)
                    oldest_file = self._get_file_path(oldest_key)
                    if oldest_file.exists():
                        oldest_file.unlink()
                    self.index["count"] -= 1
            
            # Save the new item
            file_path = self._get_file_path(key)
            torch.save(value, file_path)
            
            # Update index
            if key not in self.index["keys"]:
                self.index["keys"].append(key)
                self.index["count"] += 1
                
            # Save index occasionally (every 100 items)
            if self.index["count"] % 100 == 0:
                self._save_index()
        else:
            # In-memory cache
            if self.max_cache_size and len(self.cache) >= self.max_cache_size:
                # Simple LRU implementation - remove oldest item
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = value
    
    def save_to_disk(self, save_dir):
        """
        Save the entire in-memory cache to disk
        Only relevant when use_disk=False
        """
        if self.use_disk:
            print("Cache is already on disk, no need to save")
            return
            
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Create structure similar to disk cache
        for i in range(16):
            subdir = save_path / f"{i:x}"
            subdir.mkdir(exist_ok=True)
        
        # Save all cache items
        index = {"keys": [], "count": 0}
        for key, value in self.cache.items():
            # Use first character of key to determine subdirectory
            subdir = key[:1]
            file_path = save_path / subdir / f"{key}.pt"
            torch.save(value, file_path)
            index["keys"].append(key)
            index["count"] += 1
        
        # Save index
        with open(save_path / "index.json", 'w') as f:
            json.dump(index, f)
        
        print(f"Saved {len(self.cache)} items to disk at {save_dir}")
    
    def load_from_disk(self, load_dir):
        """
        Load the cache from disk into memory
        Only relevant when use_disk=False
        """
        if self.use_disk:
            print("Cache is already on disk, no need to load")
            return
            
        load_path = Path(load_dir)
        if not load_path.exists():
            print(f"Cache directory {load_dir} does not exist")
            return
        
        # Load index
        index_path = load_path / "index.json"
        if not index_path.exists():
            print(f"Index file not found in {load_dir}")
            return
            
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        # Clear existing cache
        self.cache.clear()
        
        # Load each item
        for key in index["keys"]:
            subdir = key[:1]
            file_path = load_path / subdir / f"{key}.pt"
            if file_path.exists():
                self.cache[key] = torch.load(file_path)
        
        print(f"Loaded {len(self.cache)} items from disk at {load_dir}")
    
    def get_stats(self):
        """Return cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        if self.use_disk:
            size = self.index["count"] if hasattr(self, 'index') else 0
        else:
            size = len(self.cache)
            
        return {
            "size": size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
            "storage": "disk" if self.use_disk else "memory"
        }