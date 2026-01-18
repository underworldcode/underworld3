"""
DMInterpolation Caching System

Caches DMInterpolation structures to avoid repeated expensive setup operations.

Key insight: DMInterpolation structure depends on:
- Coordinates (where to interpolate)
- DOF count (how many variables)

Does NOT depend on variable values! Those are fetched fresh each time.

Control caching via mesh flag:
    mesh.enable_dminterpolation_cache = False  # Disable for this mesh

Caching is enabled by default for all meshes.
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional
import underworld3 as uw


class DMInterpolationCache:
    """
    Per-mesh cache for DMInterpolation structures.

    Cache key: (coord_hash, dofcount)
    Cache value: CachedDMInterpolationInfo object (Cython wrapper)

    Automatically tracks hits, misses, and invalidations.
    """

    def __init__(self, mesh, name: str = "default"):
        self.mesh = mesh
        self.name = name
        self._cache: Dict[Tuple[int, int], object] = {}  # Stores CachedDMInterpolationInfo

        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'time_saved': 0.0,
            'time_computing': 0.0,
        }

    def _is_enabled(self) -> bool:
        """Check if caching is enabled."""
        # Check mesh flag (default: True)
        return getattr(self.mesh, 'enable_dminterpolation_cache', True)

    def get_structure(self, coords: np.ndarray, dofcount: int):
        """
        Get cached CachedDMInterpolationInfo or None.

        Parameters
        ----------
        coords : ndarray
            Evaluation coordinates
        dofcount : int
            Total DOF count for current variables

        Returns
        -------
        cached_info : CachedDMInterpolationInfo or None
            Cached structure object, or None if not cached or disabled
        """
        if not self._is_enabled():
            return None  # Caching disabled

        # Compute cache key
        coords_hash = self._hash_coords(coords)
        key = (coords_hash, dofcount)

        # Check cache
        if key in self._cache:
            # CACHE HIT!
            self._stats['hits'] += 1
            cached_info = self._cache[key]

            return cached_info
        else:
            # CACHE MISS
            self._stats['misses'] += 1
            return None

    def store_structure(self, coords: np.ndarray, dofcount: int, cached_info):
        """
        Store CachedDMInterpolationInfo in cache.

        Parameters
        ----------
        coords : ndarray
            Evaluation coordinates
        dofcount : int
            Total DOF count
        cached_info : CachedDMInterpolationInfo
            Cython wrapper object containing DMInterpolation structure
        """
        if not self._is_enabled():
            return  # Don't cache if disabled

        coords_hash = self._hash_coords(coords)
        key = (coords_hash, dofcount)

        self._cache[key] = cached_info  # Python GC keeps it alive

    def _hash_coords(self, coords: np.ndarray) -> int:
        """
        Fast coordinate hashing for cache lookups.

        Uses xxhash for speed (already used in old caching system).
        """
        import xxhash
        xxh = xxhash.xxh64()
        xxh.update(np.ascontiguousarray(coords))
        return xxh.intdigest()

    def invalidate_all(self, reason: str = "manual"):
        """
        Clear entire cache and destroy all DMInterpolation structures.

        IMPORTANT: Must be called from Cython to properly destroy C structures!
        This method only clears the Python-side tracking.
        """
        n_entries = len(self._cache)
        if n_entries > 0:
            self._stats['invalidations'] += 1

        self._cache.clear()

    def invalidate_coords(self, coords: np.ndarray):
        """Invalidate all cache entries for specific coordinates."""
        coords_hash = self._hash_coords(coords)

        # Remove all entries with this coord hash (different dofcounts)
        keys_to_remove = [k for k in self._cache.keys() if k[0] == coords_hash]

        for key in keys_to_remove:
            del self._cache[key]
            self._stats['invalidations'] += 1

    def get_stats(self) -> dict:
        """
        Get cache performance statistics.

        Returns
        -------
        dict with metrics:
            - requests: Total get() calls
            - hits: Cache hits
            - misses: Cache misses
            - hit_rate: Percentage of hits
            - entries: Current cache size
            - invalidations: Total invalidation events
        """
        total_requests = self._stats['hits'] + self._stats['misses']

        return {
            'requests': total_requests,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate': self._stats['hits'] / total_requests if total_requests > 0 else 0.0,
            'entries': len(self._cache),
            'invalidations': self._stats['invalidations'],
        }

    def print_stats(self):
        """Print cache statistics (user-facing)."""
        stats = self.get_stats()

        print(f"\n{'=' * 60}")
        print(f"DMInterpolation Cache Statistics: {self.name}")
        print(f"{'=' * 60}")
        print(f"Requests:      {stats['requests']:>10,}")
        print(f"  Hits:        {stats['hits']:>10,}  ({stats['hit_rate']*100:>5.1f}%)")
        print(f"  Misses:      {stats['misses']:>10,}")
        print(f"Cache entries: {stats['entries']:>10,}")
        print(f"Invalidations: {stats['invalidations']:>10,}")
        print(f"{'=' * 60}\n")

    def reset_stats(self):
        """Reset statistics (but keep cached entries)."""
        self._stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'time_saved': 0.0,
            'time_computing': 0.0,
        }
