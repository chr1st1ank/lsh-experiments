import collections
import hashlib
import itertools
import random
import struct
import sys
from typing import Iterable, List, Set

import numpy as np
from numpy.random import default_rng
import murmurhash.mrmr
import xxhash


def hash32(data):
    """A 32-bit hash function based on SHA1."""
    # return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]
    # return struct.unpack("<Q", hashlib.sha1(data).digest()[:8])[0]
    # return int(hashlib.sha1(data).hexdigest()[:16], 16)
    # return murmurhash.mrmr.hash(data)
    return int(xxhash.xxh32(data).hexdigest(), 16)
    return xxhash.xxh32(data).intdigest()
    # return struct.unpack("<I", xxhash.xxh32(data).digest()[:4])[0]


def make_minhash_generator(n_hashes=100, random_seed=42):
    # _mersenne_prime = np.uint64((1 << 61) - 1)
    _mersenne_prime = np.uint32((1 << 32) - 1)
    _max_hash = np.uint32((1 << 32) - 1)
    # random.seed(random_seed)
    # params = [
    #     (random.randint(0, _mersenne_prime), random.randint(0, _mersenne_prime))
    #     for _ in range(n_hashes)
    # ]
    gen = np.random.RandomState(random_seed)
    params = [
        (
            gen.randint(1, _mersenne_prime, dtype=np.uint32),
            gen.randint(0, _mersenne_prime, dtype=np.uint32),
        )
        for _ in range(n_hashes)
    ]

    def calc_minhashes(shingles: List[str]) -> np.array:
        hashes = np.array(
            [hash32(s.encode("utf-8")) for s in shingles], dtype=np.uint32
        )
        hashes = np.array([(a * hashes + b) % _mersenne_prime for a, b in params])
        minhashes = np.min(hashes, axis=1)
        return minhashes

    return calc_minhashes


class LSH:
    def __init__(self, n_hashes, bands=25):
        self.n_hashes = n_hashes
        self.n_bands = bands
        self.rows_per_band = n_hashes // bands
        self.bands = [{} for _ in range(self.n_bands)]
        self.values = []
        self._hashfunc = murmurhash.mrmr.hash

    def insert(self, minhashes: np.array, value):
        value_index = len(self.values)
        self.values.append(value)
        for band_number in range(self.n_bands):
            start_index = band_number * self.rows_per_band
            h = self._hash(minhashes[start_index : start_index + self.rows_per_band])
            self.bands[band_number].setdefault(h, set()).update([value_index])

    def query(self, minhashes: np.array):
        candidates = collections.Counter()
        for band_number in range(self.n_bands):
            start_index = band_number * self.rows_per_band
            h = self._hash(minhashes[start_index : start_index + self.rows_per_band])
            candidates.update(self.bands[band_number].get(h, set()))
        return set(self.values[c] for c in candidates)

    def _hash(self, arr: np.array):
        """Merge multiple hashes together to one hash per band"""
        return self._hashfunc(bytes(arr.data))
