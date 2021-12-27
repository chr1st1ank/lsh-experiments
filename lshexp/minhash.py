import collections
from typing import List

import numpy as np
# import murmurhash.mrmr
import xxhash
from numpy.random import default_rng


def hash32(data):
    """A 32-bit hash function based on SHA1."""
    # return binascii.crc32(data) & 0xffffffff
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
    gen = np.random.RandomState(random_seed)
    A = gen.randint(1, _mersenne_prime, size=n_hashes, dtype='uint32')
    B = gen.randint(0, _mersenne_prime, size=n_hashes, dtype='uint32')

    def calc_minhashes(shingles: List[str]) -> np.array:
        hashes = np.array(
            [hash32(s.encode("utf-8")) for s in shingles], dtype=np.uint32
        )
        hashes = hashes.repeat(A.shape[0]).reshape(hashes.shape[0], A.shape[0])
        hashes = (A * hashes + B) % _mersenne_prime
        minhashes = np.min(hashes, axis=0)
        return minhashes

    return calc_minhashes


class LSH:
    def __init__(self, n_hashes, bands=25):
        self.n_hashes = n_hashes
        self.n_bands = bands
        self.rows_per_band = n_hashes // bands
        self.bands = [{} for _ in range(self.n_bands)]
        self.values = []
        self._hashfunc = hash32
        # self._hashfunc = murmurhash.mrmr.hash

    def insert(self, minhashes: np.array, value):
        value_index = len(self.values)
        self.values.append(value)
        for band_number in range(self.n_bands):
            start_index = band_number * self.rows_per_band
            h = self._hash(minhashes[start_index : start_index + self.rows_per_band])
            self.bands[band_number].setdefault(h, set()).update([value_index])

    # TODO: def remove(self, minhashes)

    def query(self, minhashes: np.array):
        candidates = collections.Counter()
        for band_number in range(self.n_bands):
            start_index = band_number * self.rows_per_band
            h = self._hash(minhashes[start_index : start_index + self.rows_per_band])
            candidates.update(self.bands[band_number].get(h, set()))
        return set(self.values[c] for c in candidates)

    def best_hits(self, minhashes: np.array):
        candidates = collections.Counter()
        for band_number in range(self.n_bands):
            start_index = band_number * self.rows_per_band
            h = self._hash(minhashes[start_index : start_index + self.rows_per_band])
            candidates.update(self.bands[band_number].get(h, set()))
        return [self.values[key] for key, count in candidates.most_common()]

    def _hash(self, arr: np.array):
        """Merge multiple hashes together to one hash per band"""
        return self._hashfunc(bytes(arr.data))
