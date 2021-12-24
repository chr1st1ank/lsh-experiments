import collections
from typing import Iterable, List, Set, Union

import numpy.random


def count_ngrams(s, n):
    """n-gram counts in s as mapping ngram->count"""
    padded = "$" * (n - 1) + s + "$" * (n - 1)
    return collections.Counter(padded[i : i + n] for i in range(len(padded) - n + 1))


def all_ngrams(s: Union[bytes, str], n):
    """Set of n-grams in s"""
    if isinstance(s, str):
        padded = "$" * (n - 1) + s + "$" * (n - 1)
    if isinstance(s, bytes):
        padded = b"$" * (n - 1) + s + b"$" * (n - 1)
    return set(padded[i : i + n] for i in range(len(padded) - n + 1))


def shingles(s: str) -> Set[str]:
    return all_ngrams(s, 3)


def set_permutations(s: Set, n_permutations: int) -> List[List[str]]:
    """Get n random permutations of the elements in the set s"""
    rng = numpy.random.default_rng()
    return [rng.permutation(s) for _ in range(n_permutations)]
