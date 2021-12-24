from typing import Set, Iterable


def jaccard_similarity(s1: Iterable, s2: Iterable):
    if not isinstance(s1, set):
        s1 = set(s1)
    if not isinstance(s2, set):
        s2 = set(s2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def count_equal(l1: Iterable, l2: Iterable):
    return sum(a == b for a, b in zip(l1, l2))


import numpy as np

np.zeros
