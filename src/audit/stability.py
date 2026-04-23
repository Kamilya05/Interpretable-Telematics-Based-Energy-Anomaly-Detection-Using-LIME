from __future__ import annotations
from typing import Iterable, Set

def topk_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa: Set[str] = set(a)
    sb: Set[str] = set(b)

    if not sa and not sb:
        return 1.0
    
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0

def sign_consistency(signs: list[int]) -> float:
    if not signs:
        return 0.0
    
    non_zero = [s for s in signs if s != 0]

    if not non_zero:
        return 1.0
    
    pos = sum(1 for s in non_zero if s > 0)
    neg = sum(1 for s in non_zero if s < 0)
    return max(pos, neg) / len(non_zero)
