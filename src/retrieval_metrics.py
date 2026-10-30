"""Retrieval quality metrics with binary relevance.

All functions take `ranked`, the retrieved item IDs in rank order, and
`relevant`, the set of IDs labelled relevant. They work for chunk IDs and for
document IDs alike. They are undefined for unanswerable questions (no relevant
items); callers should skip those rather than score them as zero.
"""

import math


def _check(relevant, k=None):
    if not relevant:
        raise ValueError('Retrieval metrics need at least one relevant item')
    if k is not None and k < 1:
        raise ValueError('k must be >= 1')


def recall_at_k(ranked, relevant, k):
    """Share of relevant items that appear in the top k."""
    _check(relevant, k)
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def hit_at_k(ranked, relevant, k):
    """1 if any relevant item appears in the top k, else 0."""
    _check(relevant, k)
    return int(bool(set(ranked[:k]) & set(relevant)))


def reciprocal_rank(ranked, relevant, k=None):
    """1 / rank of the first relevant item (0 if none within the cutoff).

    Averaging this over questions gives Mean Reciprocal Rank (MRR).
    """
    _check(relevant, k)
    for rank, item in enumerate(ranked[:k] if k else ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    """Normalised discounted cumulative gain at k with binary gains.

    DCG = sum over relevant items at rank i of 1 / log2(i + 1); it is divided by
    the DCG of an ideal ranking that puts all relevant items first.
    """
    _check(relevant, k)
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def evidence_coverage_at_k(ranked, groups, k):
    """1 if every evidence group has at least one of its items in the top k.

    `groups` is a list of sets (AND across groups, OR within a group), as
    produced by src.schema.relevant_chunks. This is the strict multi-hop
    criterion: the answer is fully supported only when all groups are covered.
    """
    if not groups or any(not g for g in groups):
        raise ValueError('Evidence coverage needs non-empty evidence groups')
    top = set(ranked[:k])
    return int(all(top & g for g in groups))


def dedupe_docs(ranked_chunks):
    """Document IDs in order of their best-ranked chunk (for document-level metrics)."""
    seen, docs = set(), []
    for c in ranked_chunks:
        if c['doc'] not in seen:
            seen.add(c['doc'])
            docs.append(c['doc'])
    return docs
