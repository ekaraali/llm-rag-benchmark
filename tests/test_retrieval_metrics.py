"""Retrieval metrics on small hand-checked rankings."""

import math
import pytest

from src import retrieval_metrics as rm


def test_recall_hit_and_reciprocal_rank():
    ranked = ['x', 'a', 'y', 'b']
    assert rm.recall_at_k(ranked, {'a', 'b'}, 2) == 0.5
    assert rm.recall_at_k(ranked, {'a', 'b'}, 4) == 1.0
    assert rm.hit_at_k(ranked, {'b'}, 3) == 0
    assert rm.reciprocal_rank(ranked, {'a', 'b'}) == 0.5
    assert rm.reciprocal_rank(ranked, {'b'}, k=3) == 0.0


def test_ndcg():
    assert rm.ndcg_at_k(['a', 'b'], {'a'}, 2) == 1.0
    # one relevant item at rank 2: (1/log2 3) / (1/log2 2)
    assert rm.ndcg_at_k(['x', 'a'], {'a'}, 2) == pytest.approx(1 / math.log2(3))


def test_evidence_coverage_needs_every_group():
    groups = [{'c1'}, {'c5', 'c6'}]  # AND across groups, OR inside a group
    assert rm.evidence_coverage_at_k(['c1', 'c6', 'c9'], groups, 3) == 1
    assert rm.evidence_coverage_at_k(['c1', 'c9', 'c6'], groups, 2) == 0


def test_metrics_reject_empty_relevance():
    with pytest.raises(ValueError):
        rm.recall_at_k(['a'], set(), 1)


def test_dedupe_docs_keeps_best_rank():
    chunks = [{'doc': 'b.md'}, {'doc': 'a.md'}, {'doc': 'b.md'}]
    assert rm.dedupe_docs(chunks) == ['b.md', 'a.md']
