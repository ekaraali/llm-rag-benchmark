"""Regression tests for core behavior."""

from src.retrieval import split_words, decompose_query
from src.evaluate import evaluate_answer


def test_split_overlap():
    assert split_words('one two three four five', 3, 1) == ['one two three', 'three four five', 'five']


def test_invalid_chunk():
    import pytest
    with pytest.raises(ValueError):
        split_words('hi', 10, 10)


def test_evaluation():
    m = evaluate_answer('10 to 20 percent [chunking.md]', '10 to 20 percent',
                        ['chunking.md'], [{'doc': 'chunking.md'}])
    assert m['reference_phrase_match'] == 1
    assert m['retrieval_hit'] == 1
    assert m['citation_precision'] == 1


def test_decomposition():
    assert len(decompose_query('Compare chunking and retrieval?')) >= 1
