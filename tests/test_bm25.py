"""BM25 scores checked against values computed by hand from the formula."""

import math
import pytest

from src.bm25 import BM25, tokenize


def test_tokenize_keeps_hyphenated_terms():
    assert tokenize('Top-p and BM25, the model\'s Scores!') == ['top-p', 'and', 'bm25', 'the', "model's", 'scores']


def test_scores_match_hand_calculation():
    bm = BM25(['a b', 'a c c'], k1=1.5, b=0.75)
    # N = 2, avgdl = 2.5; "c" occurs twice in passage 2 (length 3) and nowhere else
    idf_c = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    norm = 1.5 * (1 - 0.75 + 0.75 * 3 / 2.5)
    expected = idf_c * 2 * (1.5 + 1) / (2 + norm)
    scores = bm.scores('c')
    assert scores[0] == 0.0
    assert scores[1] == pytest.approx(expected)


def test_common_terms_get_lower_idf_and_unknown_terms_score_zero():
    bm = BM25(['a b', 'a c'])
    assert bm.idf['a'] < bm.idf['b']
    assert bm.scores('zzz') == [0.0, 0.0]


def test_invalid_parameters():
    with pytest.raises(ValueError):
        BM25([])
    with pytest.raises(ValueError):
        BM25(['a'], b=1.5)
