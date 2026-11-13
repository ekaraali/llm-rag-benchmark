"""Agreement statistics checked against values computed by hand."""

import pytest

from src.agreement import (accuracy, cohen_kappa, confusion_matrix, macro_f1, cluster_bootstrap_ci)

CATS = ('incorrect', 'partial', 'correct')


def test_confusion_and_accuracy():
    a = ['correct', 'correct', 'incorrect', 'partial']
    b = ['correct', 'incorrect', 'incorrect', 'partial']
    assert confusion_matrix(a, b, CATS) == [[1, 0, 0], [0, 1, 0], [1, 0, 1]]
    assert accuracy(a, b) == 0.75


def test_unweighted_kappa_hand_value():
    # observed agreement 3/4, chance agreement 0.5*0.75 + 0.5*0.25 = 0.5 -> kappa 0.5
    a = ['correct', 'correct', 'incorrect', 'incorrect']
    b = ['correct', 'incorrect', 'incorrect', 'incorrect']
    assert cohen_kappa(a, b, CATS) == pytest.approx(0.5)


def test_weighted_kappa_counts_near_misses_less():
    a = ['incorrect', 'partial', 'correct']
    b = ['partial', 'partial', 'correct']
    assert cohen_kappa(a, b, CATS) == pytest.approx(0.5)
    # linear weights: observed 0.5/3, expected 0.3889 -> 1 - 0.1667/0.3889
    assert cohen_kappa(a, b, CATS, weights='linear') == pytest.approx(1 - (0.5 / 3) / (3.5 / 9))
    # a full miss (incorrect vs correct) costs more than a near miss
    far = ['correct', 'partial', 'correct']
    assert cohen_kappa(a, far, CATS, weights='linear') < cohen_kappa(a, b, CATS, weights='linear')


def test_perfect_and_degenerate_agreement():
    a = ['correct', 'partial', 'incorrect']
    assert cohen_kappa(a, a, CATS) == 1.0
    assert cohen_kappa(['correct'] * 3, ['correct'] * 3, CATS) == 1.0


def test_macro_f1_skips_absent_classes():
    # correct: tp1 fn1 fp0 -> 2/3; incorrect: tp1 fn0 fp1 -> 2/3; partial absent -> skipped
    assert macro_f1(['correct', 'correct', 'incorrect'], ['correct', 'incorrect', 'incorrect'], CATS) == pytest.approx(2 / 3)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        cohen_kappa(['correct'], ['maybe'], CATS)
    with pytest.raises(ValueError):
        cohen_kappa(['correct'], ['correct'], CATS, weights='quadratic')


def test_cluster_bootstrap_is_deterministic_and_brackets_the_estimate():
    clusters = {q: [q % 3 == 0, True] for q in range(12)}
    share = lambda items: sum(items) / len(items)  # noqa: E731
    lo, hi = cluster_bootstrap_ci(clusters, share, n_boot=500, seed=1)
    assert (lo, hi) == cluster_bootstrap_ci(clusters, share, n_boot=500, seed=1)
    point = share([x for v in clusters.values() for x in v])
    assert lo <= point <= hi
