"""Evaluation regression check: comparison logic, token metrics, and the committed baselines."""

import csv
import json
from pathlib import Path

import pytest

from experiments.check_regression import compare, run, BASELINES
from src.evaluate import token_f1, reference_token_recall

ROOT = Path(__file__).resolve().parents[1]


def test_compare_flags_drops_beyond_tolerance_only():
    base = {'mrr': 0.90, 'recall': 0.80}
    failures, improvements = compare(base, {'mrr': 0.895, 'recall': 0.70}, tolerance=0.01)
    assert failures == ['recall: 0.8000 -> 0.7000 (-0.1000)']
    assert improvements == []


def test_compare_reports_improvements_and_missing_metrics():
    failures, improvements = compare({'mrr': 0.5, 'ndcg': 0.5}, {'mrr': 0.9}, tolerance=0.01)
    assert failures == ['ndcg: missing from current results']
    assert improvements == ['mrr: 0.5000 -> 0.9000 (+0.4000)']


def test_token_metrics_match_hand_calculation():
    # "the" is dropped; answer tokens: overlap is 10 to 20 percent (6), reference: 10 to 20 percent (4)
    assert token_f1('The overlap is 10 to 20 percent.', '10 to 20 percent') == pytest.approx(2 * (4 / 6) * 1 / (4 / 6 + 1))
    assert reference_token_recall('The overlap is 10 to 20 percent.', '10 to 20 percent') == 1.0
    assert token_f1('Insufficient evidence.', '10 to 20 percent') == 0.0
    assert reference_token_recall('20 percent', '10 to 20 percent') == 0.5


def test_frozen_baselines_reproduce_published_v01_numbers():
    """The regression baselines must agree with the numbers published in reports/v0.1."""
    frozen = json.loads(BASELINES.read_text())['frozen_answers']['metrics']
    published = list(csv.DictReader((ROOT / 'reports/v0.1/benchmark_summary.csv').open()))
    for row in published:
        base = frozen[f"reports/v0.1/human_validated_reviews_k{row['top_k_run']}.csv"][row['pipeline']]
        assert base['reference_phrase_match'] == pytest.approx(float(row['reference_phrase_match']))
        assert base['citation_precision'] == pytest.approx(float(row['citation_precision_proxy']))


def test_committed_baselines_pass():
    assert run(update=False)
