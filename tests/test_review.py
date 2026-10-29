"""Regression tests for review behavior."""

import csv
from pathlib import Path
import pytest
from experiments.review_answers import prepare
from experiments.summarize_review import summarize, render


def test_review_template_and_report(tmp_path):
    src = tmp_path / 'per_question.csv'
    with src.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pipeline', 'question_id', 'question', 'answer', 'reference_answer'])
        w.writeheader()
        w.writerow(dict(pipeline='direct', question_id=1, question='Q', answer='A', reference_answer='R'))
    review = prepare(src, tmp_path / 'manual_review.csv')
    counts = summarize(review)
    assert counts['direct']['unreviewed'] == 1
    assert 'N/A' in render(counts)
    rows = list(csv.DictReader(review.open()))
    rows[0]['correctness'] = 'correct'
    with review.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    assert counts['direct']['unreviewed'] == 1
    assert '100.0%' in render(summarize(review))
    rows[0]['correctness'] = 'madeup'
    with review.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError):
        summarize(review)
