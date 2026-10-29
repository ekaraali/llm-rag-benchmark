"""Prepare a manually reviewed answer-correctness scorecard from per_question.csv."""
import argparse
import csv
from pathlib import Path

FIELDS = ['pipeline', 'question_id', 'question', 'reference_answer', 'answer',
          'correctness', 'notes']


def prepare(source: Path, output: Path):
    """Export a blank review sheet while retaining original questions/answers."""
    with source.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Input CSV contains no rows')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            # Never infer correctness from the weak phrase-match metric.
            w.writerow({**{k: row.get(k, '') for k in FIELDS}, 'correctness': '', 'notes': ''})
    return output


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('per_question', type=Path)
    a.add_argument('--output', type=Path)
    args = a.parse_args()
    output = args.output or args.per_question.with_name('manual_review.csv')
    print('Review template:', prepare(args.per_question, output))
    print('Fill correctness with: correct, partial, or incorrect. Leave genuinely unreviewed rows blank.')
