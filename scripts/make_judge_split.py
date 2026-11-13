"""Create the fixed question-level dev/test split used to calibrate the LLM judge.

Questions (not individual answers) are split, so the top_k=3 and top_k=5
answers to the same question always land on the same side and nothing about a
test question can leak into prompt design. To keep both halves similar in
difficulty, questions are sorted by how well they were answered according to
the human labels (correct = 2, partial = 1, incorrect = 0, summed over all six
answers), taken in consecutive pairs, and the pairs are assigned in an
alternating dev/test, test/dev pattern. The result is deterministic.

Run once and commit data/judge_split.json; never regenerate it after looking
at judge results.

Usage: python -m scripts.make_judge_split
"""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['reports/v0.1/human_validated_reviews_k3.csv', 'reports/v0.1/human_validated_reviews_k5.csv']
POINTS = {'correct': 2, 'partial': 1, 'incorrect': 0}


def make_split(rows):
    """Return (dev, test) lists of v0.1 question ids from human-labelled rows."""
    score = Counter()
    for r in rows:
        score[int(r['question_id'])] += POINTS[r['correctness']]
    ordered = sorted(score, key=lambda q: (score[q], q))
    dev, test = [], []
    for i in range(0, len(ordered), 2):
        pair = ordered[i:i + 2]
        first, second = (dev, test) if (i // 2) % 2 == 0 else (test, dev)
        first.append(pair[0])
        if len(pair) > 1:
            second.append(pair[1])
    return sorted(dev), sorted(test)


if __name__ == '__main__':
    rows = [r for f in FILES for r in csv.DictReader((ROOT / f).open(encoding='utf-8'))]
    dev, test = make_split(rows)
    out = {
        'description': 'Question-level split for judge calibration. Ids are v0.1 question ids (legacy_id). '
                       'Tune judge prompts on dev only; evaluate the frozen prompt on test.',
        'method': 'stratified by summed human label score, consecutive pairs, alternating assignment',
        'source_files': FILES,
        'dev': dev,
        'test': test,
    }
    path = ROOT / 'data/judge_split.json'
    path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    for name, ids in (('dev', dev), ('test', test)):
        labels = Counter(r['correctness'] for r in rows if int(r['question_id']) in ids)
        print(f'{name}: {len(ids)} questions {ids} -> {sum(labels.values())} answers {dict(labels)}')
