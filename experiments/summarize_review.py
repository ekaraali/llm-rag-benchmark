"""Summarize manually evaluated answer quality; never count blanks as incorrect."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

VALID = {'correct', 'partial', 'incorrect'}


def summarize(source: Path):
    """Count review labels per pipeline, keeping unreviewed items separate."""
    with source.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    counts = defaultdict(lambda: {'correct': 0, 'partial': 0, 'incorrect': 0, 'unreviewed': 0})
    for row in rows:
        score = row['correctness'].strip().lower()
        if score and score not in VALID:
            raise ValueError(f"Invalid correctness label {score!r} for {row['pipeline']} Q{row['question_id']}")
        counts[row['pipeline']][score or 'unreviewed'] += 1
    return counts


def render(counts):
    """Format manual review counts as a copyable Markdown report."""
    lines = ['# Human-reviewed answer quality', '',
             '> This is a manual assessment of the generated answers, not an automatic LLM judge.', '',
             '| Pipeline | Reviewed | Correct | Partial | Incorrect | Correct / reviewed |',
             '|---|---:|---:|---:|---:|---:|']
    for name, c in sorted(counts.items()):
        # Denominator excludes blank rows; unreviewed is not incorrect.
        n = sum(c[x] for x in VALID)
        score = f"{100*c['correct']/n:.1f}%" if n else 'N/A'
        lines.append(f"| {name} | {n} | {c['correct']} | {c['partial']} | {c['incorrect']} | {score} |")
        if c['unreviewed']:
            lines.append(f'*{name}: {c["unreviewed"]} answer(s) remain unreviewed.*')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('review_csv', type=Path)
    a.add_argument('--output', type=Path)
    args = a.parse_args()
    output = args.output or args.review_csv.with_name('human_review_results.md')
    output.write_text(render(summarize(args.review_csv)), encoding='utf-8')
    print('Review report:', output)
