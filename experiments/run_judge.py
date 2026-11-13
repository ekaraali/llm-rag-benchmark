"""Calibrate an LLM judge against the human-validated v0.1 labels.

Usage:
    python -m experiments.run_judge --dry-run                  # show one prompt and the cost estimate, no API calls
    python -m experiments.run_judge --split dev --version rubric
    python -m experiments.run_judge --split test --version rubric --confirm-test

Protocol: design and adjust judge prompts using the dev split only. When a
prompt version is final, evaluate it once on the test split and report that
result. Running on test requires --confirm-test so it cannot happen by accident.

Judgements are cached in results/judge_cache.jsonl (not committed), so a
repeated run with the same model and prompt costs nothing.
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.judge import (LABELS, ANSWER_STATUSES, JUDGE_PROMPTS, JudgeError, build_judge_messages,  # noqa: E402
                       cache_key, call_judge, estimate_cost, parse_judgement)
from src.agreement import (accuracy, cohen_kappa, confusion_matrix, macro_f1,  # noqa: E402
                           cluster_bootstrap_ci)

CACHE = ROOT / 'results' / 'judge_cache.jsonl'
EST_INPUT_TOKENS, EST_OUTPUT_TOKENS = 700, 80  # rough per-call estimate for --dry-run


def load_items(split_name):
    """Human-labelled v0.1 answers belonging to the requested split."""
    split = json.loads((ROOT / 'data/judge_split.json').read_text(encoding='utf-8'))
    wanted = set(split['dev']) | set(split['test']) if split_name == 'all' else set(split[split_name])
    items = []
    for path in split['source_files']:
        top_k = 5 if path.endswith('k5.csv') else 3
        for r in csv.DictReader((ROOT / path).open(encoding='utf-8')):
            qid = int(r['question_id'])
            if qid in wanted:
                items.append({'top_k': top_k, 'pipeline': r['pipeline'], 'question_id': qid,
                              'question': r['question'], 'reference': r['reference_answer'],
                              'answer': r['answer'], 'human': r['correctness']})
    return items


def load_cache():
    """Cached judgements keyed by cache_key."""
    if not CACHE.exists():
        return {}
    with CACHE.open(encoding='utf-8') as f:
        return {e['key']: e for e in map(json.loads, f)}


def judge_items(items, cfg, version):
    """Judge every item (using the cache); return (rows, new_input_tokens, new_output_tokens)."""
    cache = load_cache()
    CACHE.parent.mkdir(exist_ok=True)
    rows, tin, tout = [], 0, 0
    with CACHE.open('a', encoding='utf-8') as cache_file:
        for n, it in enumerate(items, start=1):
            key = cache_key(cfg, version, it['question'], it['reference'], it['answer'])
            if key in cache:
                reply = cache[key]['reply']
            else:
                system, user = build_judge_messages(it['question'], it['reference'], it['answer'], version)
                reply, i, o = call_judge(cfg, system, user)
                tin, tout = tin + i, tout + o
                cache[key] = {'key': key, 'reply': reply}
                cache_file.write(json.dumps(cache[key]) + '\n')
                cache_file.flush()
            try:
                parsed, error = parse_judgement(reply, version), ''
            except JudgeError as e:
                parsed, error = {'label': None}, str(e)
            rows.append({**it, 'judge': parsed['label'], 'answer_status': parsed.get('answer_status', ''),
                         'reason': parsed.get('reason', ''), 'error': error})
            print(f"[{n}/{len(items)}] Q{it['question_id']:02d} {it['pipeline']:22} k={it['top_k']} "
                  f"human={it['human']:9} judge={parsed['label'] or 'ERROR'}")
    return rows, tin, tout


def agreement_report(rows):
    """Agreement statistics between human and judge labels on the parsed rows."""
    ok = [r for r in rows if r['judge']]
    human, judge = [r['human'] for r in ok], [r['judge'] for r in ok]
    clusters = defaultdict(list)
    for r in ok:
        clusters[r['question_id']].append(r)

    def stat(fn):
        return lambda sample: fn([r['human'] for r in sample], [r['judge'] for r in sample])

    kappa = lambda h, j: cohen_kappa(h, j, LABELS)  # noqa: E731
    wkappa = lambda h, j: cohen_kappa(h, j, LABELS, weights='linear')  # noqa: E731
    return {
        'n': len(ok), 'parse_errors': len(rows) - len(ok), 'questions': len(clusters),
        'accuracy': accuracy(human, judge),
        'cohen_kappa': kappa(human, judge),
        'cohen_kappa_ci95': cluster_bootstrap_ci(clusters, stat(kappa)),
        'weighted_kappa': wkappa(human, judge),
        'weighted_kappa_ci95': cluster_bootstrap_ci(clusters, stat(wkappa)),
        'macro_f1': macro_f1(human, judge, LABELS),
        'confusion_matrix': confusion_matrix(human, judge, LABELS),
    }


def write_outputs(rows, report, cfg, args, tin, tout):
    """Save judgements, a Markdown report and a JSON summary; return the folder."""
    out = ROOT / 'results' / f"judge_{time.strftime('%Y%m%d-%H%M%S')}_{args.split}_{args.version}"
    out.mkdir(parents=True)
    with (out / 'judgements.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    cost = estimate_cost(tin, tout, cfg['price_per_mtok'])
    summary = {'split': args.split, 'prompt_version': args.version, 'provider': cfg['provider'],
               'model': cfg['model'], 'new_input_tokens': tin, 'new_output_tokens': tout,
               'estimated_cost_usd': round(cost, 4), **report}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    m = report['confusion_matrix']
    lo, hi = report['cohen_kappa_ci95']
    wlo, whi = report['weighted_kappa_ci95']
    lines = [
        f"# Judge calibration: {args.split} split, prompt `{args.version}`",
        '', f"Judge: `{cfg['model']}` via `{cfg['provider']}`. "
            f"{report['n']} answers from {report['questions']} questions; {report['parse_errors']} unparseable replies excluded.",
        '', '| Metric | Value | 95% CI (question-level bootstrap) |', '|---|---:|---:|',
        f"| Accuracy (exact label agreement) | {report['accuracy']:.3f} | |",
        f"| Cohen's κ | {report['cohen_kappa']:.3f} | {lo:.3f} – {hi:.3f} |",
        f"| Weighted κ (linear) | {report['weighted_kappa']:.3f} | {wlo:.3f} – {whi:.3f} |",
        f"| Macro-F1 (human = reference) | {report['macro_f1']:.3f} | |",
        '', 'Confusion matrix (rows = human, columns = judge):', '',
        '| human \\ judge | ' + ' | '.join(LABELS) + ' |', '|---|' + '---:|' * len(LABELS),
        *[f'| {LABELS[i]} | ' + ' | '.join(str(v) for v in m[i]) + ' |' for i in range(len(LABELS))],
    ]
    if args.version == 'rubric':
        status = Counter((r['pipeline'], r['answer_status']) for r in rows if r['judge'])
        pipelines = sorted({r['pipeline'] for r in rows})
        lines += ['', 'Answer status according to the judge:', '',
                  '| Pipeline | ' + ' | '.join(ANSWER_STATUSES) + ' |', '|---|' + '---:|' * len(ANSWER_STATUSES),
                  *[f'| {p} | ' + ' | '.join(str(status[(p, s)]) for s in ANSWER_STATUSES) + ' |' for p in pipelines]]
    disagreements = [r for r in rows if r['judge'] and r['judge'] != r['human']]
    lines += ['', f'## Disagreements ({len(disagreements)})', '',
              '| Q | Pipeline | k | Human | Judge | Judge reason |', '|---|---|---:|---|---|---|',
              *[f"| {r['question_id']} | {r['pipeline']} | {r['top_k']} | {r['human']} | {r['judge']} | "
                f"{r['reason'].replace('|', '/')} |" for r in disagreements],
              '', f'New API usage: {tin} input + {tout} output tokens, about ${cost:.4f} (cached judgements cost nothing).']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', choices=['dev', 'test', 'all'], default='dev')
    p.add_argument('--version', choices=sorted(JUDGE_PROMPTS), default='rubric', help='Judge prompt version')
    p.add_argument('--confirm-test', action='store_true', help='Required to evaluate on the test split')
    p.add_argument('--limit', type=int, help='Judge only the first N answers (quick check)')
    p.add_argument('--provider', choices=['anthropic', 'openai_compatible', 'mock'], help='Override judge provider')
    p.add_argument('--dry-run', action='store_true', help='Print one prompt and the cost estimate; no API calls')
    args = p.parse_args()

    if args.split in ('test', 'all') and not args.confirm_test:
        sys.exit('Refusing to run on the test split without --confirm-test. '
                 'Tune prompts on dev; evaluate the final prompt on test once.')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())['judge']
    if args.provider:
        cfg['provider'] = args.provider
    items = load_items(args.split)[:args.limit] if args.limit else load_items(args.split)

    if args.dry_run:
        system, user = build_judge_messages(items[0]['question'], items[0]['reference'], items[0]['answer'], args.version)
        cache = load_cache()
        todo = sum(cache_key(cfg, args.version, it['question'], it['reference'], it['answer']) not in cache
                   for it in items)
        est = estimate_cost(todo * EST_INPUT_TOKENS, todo * EST_OUTPUT_TOKENS, cfg['price_per_mtok'])
        print(f'--- system prompt ---\n{system}\n\n--- first user message ---\n{user}\n')
        print(f"{len(items)} answers on the {args.split} split, {todo} not cached. "
              f"Model {cfg['model']}: estimated cost about ${est:.4f}.")
        return

    rows, tin, tout = judge_items(items, cfg, args.version)
    if cfg['provider'] == 'mock':
        print('WARNING: mock judge; these numbers test the code path only.')
    out = write_outputs(rows, agreement_report(rows), cfg, args, tin, tout)
    print('\n' + (out / 'report.md').read_text(encoding='utf-8'))
    print(f'Saved to {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
