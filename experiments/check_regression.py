"""Evaluation regression check: compare current metrics with committed baselines.

Two checks run without any LLM, so they are fast enough for every CI push:

1. Retrieval quality: rank the benchmark questions with each listed retrieval
   method and compare Recall@K, nDCG@K, MRR and evidence coverage with the
   baselines. A drop larger than the tolerance fails the check.
2. Metric code: re-score the frozen, published v0.1 answers with the current
   answer metrics. Any change at all fails the check, because these answers
   never change; only the code that scores them can.

Usage:
    python -m experiments.check_regression                    # check, exit 1 on regression
    python -m experiments.check_regression --update-baselines # accept current values

Updating baselines is a deliberate decision: commit evals/baselines.json on its
own and say in the commit message why the values changed.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval import Retriever  # noqa: E402
from src.schema import load_benchmark  # noqa: E402
from src.retrieval_eval import evaluate_retrieval  # noqa: E402
from src.evaluate import evaluate_answer  # noqa: E402

BASELINES = ROOT / 'evals' / 'baselines.json'
RETRIEVAL_CONFIG_KEYS = ('chunk_words', 'chunk_overlap_words', 'bm25_k1', 'bm25_b')
FROZEN_METRICS = ('reference_phrase_match', 'token_f1', 'reference_token_recall',
                  'citation_precision', 'abstention_marker')
DECIMALS = 6   # metrics are stored and compared at this precision
EXACT = 1e-9   # frozen answers are deterministic, so after rounding nothing may change


def compare(baseline, current, tolerance):
    """Compare metric dicts; return (failures, improvements) as lists of messages.

    A metric fails when it drops by more than `tolerance` or is missing from
    the current results. A rise above tolerance is reported as an improvement
    (the baseline may then be updated on purpose).
    """
    failures, improvements = [], []
    for name, base in baseline.items():
        if name not in current:
            failures.append(f'{name}: missing from current results')
            continue
        cur = current[name]
        if cur < base - tolerance:
            failures.append(f'{name}: {base:.4f} -> {cur:.4f} ({cur - base:+.4f})')
        elif cur > base + tolerance:
            improvements.append(f'{name}: {base:.4f} -> {cur:.4f} ({cur - base:+.4f})')
    return failures, improvements


def current_retrieval(cfg, spec):
    """Mean retrieval metrics per method for the configured chunking and BM25 settings."""
    retriever = Retriever(ROOT / 'data/documents', cfg['chunk_words'], cfg['chunk_overlap_words'],
                          cfg['embedding_model'], cfg.get('bm25_k1', 1.5), cfg.get('bm25_b', 0.75))
    questions = load_benchmark(ROOT / 'data/benchmark.json', retriever.docs)
    return {method: {k: round(v, DECIMALS) for k, v in
                     evaluate_retrieval(retriever, questions, method, spec['top_k'])[0].items()}
            for method in spec['methods']}


def current_frozen(spec):
    """Re-score each frozen answers file per pipeline with the current answer metrics."""
    questions = json.loads((ROOT / 'data/benchmark.json').read_text(encoding='utf-8'))
    by_legacy = {q['legacy_id']: q for q in questions if 'legacy_id' in q}
    out = {}
    for rel_path in spec['files']:
        rows = list(csv.DictReader((ROOT / rel_path).open(encoding='utf-8')))
        per_pipeline = {}
        for row in rows:
            q = by_legacy[int(row['question_id'])]
            m = evaluate_answer(row['answer'], row['reference_answer'], q['relevant_doc_ids'], [])
            per_pipeline.setdefault(row['pipeline'], []).append(m)
        out[rel_path] = {p: {name: round(sum(m[name] for m in ms) / len(ms), DECIMALS) for name in FROZEN_METRICS}
                         for p, ms in per_pipeline.items()}
    return out


def run(update=False):
    """Run both checks; return True when no regression was found."""
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    baselines = json.loads(BASELINES.read_text(encoding='utf-8'))
    r_spec, f_spec = baselines['retrieval'], baselines['frozen_answers']

    cfg_now = {key: cfg.get(key) for key in RETRIEVAL_CONFIG_KEYS}
    retrieval = current_retrieval(cfg, r_spec)
    frozen = current_frozen(f_spec)

    if update:
        r_spec['config'] = cfg_now
        r_spec['metrics'] = retrieval
        f_spec['metrics'] = frozen
        BASELINES.write_text(json.dumps(baselines, indent=2) + '\n', encoding='utf-8')
        print(f'Baselines updated in {BASELINES.relative_to(ROOT)}. Review the diff and commit it with a reason.')
        return True

    ok = True
    if r_spec['config'] != cfg_now:
        ok = False
        print(f"FAIL retrieval config changed: baseline {r_spec['config']} vs current {cfg_now}.\n"
              '     Retrieval baselines only hold for the settings they were measured with; '
              're-run with --update-baselines if the change is intended.')

    print(f"\nRetrieval (top_k={r_spec['top_k']}, tolerance={r_spec['tolerance']})")
    for method in r_spec['methods']:
        failures, improvements = compare(r_spec['metrics'][method], retrieval[method], r_spec['tolerance'])
        ok &= not failures
        status = 'FAIL' if failures else 'ok  '
        print(f'  {status} {method}: ' + ', '.join(f'{k}={v:.4f}' for k, v in retrieval[method].items()))
        for msg in failures:
            print(f'       regression  {msg}')
        for msg in improvements:
            print(f'       improved    {msg}')

    print('\nFrozen v0.1 answers re-scored with the current metric code (must match exactly)')
    for path, pipes in f_spec['metrics'].items():
        for pipeline, base in pipes.items():
            cur = frozen.get(path, {}).get(pipeline, {})
            failures, improvements = compare(base, cur, EXACT)
            changes = failures + improvements  # any change is a regression for frozen answers
            ok &= not changes
            print(f"  {'FAIL' if changes else 'ok  '} {Path(path).name} / {pipeline}")
            for msg in changes:
                print(f'       changed     {msg}')

    print('\nPASS: no evaluation regression.' if ok else
          '\nFAIL: evaluation regression. Fix the cause, or run --update-baselines and commit the '
          'new baselines with the reason if the change is intended.')
    return ok


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--update-baselines', action='store_true', help='Write current values as the new baselines')
    sys.exit(0 if run(p.parse_args().update_baselines) else 1)
