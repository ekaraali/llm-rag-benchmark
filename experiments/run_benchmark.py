"""Run reproducible direct/RAG comparisons and export per-question metrics.

Usage: python -m experiments.run_benchmark [--top-k 5] [--limit 3]
The raw results and effective config are kept together in a timestamped folder.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval import Retriever, decompose_query
from src.llm import generate
from src.evaluate import evaluate_answer


def run(args):
    """Run each configured pipeline on the same questions and save artifacts."""
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if args.provider:
        cfg['provider'] = args.provider
    if args.model:
        cfg['model'] = args.model
    # CLI overrides allow controlled hyperparameter runs without editing YAML.
    if args.top_k is not None:
        cfg['top_k'] = args.top_k
    if cfg['top_k'] < 1:
        raise ValueError('top_k must be >= 1')
    if cfg['provider'] == 'mock':
        print('WARNING: mock mode is a smoke test, NOT an LLM benchmark. Do not publish metrics as model performance.')
    questions = json.loads((ROOT / 'data/benchmark.json').read_text())
    if args.limit:
        questions = questions[:args.limit]
    dense_needed = any(p != 'direct' for p in cfg['pipelines']) and not args.lexical_smoke
    retriever = Retriever(ROOT / 'data/documents', cfg['chunk_words'], cfg['chunk_overlap_words'],
                          cfg['embedding_model'])
    # Log each response individually so errors can be reviewed after the run.
    rows = []
    for pipeline in cfg['pipelines']:
        for q in questions:
            started = time.perf_counter()
            if pipeline == 'direct':
                hits = []
            elif pipeline == 'dense_rag':
                hits = retriever.search(q['question'], cfg['top_k'], lexical_only=args.lexical_smoke)
            elif pipeline == 'hybrid_reasoning_rag':
                # Hybrid = rank-fused retrieval; query decomposition only
                # occurs for compound questions handled by the heuristic.
                # This is NOT a full multi-step reasoning agent.
                expanded = decompose_query(q['question'])
                candidates = []
                for query in expanded:
                    candidates.extend(retriever.search(query, cfg['top_k'], hybrid=not args.lexical_smoke,
                                                       lexical_only=args.lexical_smoke))
                hits = list({x['doc']: x for x in candidates}.values())[:cfg['top_k']]
            else:
                raise ValueError('Unknown pipeline: ' + pipeline)
            answer = generate(q['question'], hits, cfg)
            secs = round(time.perf_counter() - started, 4)
            metrics = evaluate_answer(answer, q['reference_answer'], q['relevant_docs'], hits)
            rows.append({'pipeline': pipeline, 'question_id': q['id'], 'question': q['question'],
                         'answer': answer, 'reference_answer': q['reference_answer'],
                         'retrieved_docs': '|'.join(x['doc'] for x in hits),
                         'latency_seconds': secs, **metrics})
            print(f"{pipeline:24} Q{q['id']:02} hit={metrics['retrieval_hit']} phrase={metrics['reference_phrase_match']} {secs:.2f}s")
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = ROOT / 'results' / (('SMOKE_' if cfg['provider'] == 'mock' else '') + stamp)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'per_question.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    # Average numerical diagnostics per pipeline; these are not accuracy.
    summary = []
    for pipeline in cfg['pipelines']:
        r = [x for x in rows if x['pipeline'] == pipeline]
        summary.append({'pipeline': pipeline, 'n': len(r), **{
            metric: round(sum(x[metric] for x in r) / len(r), 4)
            for metric in ('reference_phrase_match', 'retrieval_hit', 'citation_precision', 'latency_seconds')}})
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    (out / 'README_results.md').write_text('\n'.join([
        '# Experiment results' + (' — SMOKE TEST ONLY' if cfg['provider'] == 'mock' else ''),
        '', '> Reference phrase match is a weak proxy; manually review answers before claiming correctness.',
        '', '| Pipeline | Questions | Phrase match | Retrieval hit | Citation precision | Avg latency (s) |',
        '|---|---:|---:|---:|---:|---:|',
        *[f"| {r['pipeline']} | {r['n']} | {r['reference_phrase_match']:.2f} | {r['retrieval_hit']:.2f} | {r['citation_precision']:.2f} | {r['latency_seconds']:.2f} |" for r in summary],
        '', 'Model: `' + cfg['model'] + '`, provider: `' + cfg['provider'] + '`.',
    ]))
    # Log the effective config and summary metrics for run comparison.
    # Mock runs are excluded to avoid polluting real-model experiment history.
    if cfg.get('track_mlflow') and cfg['provider'] != 'mock':
        try:
            import mlflow
            mlflow.set_experiment(cfg['mlflow_experiment'])
            for s in summary:
                with mlflow.start_run(run_name=s['pipeline']):
                    mlflow.log_params({k: str(v) for k, v in cfg.items() if k != 'api_key_env'})
                    mlflow.log_param('pipeline', s['pipeline'])
                    mlflow.log_metrics({k: v for k, v in s.items() if k not in ('pipeline', 'n')})
                    mlflow.log_artifact(str(out / 'summary.csv'))
        except ImportError:
            print('MLflow not installed; skipping experiment tracking.')
    print('\nResults:', out)
    print((out / 'README_results.md').read_text())
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--provider', choices=['ollama', 'openai_compatible', 'mock'])
    p.add_argument('--model', help='Override model name')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--top-k', type=int, default=None, help='Override retrieval depth without editing config.yaml')
    p.add_argument('--lexical-smoke', action='store_true', help='Skip embedding downloads; for smoke checks only')
    run(p.parse_args())
