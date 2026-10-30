"""Run reproducible direct/RAG comparisons and export per-question metrics.

Usage: python -m experiments.run_benchmark [--top-k 5] [--limit 3]
The raw results and effective config are kept together in a timestamped folder.

Pipelines (v0.2):
    direct       no retrieval; the model answers from its own knowledge
    dense_rag    dense semantic retrieval (sentence-transformers)
    bm25_rag     BM25 lexical retrieval
    hybrid_rag   BM25 + dense, fused with Reciprocal Rank Fusion (RRF)
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval import Retriever  # noqa: E402
from src.llm import generate  # noqa: E402
from src.evaluate import evaluate_answer  # noqa: E402
from src.schema import load_benchmark, relevant_chunks  # noqa: E402
from src import retrieval_metrics as rm  # noqa: E402

# pipeline name -> retrieval method (None = no retrieval)
PIPELINES = {'direct': None, 'dense_rag': 'dense', 'bm25_rag': 'bm25', 'hybrid_rag': 'hybrid'}
LEGACY_PIPELINES = {
    'hybrid_reasoning_rag': 'was renamed to "hybrid_rag" in v0.2 (BM25 replaces TF-IDF and the '
                            'and/versus query-splitting heuristic was removed). Check out tag v0.1.0 to reproduce v0.1.',
}
RETRIEVAL_METRICS = ('chunk_recall_at_k', 'chunk_ndcg_at_k', 'chunk_mrr', 'evidence_coverage_at_k',
                     'doc_recall_at_k', 'doc_mrr')
ANSWER_METRICS = ('reference_phrase_match', 'retrieval_hit', 'citation_precision')


def check_pipelines(names):
    """Fail early, with a clear message, on unknown or renamed pipelines."""
    for name in names:
        if name in LEGACY_PIPELINES:
            raise ValueError(f'Pipeline "{name}" {LEGACY_PIPELINES[name]}')
        if name not in PIPELINES:
            raise ValueError(f'Unknown pipeline {name!r}; expected one of {sorted(PIPELINES)}')


def retrieval_scores(ranked_chunks, q, groups, relevant_ids, k):
    """Chunk- and document-level retrieval metrics for one answerable question."""
    chunk_ids = [c['chunk_id'] for c in ranked_chunks]
    docs = rm.dedupe_docs(ranked_chunks)
    rel_docs = set(q['relevant_doc_ids'])
    return {
        'chunk_recall_at_k': rm.recall_at_k(chunk_ids, relevant_ids, k),
        'chunk_ndcg_at_k': rm.ndcg_at_k(chunk_ids, relevant_ids, k),
        'chunk_mrr': rm.reciprocal_rank(chunk_ids, relevant_ids),
        'evidence_coverage_at_k': rm.evidence_coverage_at_k(chunk_ids, groups, k),
        'doc_recall_at_k': rm.recall_at_k(docs, rel_docs, k),
        'doc_mrr': rm.reciprocal_rank(docs, rel_docs),
    }


def mean(values):
    """Average of the non-missing values, or None if there are none."""
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


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
    check_pipelines(cfg['pipelines'])
    if cfg['provider'] == 'mock':
        print('WARNING: mock mode is a smoke test, NOT an LLM benchmark. Do not publish metrics as model performance.')
    if args.lexical_smoke:
        print('WARNING: --lexical-smoke runs every retrieval pipeline with BM25 only (no embeddings).')

    retriever = Retriever(ROOT / 'data/documents', cfg['chunk_words'], cfg['chunk_overlap_words'],
                          cfg['embedding_model'], cfg.get('bm25_k1', 1.5), cfg.get('bm25_b', 0.75))
    questions = load_benchmark(ROOT / 'data/benchmark.json', retriever.docs)
    if args.limit:
        questions = questions[:args.limit]

    # Resolve evidence quotes to chunk IDs for this chunking configuration.
    evidence = {}
    for q in questions:
        if q['answerable']:
            rel, groups = relevant_chunks(q, retriever.docs, retriever.chunks)
            if any(not g for g in groups):
                raise ValueError(f"{q['question_id']}: an evidence quote is not contained in any chunk "
                                 f"with chunk_words={cfg['chunk_words']}; use larger chunks or shorter quotes")
            evidence[q['question_id']] = (rel, groups)

    rows = []
    k = cfg['top_k']
    for pipeline in cfg['pipelines']:
        method = PIPELINES[pipeline]
        if method and args.lexical_smoke:
            method = 'bm25'
        for q in questions:
            started = time.perf_counter()
            ranked = [retriever.chunks[i] for i in retriever.rank(q['question'], method)] if method else []
            hits = ranked[:k]
            answer = generate(q['question'], hits, cfg)
            secs = round(time.perf_counter() - started, 4)
            metrics = evaluate_answer(answer, q['reference_answer'], q['relevant_doc_ids'], hits)
            if not method:
                metrics.update(retrieval_hit=None, citation_precision=None)
            if method and q['answerable']:
                rel, groups = evidence[q['question_id']]
                ret = retrieval_scores(ranked, q, groups, rel, k)
            else:
                ret = dict.fromkeys(RETRIEVAL_METRICS)
            rows.append({'pipeline': pipeline, 'question_id': q['question_id'], 'question': q['question'],
                         'question_type': q['question_type'], 'answerable': q['answerable'],
                         'answer': answer, 'reference_answer': q['reference_answer'] or '',
                         'retrieved_chunks': '|'.join(c['chunk_id'] for c in hits),
                         'latency_seconds': secs, **metrics, **ret})
            shown = 'n/a' if ret['chunk_recall_at_k'] is None else f"{ret['chunk_recall_at_k']:.2f}"
            print(f"{pipeline:12} {q['question_id']} recall@{k}={shown} abstained={metrics['abstained']} {secs:.2f}s")

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = ROOT / 'results' / (('SMOKE_' if cfg['provider'] == 'mock' else '') + stamp)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'per_question.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Averages skip missing values (direct pipeline, unanswerable questions); these are not accuracy.
    summary = []
    for pipeline in cfg['pipelines']:
        r = [x for x in rows if x['pipeline'] == pipeline]
        ans = [x for x in r if x['answerable']]
        unans = [x for x in r if not x['answerable']]
        summary.append({
            'pipeline': pipeline, 'n': len(r), 'n_unanswerable': len(unans),
            **{m: mean(x[m] for x in ans) for m in ANSWER_METRICS + RETRIEVAL_METRICS},
            'false_abstention_rate': mean(x['abstained'] for x in ans),
            'abstention_accuracy_unanswerable': mean(x['abstained'] for x in unans),
            'latency_seconds': mean(x['latency_seconds'] for x in r),
        })
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (out / 'config.json').write_text(json.dumps({**cfg, 'lexical_smoke': args.lexical_smoke,
                                                 'benchmark_schema': 'v0.2'}, indent=2))

    fmt = lambda v: 'n/a' if v is None else f'{v:.2f}'  # noqa: E731
    status = Counter(q['label_status'] for q in questions)
    (out / 'README_results.md').write_text('\n'.join([
        '# Experiment results' + (' — SMOKE TEST ONLY' if cfg['provider'] == 'mock' else ''),
        '', '> Reference phrase match is a weak proxy; manually review answers before claiming correctness.',
        f'> Evidence labels: {dict(status)} (draft = not yet human-verified).',
        '', f'| Pipeline | Questions | Phrase match | Chunk recall@{k} | nDCG@{k} | MRR | Evidence coverage@{k} | Avg latency (s) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
        *[f"| {s['pipeline']} | {s['n']} | {fmt(s['reference_phrase_match'])} | {fmt(s['chunk_recall_at_k'])} | "
          f"{fmt(s['chunk_ndcg_at_k'])} | {fmt(s['chunk_mrr'])} | {fmt(s['evidence_coverage_at_k'])} | "
          f"{fmt(s['latency_seconds'])} |" for s in summary],
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
                    mlflow.log_params({key: str(v) for key, v in cfg.items() if key != 'api_key_env'})
                    mlflow.log_param('pipeline', s['pipeline'])
                    mlflow.log_metrics({key: v for key, v in s.items()
                                        if key not in ('pipeline', 'n', 'n_unanswerable') and v is not None})
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
    p.add_argument('--lexical-smoke', action='store_true', help='BM25 only, no embedding download; smoke checks only')
    run(p.parse_args())
