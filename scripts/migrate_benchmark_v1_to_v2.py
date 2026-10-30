"""One-off migration of the v0.1 benchmark (16 questions) to the v0.2 schema.

v0.1 format: {"id", "question", "reference_answer", "relevant_docs"}.
v0.2 format: see src/schema.py.

The evidence quotes below were drafted from the corpus and are exported with
label_status "draft": a human must check each quote and switch the status to
"verified" before evidence-level metrics are reported as validated.

Usage: python -m scripts.migrate_benchmark_v1_to_v2 OLD.json NEW.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.schema import validate_question  # noqa: E402

# legacy id -> (doc_id, exact sentence from the document that supports the reference answer)
DRAFT_EVIDENCE = {
    1: ('chunking.md', 'Smaller chunks such as 256 tokens can isolate facts but fragment multi-paragraph evidence.'),
    2: ('chunking.md', 'A common starting overlap is 10 to 20 percent of chunk size.'),
    3: ('rrf.md', 'Each candidate document receives a score equal to the sum of 1/(k + rank) across retrievers where it appears.'),
    4: ('rrf.md', 'RRF combines BM25 lexical and dense semantic retrieval without needing directly comparable scores.'),
    5: ('hallucination.md', 'When relevant evidence is absent, the correct response can be an explicit abstention rather than a fabricated answer.'),
    6: ('hallucination.md', 'Faithfulness checks whether generated claims agree with supplied evidence; answer correctness checks agreement with a reference answer.'),
    7: ('sampling.md', 'Temperature scales the logits before sampling and changes randomness.'),
    8: ('sampling.md', 'Top-p nucleus sampling restricts generation to a dynamically selected set of tokens whose cumulative probability reaches p.'),
    9: ('evaluation.md', 'its per-query contribution is 1 divided by the rank of the first relevant result.'),
    10: ('evaluation.md', 'Evaluate retrieval independently of generation to distinguish evidence failures from answer synthesis failures.'),
    11: ('mlops.md', 'Experiment tracking records configurations, metrics, code versions, and artifacts for reproducibility.'),
    12: ('mlops.md', 'CI smoke tests validate code paths but are not equivalent to end-to-end LLM quality tests.'),
    13: ('prompt_injection.md', 'Systems should clearly separate instructions from retrieved text and avoid following commands embedded in source passages.'),
    14: ('latency.md', 'Reranking can increase retrieval quality while adding compute and latency.'),
    15: ('sampling.md', 'Setting temperature to zero is commonly used for deterministic-style decoding but repeatability is not guaranteed for every hosted provider.'),
    16: ('mlops.md', 'do not commit API keys or proprietary datasets to public repositories.'),
}


def migrate(old_questions, docs):
    """Convert v0.1 questions to the v0.2 schema and validate the result."""
    out = []
    for q in old_questions:
        doc_id, quote = DRAFT_EVIDENCE[q['id']]
        if doc_id not in q['relevant_docs']:
            raise ValueError(f"Q{q['id']}: drafted evidence document {doc_id} is not in the v0.1 relevant_docs")
        new = {
            'question_id': f"Q{q['id']:03d}",
            'question': q['question'],
            'reference_answer': q['reference_answer'],
            'answerable': True,
            'question_type': 'single_hop',
            'relevant_doc_ids': list(q['relevant_docs']),
            'evidence': [{'evidence_id': 'E1', 'doc_id': doc_id, 'quote': quote}],
            'evidence_groups': [['E1']],
            'label_status': 'draft',
            'legacy_id': q['id'],
        }
        validate_question(new, docs)
        out.append(new)
    return out


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('old', type=Path)
    a.add_argument('new', type=Path)
    a.add_argument('--docs', type=Path, default=ROOT / 'data/documents')
    args = a.parse_args()
    corpus = {p.name: p.read_text(encoding='utf-8') for p in sorted(args.docs.glob('*.md'))}
    migrated = migrate(json.loads(args.old.read_text(encoding='utf-8')), corpus)
    args.new.write_text(json.dumps(migrated, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {len(migrated)} questions to {args.new} (label_status=draft: review every quote).')
