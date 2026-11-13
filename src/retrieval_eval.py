"""Retrieval-only evaluation: score rankings against labelled evidence, no LLM needed.

Shared by the benchmark runner (per question, alongside generated answers) and
the CI regression check (retrieval quality on its own, in seconds).
"""

from src import retrieval_metrics as rm
from src.schema import relevant_chunks

RETRIEVAL_METRICS = ('chunk_recall_at_k', 'chunk_ndcg_at_k', 'chunk_mrr', 'evidence_coverage_at_k',
                     'doc_recall_at_k', 'doc_mrr')


def resolve_evidence(questions, docs, chunks, chunk_words=None):
    """Map each answerable question to (relevant_chunk_ids, evidence_group_sets).

    Fails if an evidence quote is not contained in any chunk, which means the
    chunk size is too small for the labelled evidence.
    """
    evidence = {}
    for q in questions:
        if not q['answerable']:
            continue
        rel, groups = relevant_chunks(q, docs, chunks)
        if any(not g for g in groups):
            raise ValueError(f"{q['question_id']}: an evidence quote is not contained in any chunk "
                             f"(chunk_words={chunk_words}); use larger chunks or shorter quotes")
        evidence[q['question_id']] = (rel, groups)
    return evidence


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


def evaluate_retrieval(retriever, questions, method, k):
    """Mean retrieval metrics of `method` over the answerable questions.

    Returns (means, per_question) where per_question maps question_id to its scores.
    """
    evidence = resolve_evidence(questions, retriever.docs, retriever.chunks)
    per_question = {}
    for q in questions:
        if q['question_id'] not in evidence:
            continue
        rel, groups = evidence[q['question_id']]
        ranked = [retriever.chunks[i] for i in retriever.rank(q['question'], method)]
        per_question[q['question_id']] = retrieval_scores(ranked, q, groups, rel, k)
    if not per_question:
        raise ValueError('No answerable questions to evaluate retrieval on')
    means = {m: sum(s[m] for s in per_question.values()) / len(per_question) for m in RETRIEVAL_METRICS}
    return means, per_question
