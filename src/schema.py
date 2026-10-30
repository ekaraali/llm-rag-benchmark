"""Benchmark question schema, validation and evidence-to-chunk resolution.

Each question labels its evidence as exact text quotes from the corpus rather
than as chunk IDs, because chunk IDs change whenever chunk size or overlap
changes. Relevant chunks are derived from the quotes for the active chunking.

Schema (one JSON object per question):

    question_id        "Q001"                     unique, stable
    question           str
    reference_answer   str, or null when answerable is false
    answerable         bool; false = the corpus does not contain the answer
    question_type      "single_hop" | "multi_hop" | "unanswerable"
    relevant_doc_ids   [doc_id, ...]              every document holding evidence
    evidence           [{"evidence_id": "E1", "doc_id": "rrf.md", "quote": "..."}]
    evidence_groups    [["E1"], ["E2", "E3"]]
    label_status       "draft" | "verified"       draft = not yet checked by a human
    legacy_id          int, optional              question id in v0.1 reports

evidence_groups semantics: every group is one piece of evidence the answer
needs (groups are combined with AND); the evidence IDs inside a group are
alternatives that each satisfy that piece (OR). A single-hop question has one
group; a multi-hop question has two or more.
"""

import json
import re
from pathlib import Path

QUESTION_TYPES = {'single_hop', 'multi_hop', 'unanswerable'}
LABEL_STATUSES = {'draft', 'verified'}
REQUIRED = ('question_id', 'question', 'reference_answer', 'answerable', 'question_type',
            'relevant_doc_ids', 'evidence', 'evidence_groups', 'label_status')
MIN_EVIDENCE_OVERLAP = 0.5  # share of the quote a chunk must contain to count as relevant


class SchemaError(ValueError):
    """Raised when a benchmark question breaks the schema."""


def locate_quote(doc_text, quote):
    """Return the (start, end) character span of `quote` in `doc_text`.

    Matching ignores differences in whitespace only; any other difference
    (wording, punctuation, case) is a labelling error and raises SchemaError.
    """
    words = quote.split()
    if not words:
        raise SchemaError('Evidence quote is empty')
    match = re.search(r'\s+'.join(re.escape(w) for w in words), doc_text)
    if not match:
        raise SchemaError(f'Quote not found in document: {quote[:80]!r}')
    return match.span()


def validate_question(q, docs):
    """Check one question against the schema and the corpus `docs` ({doc_id: text})."""
    qid = q.get('question_id', '?')
    missing = [f for f in REQUIRED if f not in q]
    if missing:
        raise SchemaError(f'{qid}: missing fields {missing}')
    if not re.fullmatch(r'Q\d{3,}', str(q['question_id'])):
        raise SchemaError(f'{qid}: question_id must look like Q001')
    if q['question_type'] not in QUESTION_TYPES:
        raise SchemaError(f'{qid}: unknown question_type {q["question_type"]!r}')
    if q['label_status'] not in LABEL_STATUSES:
        raise SchemaError(f'{qid}: unknown label_status {q["label_status"]!r}')
    if not isinstance(q['answerable'], bool):
        raise SchemaError(f'{qid}: answerable must be true or false')

    if not q['answerable']:
        if q['question_type'] != 'unanswerable':
            raise SchemaError(f'{qid}: unanswerable questions need question_type "unanswerable"')
        if q['reference_answer'] is not None or q['evidence'] or q['evidence_groups'] or q['relevant_doc_ids']:
            raise SchemaError(f'{qid}: unanswerable questions take null reference_answer and empty evidence fields')
        return

    if q['question_type'] == 'unanswerable':
        raise SchemaError(f'{qid}: answerable question cannot have question_type "unanswerable"')
    if not isinstance(q['reference_answer'], str) or not q['reference_answer'].strip():
        raise SchemaError(f'{qid}: answerable questions need a reference_answer')
    for d in q['relevant_doc_ids']:
        if d not in docs:
            raise SchemaError(f'{qid}: unknown document {d!r}')

    ids = [e.get('evidence_id') for e in q['evidence']]
    if not ids or len(ids) != len(set(ids)):
        raise SchemaError(f'{qid}: evidence IDs must be present and unique')
    for e in q['evidence']:
        if e.get('doc_id') not in q['relevant_doc_ids']:
            raise SchemaError(f'{qid}: evidence {e.get("evidence_id")} uses a document not in relevant_doc_ids')
        locate_quote(docs[e['doc_id']], e.get('quote', ''))

    groups = q['evidence_groups']
    if not groups or any(not g for g in groups):
        raise SchemaError(f'{qid}: evidence_groups needs at least one non-empty group')
    grouped = {eid for g in groups for eid in g}
    if grouped - set(ids):
        raise SchemaError(f'{qid}: evidence_groups reference unknown IDs {sorted(grouped - set(ids))}')
    if set(ids) - grouped:
        raise SchemaError(f'{qid}: evidence {sorted(set(ids) - grouped)} is not used by any group')
    if q['question_type'] == 'single_hop' and len(groups) != 1:
        raise SchemaError(f'{qid}: single_hop questions have exactly one evidence group')
    if q['question_type'] == 'multi_hop' and len(groups) < 2:
        raise SchemaError(f'{qid}: multi_hop questions need at least two evidence groups')
    used_docs = {e['doc_id'] for e in q['evidence']}
    if set(q['relevant_doc_ids']) - used_docs:
        raise SchemaError(f'{qid}: relevant_doc_ids lists documents without evidence')


def load_benchmark(path, docs):
    """Load and validate the benchmark JSON against corpus `docs` ({doc_id: text})."""
    questions = json.loads(Path(path).read_text(encoding='utf-8'))
    seen = set()
    for q in questions:
        validate_question(q, docs)
        if q['question_id'] in seen:
            raise SchemaError(f'Duplicate question_id {q["question_id"]}')
        seen.add(q['question_id'])
    return questions


def relevant_chunks(question, docs, chunks):
    """Map a question's evidence to chunk IDs for the active chunking.

    Returns (relevant_chunk_ids, group_chunk_sets): the set of all relevant
    chunk IDs, and per evidence group the set of chunk IDs that satisfy it.
    A chunk supports a quote when it contains at least MIN_EVIDENCE_OVERLAP of
    the quote's characters.
    """
    by_evidence = {}
    for e in question['evidence']:
        start, end = locate_quote(docs[e['doc_id']], e['quote'])
        need = MIN_EVIDENCE_OVERLAP * (end - start)
        by_evidence[e['evidence_id']] = {
            c['chunk_id'] for c in chunks
            if c['doc'] == e['doc_id'] and min(end, c['end']) - max(start, c['start']) >= need}
    groups = [set().union(*(by_evidence[eid] for eid in g)) for g in question['evidence_groups']]
    return set().union(*by_evidence.values()) if by_evidence else set(), groups
