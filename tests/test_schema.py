"""Benchmark schema validation and evidence-to-chunk mapping."""

import copy
from pathlib import Path
import pytest

from src.schema import SchemaError, load_benchmark, locate_quote, relevant_chunks, validate_question
from src.retrieval import chunk_document

ROOT = Path(__file__).resolve().parents[1]
DOCS = {p.name: p.read_text(encoding='utf-8') for p in (ROOT / 'data/documents').glob('*.md')}

GOOD = {
    'question_id': 'Q900', 'question': 'Which methods can RRF combine?',
    'reference_answer': 'BM25 and dense retrieval', 'answerable': True, 'question_type': 'single_hop',
    'relevant_doc_ids': ['rrf.md'],
    'evidence': [{'evidence_id': 'E1', 'doc_id': 'rrf.md',
                  'quote': 'RRF combines BM25 lexical and dense semantic retrieval'}],
    'evidence_groups': [['E1']], 'label_status': 'draft',
}


def test_repository_benchmark_is_valid():
    questions = load_benchmark(ROOT / 'data/benchmark.json', DOCS)
    assert len(questions) >= 16
    assert all(q['question_id'].startswith('Q') for q in questions)


def test_quote_matching_ignores_whitespace_only():
    text = 'Alpha beta\n  gamma.'
    assert locate_quote(text, 'beta gamma') == (6, 18)
    with pytest.raises(SchemaError):
        locate_quote(text, 'Beta gamma')


@pytest.mark.parametrize('change, message', [
    (lambda q: q.update(question_id='17'), 'question_id'),
    (lambda q: q['evidence'][0].update(quote='RRF combines everything'), 'not found'),
    (lambda q: q.update(evidence_groups=[['E2']]), 'unknown IDs'),
    (lambda q: q.update(question_type='multi_hop'), 'at least two'),
    (lambda q: q.update(relevant_doc_ids=['rrf.md', 'mlops.md']), 'without evidence'),
    (lambda q: q.update(answerable=False), 'unanswerable'),
])
def test_invalid_questions_are_rejected(change, message):
    q = copy.deepcopy(GOOD)
    change(q)
    with pytest.raises(SchemaError, match=message):
        validate_question(q, DOCS)


def test_unanswerable_question_shape():
    q = {**copy.deepcopy(GOOD), 'answerable': False, 'question_type': 'unanswerable',
         'reference_answer': None, 'relevant_doc_ids': [], 'evidence': [], 'evidence_groups': []}
    validate_question(q, DOCS)


def test_relevant_chunks_follow_chunking():
    text = ' '.join(f'w{i}' for i in range(30))
    docs = {'d.md': text}
    q = {**copy.deepcopy(GOOD), 'relevant_doc_ids': ['d.md'], 'evidence_groups': [['E1'], ['E2']],
         'evidence': [{'evidence_id': 'E1', 'doc_id': 'd.md', 'quote': 'w2 w3 w4'},
                      {'evidence_id': 'E2', 'doc_id': 'd.md', 'quote': 'w25 w26'}]}
    small = chunk_document('d.md', text, size=10, overlap=0)
    rel, groups = relevant_chunks(q, docs, small)
    assert rel == {'d.md#c0', 'd.md#c2'}
    assert groups == [{'d.md#c0'}, {'d.md#c2'}]
    big = chunk_document('d.md', text, size=30, overlap=0)
    assert relevant_chunks(q, docs, big)[0] == {'d.md#c0'}
