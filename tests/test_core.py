"""Regression tests for chunking, retrieval fusion, answer metrics and pipeline names."""

import pytest

from src.retrieval import split_words, chunk_document, rrf_fuse, Retriever
from src.evaluate import evaluate_answer, has_abstention_marker
from src.llm import build_messages, prompt_version, RAG_SYSTEM_PROMPTS
from experiments.run_benchmark import check_pipelines


def test_split_overlap():
    assert split_words('one two three four five', 3, 1) == ['one two three', 'three four five', 'five']


def test_invalid_chunk():
    with pytest.raises(ValueError):
        split_words('hi', 10, 10)


def test_chunk_spans_and_ids():
    text = 'alpha beta\ngamma  delta epsilon'
    chunks = chunk_document('doc.md', text, size=3, overlap=1)
    assert [c['chunk_id'] for c in chunks] == ['doc.md#c0', 'doc.md#c1', 'doc.md#c2']
    assert chunks[0]['text'] == 'alpha beta gamma'
    assert chunks[2]['text'] == 'epsilon'
    assert text[chunks[1]['start']:chunks[1]['end']] == 'gamma  delta epsilon'


def test_rrf_fuse():
    # item 2 is first in both lists; items 0 and 1 swap places, so they tie
    assert rrf_fuse([2, 0, 1], [2, 1, 0], k=60) == [2, 0, 1]  # tie broken by index
    # an item found by only one retriever still receives a score
    assert rrf_fuse([0, 1], [3], k=60) == [0, 3, 1]


def test_bm25_retriever_ranks_exact_term_first(tmp_path):
    (tmp_path / 'a.md').write_text('Reciprocal rank fusion merges rankings.')
    (tmp_path / 'b.md').write_text('Temperature changes sampling randomness.')
    r = Retriever(tmp_path, chunk_words=50, overlap_words=5)
    top = r.search('what does temperature change', k=1, method='bm25')
    assert top[0]['doc'] == 'b.md'
    with pytest.raises(ValueError):
        r.rank('x', 'tfidf')


def test_evaluation():
    m = evaluate_answer('10 to 20 percent [chunking.md]', '10 to 20 percent',
                        ['chunking.md'], [{'doc': 'chunking.md'}])
    assert m['reference_phrase_match'] == 1
    assert m['retrieval_hit'] == 1
    assert m['citation_precision'] == 1
    assert m['abstention_marker'] == 0


def test_unanswerable_evaluation():
    m = evaluate_answer('Insufficient evidence.', None, [], [])
    assert m['abstention_marker'] == 1
    assert m['reference_phrase_match'] is None
    # a hedged answer still carries the marker; the metric cannot tell it from a refusal
    assert has_abstention_marker('Insufficient evidence. However, the excerpt says 10 to 20 percent.')


def test_pipeline_names():
    check_pipelines(['direct', 'dense_rag', 'bm25_rag', 'hybrid_rag'])
    with pytest.raises(ValueError, match='renamed'):
        check_pipelines(['hybrid_reasoning_rag'])
    with pytest.raises(ValueError, match='Unknown'):
        check_pipelines(['reranked_rag'])


class _BagOfWordsEncoder:
    """Tiny stand-in for a sentence-transformer: normalised word-count vectors."""
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np
        rows = []
        for t in texts:
            words = t.lower().replace('.', ' ').split()
            v = np.array([words.count(w) for w in self.vocab], dtype=float)
            rows.append(v / (np.linalg.norm(v) or 1.0))
        return np.array(rows)


def test_dense_and_hybrid_rankings(tmp_path):
    (tmp_path / 'a.md').write_text('rank fusion merges rankings')
    (tmp_path / 'b.md').write_text('temperature changes randomness')
    (tmp_path / 'c.md').write_text('latency and cost of rankings')
    enc = _BagOfWordsEncoder(['rank', 'fusion', 'rankings', 'temperature', 'randomness', 'latency', 'cost'])
    r = Retriever(tmp_path, chunk_words=50, overlap_words=5, embedding_model=enc)
    assert r.search('temperature randomness', 1, method='dense')[0]['doc'] == 'b.md'
    hybrid = [c['doc'] for c in r.search('rank fusion rankings', 3, method='hybrid')]
    assert hybrid[0] == 'a.md' and sorted(hybrid) == ['a.md', 'b.md', 'c.md']


def test_model_name_string_loads_sentence_transformer(tmp_path, monkeypatch):
    """Regression: a model *name* (str) must be loaded, not mistaken for an encoder (str has .encode)."""
    import sys
    import types
    loaded = {}

    class FakeST(_BagOfWordsEncoder):
        def __init__(self, name, device=None):
            loaded['name'] = name
            super().__init__(['temperature', 'fusion'])

    monkeypatch.setitem(sys.modules, 'sentence_transformers', types.SimpleNamespace(SentenceTransformer=FakeST))
    (tmp_path / 'a.md').write_text('temperature changes randomness')
    r = Retriever(tmp_path, chunk_words=50, overlap_words=5, embedding_model='some/model-name')
    assert r.search('temperature', 1, method='dense')[0]['doc'] == 'a.md'
    assert loaded['name'] == 'some/model-name'


def test_prompt_versions():
    passages = [{'doc': 'rrf.md', 'text': 'RRF fuses ranks.'}]
    assert prompt_version({}) == 'strict'  # default keeps v0.2 behaviour
    strict = build_messages('Q?', passages, {})
    inference = build_messages('Q?', passages, {'prompt_version': 'inference'})
    assert strict[0]['content'] == RAG_SYSTEM_PROMPTS['strict']
    assert inference[0]['content'] == RAG_SYSTEM_PROMPTS['inference']
    assert '[rrf.md] RRF fuses ranks.' in strict[1]['content']
    # the direct pipeline (no passages) is unaffected by prompt_version
    assert build_messages('Q?', [], {'prompt_version': 'inference'})[0]['content'] == build_messages('Q?', [], {})[0]['content']
    with pytest.raises(ValueError, match='prompt_version'):
        prompt_version({'prompt_version': 'loose'})
