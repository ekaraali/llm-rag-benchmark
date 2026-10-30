"""Document chunking and BM25 / dense / hybrid retrieval for the benchmark.

Every chunk keeps a stable ID and the character span it covers in its source
document. Retrieval returns ranked *chunks*, so chunk-level metrics (Recall@K,
MRR, nDCG) can be computed; document-level views are derived from the chunks.
"""

import re
from pathlib import Path

from src.bm25 import BM25

METHODS = ('dense', 'bm25', 'hybrid')
RRF_K = 60  # standard rank-smoothing constant from Cormack et al. (2009)


def _word_windows(text, size, overlap):
    """Yield (start_char, end_char) spans of overlapping word windows."""
    if not (0 <= overlap < size):
        raise ValueError('Require 0 <= chunk overlap < chunk size')
    spans = [m.span() for m in re.finditer(r'\S+', text)]
    step = size - overlap
    for i in range(0, len(spans), step):
        window = spans[i:i + size]
        if window:
            yield window[0][0], window[-1][1]


def split_words(text, size=100, overlap=20):
    """Split text into overlapping *word* windows (not tokenizer tokens)."""
    return [' '.join(text[s:e].split()) for s, e in _word_windows(text, size, overlap)]


def chunk_document(doc_id, text, size=100, overlap=20):
    """Chunk one document; each chunk records its ID and character span.

    Chunk IDs (`<doc_id>#c<n>`) are only stable for a given size/overlap, so
    benchmark labels point at evidence *text*, and relevant chunks are derived
    from that text for whatever chunking configuration is being run.
    """
    return [{'chunk_id': f'{doc_id}#c{n}', 'doc': doc_id, 'chunk': n,
             'start': s, 'end': e, 'text': ' '.join(text[s:e].split())}
            for n, (s, e) in enumerate(_word_windows(text, size, overlap))]


def rrf_fuse(*rankings, k=RRF_K):
    """Reciprocal Rank Fusion of several rankings (lists of item indices).

    Each item scores sum(1 / (k + rank)) over the rankings it appears in.
    Rank positions are fused instead of raw scores, whose scales differ.
    Ties are broken by item index so results are deterministic.
    """
    fused = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda idx: (-fused[idx], idx))


class Retriever:
    """Index Markdown documents and rank their chunks with BM25, dense or hybrid search."""

    def __init__(self, folder, chunk_words, overlap_words, embedding_model=None,
                 bm25_k1=1.5, bm25_b=0.75):
        """Chunk every .md file and build the BM25 index; dense embeddings are lazy."""
        self.docs = {}
        self.chunks = []
        for path in sorted(Path(folder).glob('*.md')):
            text = path.read_text(encoding='utf-8')
            self.docs[path.name] = text
            self.chunks.extend(chunk_document(path.name, text, chunk_words, overlap_words))
        if not self.chunks:
            raise ValueError('Document folder must have .md files')
        self.bm25 = BM25([c['text'] for c in self.chunks], k1=bm25_k1, b=bm25_b)
        self.embedding_model = embedding_model
        self.embeddings = None

    def ensure_dense(self):
        """Lazily embed the corpus once, using CPU to avoid macOS MPS crashes.

        `embedding_model` may be a model name or an already loaded object with
        an `encode` method (handy for tests and for sharing one model).
        """
        if self.embeddings is None:
            if not hasattr(self.embedding_model, 'encode'):
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(
                    self.embedding_model or 'sentence-transformers/all-MiniLM-L6-v2', device='cpu')
            self.embeddings = self.embedding_model.encode(
                [c['text'] for c in self.chunks], normalize_embeddings=True, show_progress_bar=False)

    def _order(self, scores):
        """Indices sorted by descending score; ties keep corpus order."""
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    def rank(self, query, method):
        """Return every chunk index ranked for `query` by `method`."""
        if method not in METHODS:
            raise ValueError(f'Unknown retrieval method {method!r}; expected one of {METHODS}')
        bm25_order = self._order(self.bm25.scores(query)) if method in ('bm25', 'hybrid') else None
        if method == 'bm25':
            return bm25_order
        self.ensure_dense()
        dense_q = self.embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        dense_order = self._order((self.embeddings @ dense_q).tolist())
        if method == 'dense':
            return dense_order
        return rrf_fuse(dense_order, bm25_order)

    def search(self, query, k, method='dense'):
        """Return the top-k chunks (dicts) for `query`."""
        if k < 1:
            raise ValueError('k must be >= 1')
        return [self.chunks[i] for i in self.rank(query, method)[:k]]
