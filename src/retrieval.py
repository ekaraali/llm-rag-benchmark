"""Document chunking and lexical/dense retrieval for the benchmark.

Retrieval deliberately returns one passage per source document so the existing
benchmark can report *document-level* hits; it does not measure chunk recall.
"""

import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_words(text, size=100, overlap=20):
    """Split text into overlapping *word* windows (not tokenizer tokens)."""
    if not (0 <= overlap < size):
        raise ValueError('Require 0 <= chunk overlap < chunk size')
    words = text.split()
    # Move by fewer than `size` words so adjacent windows share context.
    step = size - overlap
    return [' '.join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


class Retriever:
    """Index Markdown documents and search using lexical or dense similarity."""
    def __init__(self, folder, chunk_words, overlap_words, embedding_model=None):
        """Prepare chunks and TF-IDF; defer dense embedding work until requested."""
        self.chunks = []
        for path in sorted(Path(folder).glob('*.md')):
            for n, passage in enumerate(split_words(path.read_text(encoding='utf-8'), chunk_words, overlap_words)):
                self.chunks.append({'doc': path.name, 'chunk': n, 'text': passage})
        if not self.chunks:
            raise ValueError('Document folder must have .md files')
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.lexical = self.vectorizer.fit_transform([x['text'] for x in self.chunks])
        self.embedding_model = embedding_model
        self.embeddings = None

    def ensure_dense(self):
        """Lazily embed the corpus once, using CPU to avoid macOS MPS crashes."""
        if self.embeddings is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model or 'sentence-transformers/all-MiniLM-L6-v2', device='cpu')
            self.embeddings = self.embedding_model.encode([x['text'] for x in self.chunks], normalize_embeddings=True, show_progress_bar=False)

    def search(self, query, k, hybrid=False, lexical_only=False):
        """Return top-k unique documents using dense, TF-IDF, or RRF ranking.

        `hybrid=True` combines dense and lexical ranks, rather than blending
        scores whose numeric scales are not directly comparable.
        """
        import numpy as np
        lex = cosine_similarity(self.vectorizer.transform([query]), self.lexical)[0]
        if lexical_only:
            ranked = np.argsort(-lex).tolist()
        else:
            self.ensure_dense()
            dense_q = self.embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            sim = self.embeddings @ dense_q
            dense_order = np.argsort(-sim).tolist()
            if hybrid:
                lexical_order = np.argsort(-lex).tolist()
                # Reciprocal Rank Fusion rewards documents ranked highly by
                # either search method; 60 is a standard rank-smoothing constant.
                rrf = {}
                for order in (dense_order, lexical_order):
                    for rank, idx in enumerate(order, start=1):
                        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)
                ranked = sorted(rrf, key=lambda idx: (-rrf[idx], idx))
            else:
                ranked = dense_order
        # Unique source docs: benchmark measures document-level retrieval.
        hits, seen = [], set()
        for idx in ranked:
            item = self.chunks[idx]
            if item['doc'] not in seen:
                hits.append(item)
                seen.add(item['doc'])
            if len(hits) == k:
                break
        return hits


def decompose_query(question):
    """Expand simple compound queries via a transparent text-splitting heuristic."""
    # A deterministic, transparent query-expansion heuristic, not chain-of-thought.
    pieces = re.split(r'\s+and\s+|\s+versus\s+', question, flags=re.I)
    return [question] + ([p.strip(' ?') for p in pieces if len(p.split()) >= 3] if len(pieces) > 1 else [])
