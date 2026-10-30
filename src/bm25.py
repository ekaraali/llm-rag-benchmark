"""Okapi BM25 lexical scoring, implemented without extra dependencies.

The formula follows Robertson & Zaragoza (2009) with the non-negative IDF
variant used by Lucene/Elasticsearch:

    idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
    score(q, d) = sum over query terms t of
                  idf(t) * f(t, d) * (k1 + 1) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))

where N is the number of indexed passages, n_t the number of passages that
contain t, f(t, d) the term frequency of t in d, |d| the passage length in
tokens and avgdl the mean passage length. A query term repeated in the query
contributes once per occurrence, as in the classic formulation.
"""

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text):
    """Lowercase and split text into alphanumeric word tokens.

    Hyphenated and apostrophe-joined words stay whole ("top-p", "model's").
    No stemming or stop-word removal is applied, so results stay easy to audit.
    """
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Score a fixed set of passages against free-text queries with Okapi BM25."""

    def __init__(self, passages, k1=1.5, b=0.75):
        """Index passages (a list of strings). k1 controls term-frequency
        saturation; b controls document-length normalisation (0 = none)."""
        if not passages:
            raise ValueError('BM25 needs at least one passage')
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError('Require k1 >= 0 and 0 <= b <= 1')
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(p)) for p in passages]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avgdl = sum(self.lengths) / len(self.lengths) or 1.0
        doc_freq = Counter(term for tf in self.term_freqs for term in tf)
        n = len(passages)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def scores(self, query):
        """Return one BM25 score per indexed passage, in index order."""
        terms = tokenize(query)
        out = []
        for tf, length in zip(self.term_freqs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out
