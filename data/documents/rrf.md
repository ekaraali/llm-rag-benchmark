# Reciprocal Rank Fusion
Reciprocal Rank Fusion (RRF) merges ranked lists from separate retrievers. Each candidate document receives a score equal to the sum of 1/(k + rank) across retrievers where it appears. RRF combines BM25 lexical and dense semantic retrieval without needing directly comparable scores. RRF is a rank fusion method, not a cross-encoder reranker.
