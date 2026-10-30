"""Fast diagnostic metrics for generated answers; these are NOT human-verified scores."""

import re

ABSTENTION_MARKER = 'insufficient evidence'  # phrase the RAG system prompt asks for


def normalize(s):
    """Normalize case, punctuation and whitespace for exact phrase checks."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w/()+%-]', ' ', s.lower())).strip()


def is_abstention(answer):
    """True when the answer uses the system prompt's abstention phrase."""
    return ABSTENTION_MARKER in answer.lower()


def evaluate_answer(answer, reference, relevant_docs, retrieved):
    """Compute weak string match, document hit, citation precision and abstention.

    The hit metric checks whether *any* relevant source document was returned.
    Citation precision compares cited filenames with labelled relevant documents;
    it does not prove that the cited passage supports the answer.
    For unanswerable questions (reference is None) the answer-matching and
    retrieval metrics are not defined and are returned as None; `abstained`
    then tells whether the system correctly declined to answer.
    """
    abstained = int(is_abstention(answer))
    citations = set(re.findall(r'\[([^\[\]]+\.md)\]', answer))
    if reference is None:
        return {'reference_phrase_match': None, 'retrieval_hit': None,
                'citation_precision': None, 'abstained': abstained}
    # Weak automated proxy, NOT semantic correctness or calibrated faithfulness.
    # Paraphrases and equivalent equations can fail this exact-substring test.
    token_match = normalize(reference) in normalize(answer)
    doc_hit = bool(set(relevant_docs) & {x['doc'] for x in retrieved})
    citation_precision = len(citations & set(relevant_docs)) / len(citations) if citations else 0.0
    return {'reference_phrase_match': int(token_match), 'retrieval_hit': int(doc_hit),
            'citation_precision': citation_precision, 'abstained': abstained}
