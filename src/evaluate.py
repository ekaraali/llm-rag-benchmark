"""Fast diagnostic metrics for generated answers; these are NOT human-verified scores."""

import re

ABSTENTION_MARKER = 'insufficient evidence'  # phrase the RAG system prompts ask for


def normalize(s):
    """Normalize case, punctuation and whitespace for exact phrase checks."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w/()+%-]', ' ', s.lower())).strip()


def has_abstention_marker(answer):
    """True when the answer contains the abstention phrase anywhere.

    This cannot tell a real refusal from an answer that first says
    "Insufficient evidence." and then answers anyway (a hedged answer), so it
    is reported as a marker rate, not as a refusal rate.
    """
    return ABSTENTION_MARKER in answer.lower()


def evaluate_answer(answer, reference, relevant_docs, retrieved):
    """Compute weak string match, document hit, citation precision and abstention marker.

    The hit metric checks whether *any* relevant source document was returned.
    Citation precision compares cited filenames with labelled relevant documents;
    it does not prove that the cited passage supports the answer.
    For unanswerable questions (reference is None) the answer-matching and
    retrieval metrics are not defined and are returned as None.
    `abstention_marker` is 1 when the answer contains the abstention phrase.
    """
    marker = int(has_abstention_marker(answer))
    citations = set(re.findall(r'\[([^\[\]]+\.md)\]', answer))
    if reference is None:
        return {'reference_phrase_match': None, 'retrieval_hit': None,
                'citation_precision': None, 'abstention_marker': marker}
    # Weak automated proxy, NOT semantic correctness or calibrated faithfulness.
    # Paraphrases and equivalent equations can fail this exact-substring test.
    token_match = normalize(reference) in normalize(answer)
    doc_hit = bool(set(relevant_docs) & {x['doc'] for x in retrieved})
    citation_precision = len(citations & set(relevant_docs)) / len(citations) if citations else 0.0
    return {'reference_phrase_match': int(token_match), 'retrieval_hit': int(doc_hit),
            'citation_precision': citation_precision, 'abstention_marker': marker}
