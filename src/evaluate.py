"""Fast diagnostic metrics; these are NOT human-verified answer scores."""

import re


def normalize(s):
    """Normalize case, punctuation and whitespace for exact phrase checks."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w/()+%-]', ' ', s.lower())).strip()


def evaluate_answer(answer, reference, relevant_docs, retrieved):
    """Compute weak string match, document hit and citation precision proxies.

    The hit metric checks whether *any* relevant source document was returned.
    Citation precision compares cited filenames with labeled relevant documents;
    it does not prove that the cited passage supports the answer.
    """
    # Weak automated proxy, NOT semantic correctness or calibrated faithfulness.
    # Paraphrases and equivalent equations can fail this exact-substring test.
    token_match = normalize(reference) in normalize(answer)
    doc_hit = bool(set(relevant_docs) & {x['doc'] for x in retrieved})
    citations = set(re.findall(r'\[([^\[\]]+\.md)\]', answer))
    citation_precision = len(citations & set(relevant_docs)) / len(citations) if citations else 0.0
    return {'reference_phrase_match': int(token_match), 'retrieval_hit': int(doc_hit),
            'citation_precision': citation_precision}
