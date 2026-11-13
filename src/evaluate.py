"""Fast diagnostic metrics for generated answers; these are NOT human-verified scores."""

import re
import string
from collections import Counter

ABSTENTION_MARKER = 'insufficient evidence'  # phrase the RAG system prompts ask for


def normalize(s):
    """Normalize case, punctuation and whitespace for exact phrase checks."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w/()+%-]', ' ', s.lower())).strip()


def squad_tokens(text):
    """SQuAD-style answer normalisation: lowercase, drop punctuation and
    articles (a, an, the), split on whitespace."""
    text = ''.join(ch for ch in text.lower() if ch not in string.punctuation)
    return re.sub(r'\b(a|an|the)\b', ' ', text).split()


def token_f1(answer, reference):
    """SQuAD token-overlap F1 between an answer and the reference answer.

    Long generative answers get low precision even when correct, so this
    rewards concise answers; treat it as a standard baseline, not as correctness.
    """
    a, r = squad_tokens(answer), squad_tokens(reference)
    common = sum((Counter(a) & Counter(r)).values())
    if not a or not r or not common:
        return 0.0
    precision, recall = common / len(a), common / len(r)
    return 2 * precision * recall / (precision + recall)


def reference_token_recall(answer, reference):
    """Share of reference tokens that appear in the answer (SQuAD normalisation).

    Unlike F1 it does not penalise verbose answers, but it cannot see
    paraphrases or whether the tokens are used correctly.
    """
    a, r = Counter(squad_tokens(answer)), Counter(squad_tokens(reference))
    if not r:
        return 0.0
    return sum((a & r).values()) / sum(r.values())


def has_abstention_marker(answer):
    """True when the answer contains the abstention phrase anywhere.

    This cannot tell a real refusal from an answer that first says
    "Insufficient evidence." and then answers anyway (a hedged answer), so it
    is reported as a marker rate, not as a refusal rate.
    """
    return ABSTENTION_MARKER in answer.lower()


def evaluate_answer(answer, reference, relevant_docs, retrieved):
    """Compute weak string/token matches, document hit, citation precision and abstention marker.

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
        return {'reference_phrase_match': None, 'token_f1': None, 'reference_token_recall': None,
                'retrieval_hit': None, 'citation_precision': None, 'abstention_marker': marker}
    # Weak automated proxy, NOT semantic correctness or calibrated faithfulness.
    # Paraphrases and equivalent equations can fail this exact-substring test.
    token_match = normalize(reference) in normalize(answer)
    doc_hit = bool(set(relevant_docs) & {x['doc'] for x in retrieved})
    citation_precision = len(citations & set(relevant_docs)) / len(citations) if citations else 0.0
    return {'reference_phrase_match': int(token_match), 'token_f1': token_f1(answer, reference),
            'reference_token_recall': reference_token_recall(answer, reference), 'retrieval_hit': int(doc_hit),
            'citation_precision': citation_precision, 'abstention_marker': marker}
