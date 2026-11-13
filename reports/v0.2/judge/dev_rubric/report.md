# Judge calibration: dev split, prompt `rubric`

Judge: `claude-haiku-5-5` via `anthropic`, thinking: `disabled`. 44 answers from 8 questions; 4 unparseable replies excluded.

| Metric | Value | 95% CI (question-level bootstrap) |
|---|---:|---:|
| Accuracy (exact label agreement) | 0.864 | |
| Cohen's κ | 0.793 | 0.598 – 1.000 |
| Weighted κ (linear) | 0.852 | 0.673 – 1.000 |
| Macro-F1 (human = reference) | 0.861 | |

Confusion matrix (rows = human, columns = judge):

| human \ judge | incorrect | partial | correct |
|---|---:|---:|---:|
| incorrect | 14 | 0 | 0 |
| partial | 0 | 9 | 4 |
| correct | 0 | 2 | 15 |

Answer status according to the judge:

| Pipeline | answered | hedged | refused |
|---|---:|---:|---:|
| dense_rag | 12 | 1 | 1 |
| direct | 6 | 4 | 6 |
| hybrid_reasoning_rag | 10 | 4 | 0 |

## Disagreements (6)

| Q | Pipeline | k | Human | Judge | Judge reason |
|---|---|---:|---|---|---|
| 11 | direct | 3 | correct | partial | Mentions parameters, metrics, and results, but omits code versions and artifacts explicitly named in the reference, though the core idea is right. |
| 14 | direct | 3 | partial | correct | Answer states the tradeoff as increased computational cost (and training time), matching the reference's compute/latency tradeoff, though it hedges about uncertainty. |
| 7 | dense_rag | 3 | partial | correct | Temperature scaling the logits before sampling is the mechanism controlling randomness, matching the reference answer's core fact. |
| 7 | hybrid_reasoning_rag | 3 | partial | correct | Temperature scaling the logits before sampling is the mechanism controlling randomness, matching the reference answer's core fact. |
| 11 | direct | 5 | correct | partial | Mentions parameters, metrics, and results, but omits code versions and artifacts explicitly named in the reference, though the core idea is right. |
| 14 | direct | 5 | partial | correct | Answer states the tradeoff as increased computational cost (and training time), matching the reference's compute/latency tradeoff, though it hedges about uncertainty. |

New API usage: 15078 input + 2113 output tokens, about $0.0026 (cached judgements cost nothing).
