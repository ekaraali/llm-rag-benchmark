# Judge calibration: dev split, prompt `rubric_v2`

Judge: `claude-haiku-5-5` via `anthropic`, thinking: `disabled`. 48 answers from 8 questions; 0 unparseable replies excluded.

| Metric | Value | 95% CI (question-level bootstrap) |
|---|---:|---:|
| Accuracy (exact label agreement) | 0.917 | |
| Cohen's κ | 0.875 | 0.683 – 1.000 |
| Weighted κ (linear) | 0.905 | 0.755 – 1.000 |
| Macro-F1 (human = reference) | 0.918 | |

Confusion matrix (rows = human, columns = judge):

| human \ judge | incorrect | partial | correct |
|---|---:|---:|---:|
| incorrect | 14 | 0 | 0 |
| partial | 2 | 15 | 0 |
| correct | 0 | 2 | 15 |

Rubric rule violations (hedged but labelled correct): 0

Answer status according to the judge:

| Pipeline | answered | hedged | refused |
|---|---:|---:|---:|
| dense_rag | 13 | 3 | 0 |
| direct | 6 | 4 | 6 |
| hybrid_reasoning_rag | 12 | 3 | 1 |

## Disagreements (4)

| Q | Pipeline | k | Human | Judge | Judge reason |
|---|---|---:|---|---|---|
| 11 | direct | 3 | correct | partial | Mentions parameters, metrics, and results tracking, but omits code versions and artifacts; reference lists configurations, metrics, code versions, and artifacts. |
| 11 | direct | 5 | correct | partial | Mentions parameters, metrics, and results tracking, but omits code versions and artifacts; reference lists configurations, metrics, code versions, and artifacts. |
| 1 | dense_rag | 5 | partial | incorrect | Generated answer claims insufficient evidence and does not state the key point, though it mentions fragmenting multi-paragraph evidence only in passing. |
| 1 | hybrid_reasoning_rag | 5 | partial | incorrect | Generated answer claims insufficient evidence and does not state the key point, though it mentions fragmenting multi-paragraph evidence only in passing. |

New API usage: 32634 input + 3406 output tokens, about $0.0050 (cached judgements cost nothing).
