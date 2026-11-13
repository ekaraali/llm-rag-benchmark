# Judge calibration: test split, prompt `rubric_v2`

Judge: `claude-haiku-5-5` via `anthropic`, thinking: `disabled`. 48 answers from 8 questions; 0 unparseable replies excluded.

| Metric | Value | 95% CI (question-level bootstrap) |
|---|---:|---:|
| Accuracy (exact label agreement) | 0.958 | |
| Cohen's κ | 0.938 | 0.805 – 1.000 |
| Weighted κ (linear) | 0.954 | 0.843 – 1.000 |
| Macro-F1 (human = reference) | 0.957 | |

Confusion matrix (rows = human, columns = judge):

| human \ judge | incorrect | partial | correct |
|---|---:|---:|---:|
| incorrect | 15 | 2 | 0 |
| partial | 0 | 14 | 0 |
| correct | 0 | 0 | 17 |

Rubric rule violations (hedged but labelled correct): 0

Answer status according to the judge:

| Pipeline | answered | hedged | refused |
|---|---:|---:|---:|
| dense_rag | 6 | 8 | 2 |
| direct | 6 | 6 | 4 |
| hybrid_reasoning_rag | 7 | 7 | 2 |

## Disagreements (2)

| Q | Pipeline | k | Human | Judge | Judge reason |
|---|---|---:|---|---|---|
| 15 | dense_rag | 3 | incorrect | partial | Says insufficient evidence but conveys the key point that temperature zero does not explicitly guarantee reproducibility for all hosted models; hedged, not stated outright. |
| 15 | hybrid_reasoning_rag | 3 | incorrect | partial | Says insufficient evidence but conveys the key point that temperature zero does not explicitly guarantee reproducibility for all hosted models; hedged, not stated outright. |

New API usage: 37149 input + 3893 output tokens, about $0.0057 (cached judgements cost nothing).
