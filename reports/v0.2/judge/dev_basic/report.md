# Judge calibration: dev split, prompt `basic`

Judge: `claude-haiku-5-5` via `anthropic`, thinking: `disabled`. 48 answers from 8 questions; 0 unparseable replies excluded.

| Metric | Value | 95% CI (question-level bootstrap) |
|---|---:|---:|
| Accuracy (exact label agreement) | 0.854 | |
| Cohen's κ | 0.780 | 0.443 – 1.000 |
| Weighted κ (linear) | 0.836 | 0.562 – 1.000 |
| Macro-F1 (human = reference) | 0.862 | |

Confusion matrix (rows = human, columns = judge):

| human \ judge | incorrect | partial | correct |
|---|---:|---:|---:|
| incorrect | 14 | 0 | 0 |
| partial | 0 | 12 | 5 |
| correct | 0 | 2 | 15 |

## Disagreements (7)

| Q | Pipeline | k | Human | Judge | Judge reason |
|---|---|---:|---|---|---|
| 11 | direct | 3 | correct | partial |  |
| 14 | direct | 3 | partial | correct |  |
| 14 | dense_rag | 3 | partial | correct |  |
| 14 | hybrid_reasoning_rag | 3 | partial | correct |  |
| 11 | direct | 5 | correct | partial |  |
| 14 | direct | 5 | partial | correct |  |
| 14 | hybrid_reasoning_rag | 5 | partial | correct |  |

New API usage: 7126 input + 324 output tokens, about $0.0009 (cached judgements cost nothing).
