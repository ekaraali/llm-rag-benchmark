"""Agreement statistics between two raters (e.g. a human and an LLM judge).

All functions take two equal-length label lists. `categories` fixes the label
order (for ordinal labels such as incorrect < partial < correct it must be
given in that order for weighted kappa to make sense).
"""

import random


def confusion_matrix(a, b, categories):
    """Counts as rows = labels of rater a, columns = labels of rater b."""
    _check(a, b, categories)
    index = {c: i for i, c in enumerate(categories)}
    m = [[0] * len(categories) for _ in categories]
    for x, y in zip(a, b):
        m[index[x]][index[y]] += 1
    return m


def accuracy(a, b):
    """Share of items where both raters give the same label."""
    if len(a) != len(b) or not a:
        raise ValueError('Need two non-empty label lists of equal length')
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a, b, categories, weights=None):
    """Cohen's kappa: agreement corrected for the agreement expected by chance.

    kappa = 1 - sum(w * observed) / sum(w * expected), with disagreement weights w:
      weights=None      w = 0 on the diagonal, 1 elsewhere (unweighted kappa)
      weights='linear'  w = |i - j| / (n - 1), for ordinal categories
    1 = perfect agreement, 0 = chance level, < 0 = worse than chance.
    Returns 1.0 when both raters use one identical label for every item.
    """
    m = confusion_matrix(a, b, categories)
    n, k = len(a), len(categories)
    if weights is None:
        w = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    elif weights == 'linear':
        w = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    else:
        raise ValueError("weights must be None or 'linear'")
    rows = [sum(m[i]) / n for i in range(k)]
    cols = [sum(m[i][j] for i in range(k)) / n for j in range(k)]
    observed = sum(w[i][j] * m[i][j] / n for i in range(k) for j in range(k))
    expected = sum(w[i][j] * rows[i] * cols[j] for i in range(k) for j in range(k))
    return 1.0 if expected == 0 else 1 - observed / expected


def macro_f1(reference, predicted, categories):
    """Unweighted mean of per-class F1, treating `reference` as the ground truth.

    A class absent from both lists is skipped; a class predicted but never
    present (or vice versa) contributes F1 = 0.
    """
    m = confusion_matrix(reference, predicted, categories)
    scores = []
    for i in range(len(categories)):
        tp = m[i][i]
        fn = sum(m[i]) - tp
        fp = sum(m[r][i] for r in range(len(categories))) - tp
        if tp + fn + fp == 0:
            continue
        scores.append(2 * tp / (2 * tp + fn + fp))
    return sum(scores) / len(scores)


def cluster_bootstrap_ci(clusters, statistic, n_boot=2000, alpha=0.05, seed=0):
    """Percentile confidence interval of `statistic` by resampling whole clusters.

    `clusters` maps a cluster id (here: a question) to its list of items (here:
    that question's answers). Answers to the same question are correlated, so
    questions, not answers, are resampled with replacement. `statistic` takes a
    flat list of items and returns a number. Resamples where the statistic
    cannot be computed (it raises ValueError or ZeroDivisionError) are skipped.
    """
    ids = sorted(clusters)
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sample = [item for cid in (rng.choice(ids) for _ in ids) for item in clusters[cid]]
        try:
            values.append(statistic(sample))
        except (ValueError, ZeroDivisionError):
            continue
    if not values:
        raise ValueError('Statistic could not be computed on any bootstrap resample')
    values.sort()
    lo = values[int((alpha / 2) * (len(values) - 1))]
    hi = values[int((1 - alpha / 2) * (len(values) - 1))]
    return lo, hi


def _check(a, b, categories):
    if len(a) != len(b) or not a:
        raise ValueError('Need two non-empty label lists of equal length')
    unknown = (set(a) | set(b)) - set(categories)
    if unknown:
        raise ValueError(f'Labels outside categories: {sorted(unknown)}')
