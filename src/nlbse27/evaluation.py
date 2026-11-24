import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)

TASKS = ["Vulnerability", "SATD"]
GRID = np.round(np.arange(0.02, 0.96, 0.02), 2)


def tune_thresholds(probs, y):
    """Per-label threshold maximising F1 (use on validation data only)."""
    return [float(max(GRID, key=lambda t: f1_score(y[:, j], probs[:, j] > t, zero_division=0)))
            for j in range(2)]


def metrics(probs, y, ths):
    pred = (probs > np.array(ths)).astype(int)
    row = {}
    for j, t in enumerate(TASKS):
        row[f"{t} P"] = precision_score(y[:, j], pred[:, j], zero_division=0)
        row[f"{t} R"] = recall_score(y[:, j], pred[:, j], zero_division=0)
        row[f"{t} F1"] = f1_score(y[:, j], pred[:, j], zero_division=0)
        row[f"{t} macro-F1"] = f1_score(y[:, j], pred[:, j], average="macro", zero_division=0)
        row[f"{t} MCC"] = matthews_corrcoef(y[:, j], pred[:, j])
        row[f"{t} PR-AUC"] = average_precision_score(y[:, j], probs[:, j])
    row["Mean F1"] = (row["Vulnerability F1"] + row["SATD F1"]) / 2
    return row


def mcnemar_test(pa, pb, y):
    """McNemar test (continuity-corrected) on paired predictions of two models."""
    from statsmodels.stats.contingency_tables import mcnemar
    a_ok, b_ok = pa == y, pb == y
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    r = mcnemar([[0, b], [c, 0]], exact=False, correction=True)
    return b, c, float(r.statistic), float(r.pvalue)


def _fast_f1(y, p):
    tp = (y & p).sum(0)
    fp = (~y & p).sum(0)
    fn = (y & ~p).sum(0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def bootstrap_diff(pa, pb, y, n=1000, seed=0):
    """Mean and 95% CI of the difference in mean F1 (A - B) over bootstrap resamples."""
    rng = np.random.default_rng(seed)
    y, pa, pb = y.astype(bool), pa.astype(bool), pb.astype(bool)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(_fast_f1(y[i], pa[i]).mean() - _fast_f1(y[i], pb[i]).mean())
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
