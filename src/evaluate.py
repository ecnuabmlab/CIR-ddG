"""Evaluation utilities for ΔΔG prediction.

Computes standard metrics: per-structure Pearson/Spearman, overall correlation,
RMSE (linear-calibrated), MAE, and AUROC.
"""

import math
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
)


def per_structure_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    complexes: np.ndarray,
    min_rows: int = 10,
) -> dict:
    """Compute per-structure (per-complex) correlation metrics.

    Groups mutations by complex, computes Pearson/Spearman within each,
    and averages across complexes with >= min_rows mutations.

    Args:
        y_true: Ground truth ΔΔG values.
        y_pred: Predicted ΔΔG values.
        complexes: Complex identifiers for grouping.
        min_rows: Minimum mutations per complex to include.

    Returns:
        Dict with mean Pearson, Spearman, and number of valid complexes.
    """
    by_complex = defaultdict(list)
    for i in range(len(y_true)):
        if np.isfinite(y_pred[i]):
            by_complex[complexes[i]].append(i)

    pearson_values, spearman_values = [], []
    for cplx, indices in by_complex.items():
        if len(indices) < min_rows:
            continue
        yt = y_true[indices]
        yp = y_pred[indices]
        if np.std(yt) > 0 and np.std(yp) > 0:
            r_p, _ = pearsonr(yt, yp)
            r_s, _ = spearmanr(yt, yp)
            if np.isfinite(r_p):
                pearson_values.append(float(r_p))
            if np.isfinite(r_s):
                spearman_values.append(float(r_s))

    return {
        "pearson": float(np.mean(pearson_values)) if pearson_values else math.nan,
        "spearman": float(np.mean(spearman_values)) if spearman_values else math.nan,
        "n_complexes": len(spearman_values),
    }


def overall_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute overall (pooled) correlation metrics.

    Args:
        y_true: Ground truth ΔΔG values.
        y_pred: Predicted ΔΔG values.

    Returns:
        Dict with Pearson, Spearman, RMSE (calibrated), MAE, AUROC.
    """
    valid = np.isfinite(y_pred) & np.isfinite(y_true)
    yt = y_true[valid]
    yp = y_pred[valid]

    if len(yt) < 3:
        return {k: math.nan for k in ["pearson", "spearman", "rmse", "mae", "auroc"]}

    r_pearson, _ = pearsonr(yt, yp)
    r_spearman, _ = spearmanr(yt, yp)

    # Linear-calibrated RMSE/MAE
    lr = LinearRegression().fit(yp.reshape(-1, 1), yt)
    yp_cal = lr.predict(yp.reshape(-1, 1))
    rmse = float(np.sqrt(mean_squared_error(yt, yp_cal)))
    mae = float(mean_absolute_error(yt, yp_cal))

    # AUROC (binary: ddG < 0 = improved binding = positive class)
    labels = (yt < 0).astype(int)
    pos_frac = labels.mean()
    if 0 < pos_frac < 1:
        auroc = float(roc_auc_score(labels, -yp))
    else:
        auroc = math.nan

    return {
        "pearson": float(r_pearson),
        "spearman": float(r_spearman),
        "rmse": rmse,
        "mae": mae,
        "auroc": auroc,
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    complexes: np.ndarray,
    mask: np.ndarray = None,
    min_rows: int = 10,
) -> dict:
    """Full evaluation suite for ΔΔG predictions.

    Args:
        y_true: Ground truth ΔΔG.
        y_pred: Predicted ΔΔG.
        complexes: Complex identifiers.
        mask: Optional boolean mask to evaluate on a subset.
        min_rows: Minimum mutations per complex for per-structure metrics.

    Returns:
        Dict with per-structure and overall metrics.
    """
    if mask is not None:
        idx = np.where(mask)[0]
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        complexes = complexes[idx]

    return {
        "per_structure": per_structure_correlation(y_true, y_pred, complexes, min_rows),
        "overall": overall_correlation(y_true, y_pred),
        "n_entries": int(len(y_true)),
    }
