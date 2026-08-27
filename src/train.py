"""Training script for CIR-DDG residual module.

Trains the cross-chain interaction residual on top of any backbone's
base predictions, using protein-level cross-validation.
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, wilcoxon

from model import CrossChainResidual


def seed_all(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def protein_level_split(pdbcodes: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign fold IDs by protein complex (no same-complex leakage).

    All mutations from the same PDB structure are placed in the same fold.

    Args:
        pdbcodes: Array of PDB codes for each entry.
        n_folds: Number of cross-validation folds.
        seed: Random seed for shuffling.

    Returns:
        Array of fold assignments (0 to n_folds-1).
    """
    unique_pdbs = sorted(set(pdbcodes.tolist()))
    random.Random(seed).shuffle(unique_pdbs)
    fold_assign = np.zeros(len(pdbcodes), dtype=int)
    chunk_size = math.ceil(len(unique_pdbs) / n_folds)
    pdb_to_fold = {}
    for fi in range(n_folds):
        for pdb in unique_pdbs[fi * chunk_size: min((fi + 1) * chunk_size, len(unique_pdbs))]:
            pdb_to_fold[pdb] = fi
    for i, pdb in enumerate(pdbcodes):
        fold_assign[i] = pdb_to_fold[pdb]
    return fold_assign


def train_residual(
    base_predictions: np.ndarray,
    geometry: np.ndarray,
    targets: np.ndarray,
    interface_mask: np.ndarray,
    complexes: np.ndarray,
    pdbcodes: np.ndarray,
    seed: int,
    device: torch.device,
    n_folds: int = 3,
    max_epochs: int = 300,
    patience: int = 60,
    lr: float = 5e-4,
) -> np.ndarray:
    """Train CCIR module and return CV predictions.

    Args:
        base_predictions: Backbone's ΔΔG predictions, shape (N,).
        geometry: Raw 22d geometry features, shape (N, 22).
        targets: Ground truth ΔΔG, shape (N,).
        interface_mask: Binary mask for interface mutations, shape (N,).
        complexes: Complex identifiers for per-structure evaluation.
        pdbcodes: PDB codes for protein-level splitting.
        seed: Random seed.
        device: Torch device.
        n_folds: Number of CV folds for residual training.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.
        lr: Learning rate.

    Returns:
        CV predictions (base + residual correction), shape (N,).
    """
    seed_all(seed)
    folds = protein_level_split(pdbcodes, n_folds, seed)
    cv_pred = np.copy(base_predictions)

    for fold in range(n_folds):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        # Standardize geometry using training statistics
        z_mu = geometry[train_idx].mean(0)
        z_std = geometry[train_idx].std(0) + 1e-6

        # Prepare tensors
        tz = torch.from_numpy(((geometry[train_idx] - z_mu) / z_std).astype(np.float32)).to(device)
        tb = torch.from_numpy(base_predictions[train_idx]).to(device)
        ty = torch.from_numpy(targets[train_idx]).to(device)
        tm = torch.from_numpy(interface_mask[train_idx]).float().to(device)
        vz = torch.from_numpy(((geometry[val_idx] - z_mu) / z_std).astype(np.float32)).to(device)
        vb = torch.from_numpy(base_predictions[val_idx]).to(device)
        vm = torch.from_numpy(interface_mask[val_idx]).float().to(device)

        # Train
        model = CrossChainResidual().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best_score, best_state, bad_epochs = -1e9, None, 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model(tb, tz, tm, antisymmetry_sign=1.0)
            loss = F.mse_loss(pred, ty)
            # Focus loss on interface entries
            if tm.any():
                loss = loss + F.mse_loss(pred[tm.bool()], ty[tm.bool()])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Evaluate
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    vp = model(vb, vz, vm, antisymmetry_sign=1.0)
                val_pred = vp.cpu().numpy()
                score = _per_structure_spearman(
                    val_pred, targets[val_idx], complexes[val_idx], interface_mask[val_idx], min_rows=3
                )
                if score > best_score + 1e-4:
                    best_score = score
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 5
                    if bad_epochs >= patience:
                        break

        # Predict validation set with best model
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vp = model(vb, vz, vm, antisymmetry_sign=1.0)
        cv_pred[val_idx] = vp.cpu().numpy()

    return cv_pred


def _per_structure_spearman(
    pred: np.ndarray, y: np.ndarray, complexes: np.ndarray,
    mask: np.ndarray, min_rows: int,
) -> float:
    """Compute mean per-structure Spearman on masked subset."""
    rhos = []
    valid = np.isfinite(pred) & mask.astype(bool)
    for cplx in set(complexes[valid]):
        idx = np.where((complexes == cplx) & valid)[0]
        if len(idx) < min_rows:
            continue
        if np.std(y[idx]) > 0 and np.std(pred[idx]) > 0:
            r, _ = spearmanr(y[idx], pred[idx])
            if np.isfinite(r):
                rhos.append(float(r))
    return float(np.mean(rhos)) if rhos else -1.0


def main():
    parser = argparse.ArgumentParser(description="Train CIR-DDG residual module")
    parser.add_argument("--base-predictions", type=str, required=True,
                        help="Path to .npz with keys: keys, ddg, ddg_pred, complexes")
    parser.add_argument("--geometry", type=str, required=True,
                        help="Path to geometry features .npz (keys, geom)")
    parser.add_argument("--interface-mask", type=str, default=None,
                        help="Path to interface mask .npy (optional, auto-detect if not given)")
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    base_data = np.load(args.base_predictions, allow_pickle=True)
    geom_data = np.load(args.geometry, allow_pickle=True)

    # Align keys
    base_idx = {str(k): i for i, k in enumerate(base_data["keys"])}
    geom_idx = {str(k): i for i, k in enumerate(geom_data["keys"])}
    common = [k for k in base_data["keys"] if k in geom_idx]

    bp = np.array([base_data["ddg_pred"][base_idx[k]] for k in common], np.float32)
    z = np.array([geom_data["geom"][geom_idx[k]] for k in common], np.float32)
    y = np.array([base_data["ddg"][base_idx[k]] for k in common], np.float32)
    cx = np.array([base_data["complexes"][base_idx[k]] for k in common])
    pdb = np.array([k.split("_")[0] for k in common])

    # Interface mask: use provided or auto-detect (min_dist <= 5Å)
    if args.interface_mask:
        mask = np.load(args.interface_mask)
    else:
        mask = (z[:, 0] > 0) & (z[:, 0] <= 5.0)  # geom[0] = min_dist

    print(f"Entries: {len(common)}, Interface: {mask.sum()}")

    # Run multi-seed evaluation
    results = []
    for seed in args.seeds:
        cv = train_residual(bp, z, y, mask, cx, pdb, seed, device)
        # Evaluate
        score_all = _per_structure_spearman(cv, y, cx, np.ones(len(y), bool), 10)
        score_int = _per_structure_spearman(cv, y, cx, mask, 10)
        results.append({"seed": seed, "all_pc_spearman": score_all, "interface_pc_spearman": score_int})
        print(f"  Seed {seed}: all={score_all:.4f}, interface={score_int:.4f}")

    # Summary
    base_score = _per_structure_spearman(bp, y, cx, mask, 10)
    res_scores = np.array([r["interface_pc_spearman"] for r in results])
    delta = res_scores.mean() - base_score
    _, p_val = wilcoxon(res_scores - base_score, alternative="greater")

    summary = {
        "base_interface_pc_spearman": float(base_score),
        "residual_interface_pc_spearman_mean": float(res_scores.mean()),
        "residual_interface_pc_spearman_std": float(res_scores.std()),
        "delta": float(delta),
        "p_value": float(p_val),
        "seeds": results,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"\nBase: {base_score:.4f} → +Residual: {res_scores.mean():.4f} (Δ={delta:+.4f}, p={p_val:.4f})")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
