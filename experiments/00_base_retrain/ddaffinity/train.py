"""
DDAffinity retraining on unified per-mutation 5-fold split.

DDAffinity uses ProteinMPNN-style encoder + ddG readout head.
Training: MSE loss, warm-up optimizer, 300 epochs, early stopping at 250.

Key: DDAffinity uses LMDB-cached structures (wt + mt PDBs).
Its split is also protein-level (complex-level) by default.
We inject our per-mutation fold assignments.
"""

import os
import sys
import json
import copy
import math
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# Paths
DDA_DIR = Path(__file__).parent.parent.parent.parent / "CIR-DDG" / "baselines" / "DDAffinity"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
COMMON_DIR = PROJECT_ROOT / "experiments_v3" / "common"
OUTPUT_DIR = PROJECT_ROOT / "experiments_v3" / "00_base_retrain" / "ddaffinity"

sys.path.insert(0, str(DDA_DIR))
os.chdir(str(DDA_DIR))

from rde.utils.misc import load_config, seed_all
from rde.models.protein_mpnn_network_2 import ProteinMPNN_NET
from rde.utils.skempi_mpnn import SkempiDatasetManager
from rde.datasets.skempi_parallel import SkempiDataset_lmdb, load_skempi_entries
from rde.utils.transforms import get_transform
from rde.utils.data_skempi_mpnn import PaddingCollate
from rde.utils.train_mpnn import CrossValidation, recursive_to, sum_weighted_losses, ScalarMetricAccumulator
from rde.utils.early_stopping import EarlyStopping

sys.path.insert(0, str(PROJECT_ROOT))


class SkempiDatasetCustomSplit(SkempiDataset_lmdb):
    """
    SkempiDataset_lmdb with external per-mutation fold assignments.
    Overrides the complex-level split with per-mutation assignment.
    """

    def __init__(self, csv_path, pdb_wt_dir, pdb_mt_dir, cache_dir,
                 fold_assignments, cvfold_index=0, split='train',
                 transform=None, blocklist=frozenset({'1KBH'}),
                 is_single=2):
        # We need to call the parent but override the split logic
        # Call Dataset.__init__ directly and replicate setup
        Dataset.__init__(self)
        self.csv_path = csv_path
        self.pdb_wt_dir = pdb_wt_dir
        self.pdb_mt_dir = pdb_mt_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.blocklist = blocklist
        self.transform = transform
        self.cvfold_index = cvfold_index
        self.num_cvfolds = 5
        self.split = split
        self.split_seed = 2026
        self.is_single = is_single
        self.fold_assignments = fold_assignments

        # Load entries (reuse parent's cache)
        self.entries_cache = os.path.join(cache_dir, 'entries.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries_custom()

        # Load structures (reuse parent's LMDB)
        self.structures_cache = os.path.join(cache_dir, 'structures.lmdb')
        self.structures = None
        self.db_conn = None
        self.db_keys = None
        self._load_structures(False)

    def _load_entries_custom(self):
        if not os.path.exists(self.entries_cache):
            self.entries_full = load_skempi_entries(
                self.csv_path, self.pdb_wt_dir, self.pdb_mt_dir, self.blocklist
            )
            with open(self.entries_cache, 'wb') as f:
                pickle.dump(self.entries_full, f)
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        # Filter by is_single
        if self.is_single == 0:
            self.entries_full = [e for e in self.entries_full if e['num_muts'] == 1]
        elif self.is_single == 1:
            self.entries_full = [e for e in self.entries_full if e['num_muts'] > 1]

        # Apply complex-level fold assignment (NO dedup, all entries used)
        # DDAffinity's complex field = PDB code (e.g. '1CSE')
        # complex_to_fold uses RDE format (e.g. '1CSE_E_I')
        # Build pdb_to_fold mapping
        pdb_to_fold = {}
        for cplx, fold in self.fold_assignments.items():
            pdb = cplx.split('_')[0]
            if pdb not in pdb_to_fold:
                pdb_to_fold[pdb] = fold

        entries = []
        for e in self.entries_full:
            pdb = e['complex']  # DDAffinity complex = just PDB code
            if pdb not in pdb_to_fold:
                continue
            entry_fold = pdb_to_fold[pdb]
            if self.split == 'val' and entry_fold == self.cvfold_index:
                entries.append(e)
            elif self.split == 'train' and entry_fold != self.cvfold_index:
                entries.append(e)

        self.entries = entries


def train_fold(fold, config, fold_assignments, device):
    """Train a single fold of DDAffinity."""
    transform_train = get_transform(config.data.train.transform)
    transform_val = get_transform(config.data.val.transform)

    train_dataset = SkempiDatasetCustomSplit(
        csv_path=config.data.csv_path,
        pdb_wt_dir=config.data.pdb_wt_dir,
        pdb_mt_dir=config.data.pdb_mt_dir,
        cache_dir=config.data.cache_dir,
        fold_assignments=fold_assignments,
        cvfold_index=fold, split='train',
        transform=transform_train,
        is_single=config.data.is_single,
    )
    val_dataset = SkempiDatasetCustomSplit(
        csv_path=config.data.csv_path,
        pdb_wt_dir=config.data.pdb_wt_dir,
        pdb_mt_dir=config.data.pdb_mt_dir,
        cache_dir=config.data.cache_dir,
        fold_assignments=fold_assignments,
        cvfold_index=fold, split='val',
        transform=transform_val,
        is_single=config.data.is_single,
    )

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    patch_size = config.data.val.transform[1].patch_size
    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch_size,
        shuffle=True, collate_fn=PaddingCollate(patch_size), num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.train.batch_size,
        shuffle=False, collate_fn=PaddingCollate(patch_size), num_workers=2
    )

    # Build model
    model = ProteinMPNN_NET(config.model).to(device)

    # Warm-up optimizer (as in DDAffinity paper)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.999))

    # Warm-up scheduler helper
    d_model = config.train.optimizer.d_model
    warmup = config.train.optimizer.warmup
    step_num = [0]

    def update_lr():
        step_num[0] += 1
        lr = (d_model ** -0.5) * min(step_num[0] ** -0.5, step_num[0] * warmup ** -1.5)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    early_stopping = EarlyStopping(
        save_path=str(OUTPUT_DIR / 'checkpoints'),
        patience=10, delta=0.001
    )

    best_results = None
    for epoch in range(config.train.max_epochs):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = recursive_to(batch, device)
            loss_dict, _ = model(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss.backward()
            clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            update_lr()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        # Validate every val_freq epochs
        if epoch % config.train.val_freq == 0:
            model.eval()
            results = []
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = recursive_to(batch, device)
                    loss_dict, output_dict = model(batch)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    val_losses.append(loss.item())
                    for cplx, mutstr, ddg_true, ddg_pred in zip(
                        batch['wt']['complex'], batch['wt']['mutstr'],
                        output_dict['ddG_true'], output_dict['ddG_pred']
                    ):
                        results.append({
                            'complex': cplx, 'mutstr': mutstr,
                            'ddG': ddg_true.item(), 'ddG_pred': ddg_pred.item()
                        })

            avg_val_loss = np.mean(val_losses)
            best_results = pd.DataFrame(results)

            if epoch % 50 == 0:
                r, _ = pearsonr(best_results['ddG'], best_results['ddG_pred'])
                rho, _ = spearmanr(best_results['ddG'], best_results['ddG_pred'])
                print(f"    Epoch {epoch}: train_loss={np.mean(train_losses):.4f} "
                      f"val_loss={avg_val_loss:.4f} R={r:.4f} ρ={rho:.4f}")

            # Early stopping after threshold epoch
            if epoch >= config.train.early_stopping_epoch:
                early_stopping(avg_val_loss, model, fold)
                if early_stopping.early_stop:
                    print(f"    Early stopping at epoch {epoch}")
                    break

    return best_results


def run_phase_b(device):
    """Train on full 6706 entries, complex-level 5-fold split."""
    print("=" * 60)
    print("DDAffinity v3: complex-level 5-fold, full 6706 entries")
    print("=" * 60)

    config, _ = load_config(str(DDA_DIR / 'configs' / 'train' / 'mpnn_ddg.yml'))
    seed_all(2026)

    with open(COMMON_DIR / "complex_to_fold.json") as f:
        complex_to_fold = json.load(f)
    print(f"Loaded complex_to_fold: {len(complex_to_fold)} complexes")

    all_results = []
    for fold in range(5):
        print(f"\n--- Fold {fold}/5 ---")
        fold_preds = train_fold(fold, config, complex_to_fold, device)
        if fold_preds is not None:
            fold_preds['fold'] = fold
            pdb_col = fold_preds['complex'].str.split('_').str[0]
            fold_preds['key'] = pdb_col + '_' + fold_preds['mutstr']
            fold_preds.to_csv(OUTPUT_DIR / 'predictions' / f'fold{fold}_predictions.csv', index=False)
            all_results.append(fold_preds)

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(OUTPUT_DIR / 'predictions' / 'phase_b_all_predictions.csv', index=False)

    r, _ = pearsonr(df_all['ddG'], df_all['ddG_pred'])
    rho, _ = spearmanr(df_all['ddG'], df_all['ddG_pred'])
    print(f"\nPhase B Final:")
    print(f"  Overall Pearson={r:.4f} (paper≈0.658)")
    print(f"  Overall Spearman={rho:.4f} (paper≈0.522)")

    return df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, choices=['A', 'B'], default='B')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    if args.phase == 'B':
        run_phase_b(args.device)
