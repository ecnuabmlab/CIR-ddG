"""
RDE-Network v3: Train on full 6706 SKEMPI entries, complex-level 5-fold (seed 2026).
Fold assigned by complex_to_fold.json. No dedup — all entries (including repeated measurements) used.
Saves checkpoints + per-fold predictions + combined predictions.
"""
import os, sys, json, copy, math, random, pickle, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr

RDE_DIR = Path('/data/yuwl/paper1/CIR-DDG/baselines/RDE-Network')
sys.path.insert(0, str(RDE_DIR))
os.chdir(str(RDE_DIR))

from rde.utils.misc import load_config, seed_all
from rde.utils.train import sum_weighted_losses, recursive_to
from rde.models.rde_ddg import DDG_RDE_Network
from rde.datasets.skempi import SkempiDataset
from rde.utils.transforms import get_transform
from rde.utils.data import PaddingCollate

COMMON = Path('/data/yuwl/paper1/experiments_v3/common')
OUTPUT = Path('/data/yuwl/paper1/experiments_v3/00_base_retrain/rde')


class SkempiDatasetByComplex(SkempiDataset):
    """SkempiDataset with external complex-level fold assignment. No dedup."""

    def __init__(self, csv_path, pdb_dir, cache_dir, complex_to_fold,
                 cvfold_index=0, split='train', transform=None,
                 blocklist=frozenset({'1KBH'})):
        # Init parent partially — we override _load_entries
        from torch.utils.data import Dataset
        Dataset.__init__(self)
        self.csv_path = csv_path
        self.pdb_dir = pdb_dir
        self.pdb_wt_dir = pdb_dir
        self.pdb_mt_dir = None
        self.cache_dir = cache_dir
        self.complex_to_fold = complex_to_fold
        self.cvfold_index = cvfold_index
        self.split = split
        self.transform = transform
        self.blocklist = blocklist

        self.entries_cache = os.path.join(cache_dir, 'entries.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries_by_complex()

        self.structures_cache = os.path.join(cache_dir, 'structures.lmdb')
        self.structures = None
        self.db_conn = None
        self.db_keys = None
        self._load_structures(False)

    def _load_entries_by_complex(self):
        with open(self.entries_cache, 'rb') as f:
            self.entries_full = pickle.load(f)

        entries = []
        for e in self.entries_full:
            cplx = e['complex']
            if cplx not in self.complex_to_fold:
                continue
            entry_fold = self.complex_to_fold[cplx]
            if self.split == 'val' and entry_fold == self.cvfold_index:
                entries.append(e)
            elif self.split == 'train' and entry_fold != self.cvfold_index:
                entries.append(e)
        self.entries = entries


def train_fold(fold, config, complex_to_fold, device):
    transform = get_transform(config.data.transform)

    dataset_train = SkempiDatasetByComplex(
        csv_path=config.data.csv_path, pdb_dir=config.data.pdb_dir,
        cache_dir=config.data.cache_dir, complex_to_fold=complex_to_fold,
        cvfold_index=fold, split='train', transform=transform)
    dataset_val = SkempiDatasetByComplex(
        csv_path=config.data.csv_path, pdb_dir=config.data.pdb_dir,
        cache_dir=config.data.cache_dir, complex_to_fold=complex_to_fold,
        cvfold_index=fold, split='val', transform=transform)

    print(f'  Train: {len(dataset_train)}, Val: {len(dataset_val)}')

    collate_fn = PaddingCollate()
    train_loader = DataLoader(dataset_train, batch_size=config.train.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(dataset_val, batch_size=config.train.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Build model from config and initialize RDE encoder from pretrained weights
    model = DDG_RDE_Network(config.model).to(device)
    rde_ckpt = torch.load(str(RDE_DIR / 'trained_models' / 'RDE.pt'), map_location=device)
    model.rde.load_state_dict(rde_ckpt['model'], strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    max_iters = config.train.max_iters
    val_freq = config.train.val_freq
    train_iter = iter(train_loader)

    for it in tqdm(range(1, max_iters + 1), desc=f'Fold {fold}'):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        batch = recursive_to(batch, device)
        loss_dict, _ = model(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss.backward()
        clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if it % val_freq == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = recursive_to(vbatch, device)
                    vl_dict, _ = model(vbatch)
                    vl = sum_weighted_losses(vl_dict, config.train.loss_weights)
                    val_losses.append(vl.item())
            avg_vl = np.mean(val_losses)
            scheduler.step(avg_vl)

            if avg_vl < best_val_loss:
                best_val_loss = avg_vl
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f'  Early stopping at iter {it}')
                    break

    # Predict val with best model
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    results = []
    with torch.no_grad():
        for vbatch in val_loader:
            vbatch = recursive_to(vbatch, device)
            _, output_dict = model(vbatch)
            for cplx, mutstr, ddg_true, ddg_pred in zip(
                vbatch['complex'], vbatch['mutstr'],
                output_dict['ddG_true'], output_dict['ddG_pred']
            ):
                results.append({
                    'complex': cplx, 'mutstr': mutstr,
                    'ddG': ddg_true.item(), 'ddG_pred': ddg_pred.item(), 'fold': fold,
                })

    # Save checkpoint
    ckpt_path = OUTPUT / 'checkpoints' / f'fold{fold}_best.pt'
    torch.save({'model': best_state, 'fold': fold}, ckpt_path)
    print(f'  Saved checkpoint: {ckpt_path}')

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = args.device

    config, _ = load_config(str(RDE_DIR / 'configs' / 'train' / 'rde_ddg_skempi.yml'))
    seed_all(2026)

    with open(COMMON / 'complex_to_fold.json') as f:
        complex_to_fold = json.load(f)
    print(f'Loaded complex_to_fold: {len(complex_to_fold)} complexes')

    all_results = []
    for fold in range(5):
        print(f'\n--- Fold {fold}/5 ---')
        fold_df = train_fold(fold, config, complex_to_fold, device)
        fold_df.to_csv(OUTPUT / 'predictions' / f'fold{fold}_predictions.csv', index=False)
        all_results.append(fold_df)
        r = pearsonr(fold_df['ddG'], fold_df['ddG_pred'])[0]
        rho = spearmanr(fold_df['ddG'], fold_df['ddG_pred'])[0]
        print(f'  Fold {fold}: n={len(fold_df)}, R={r:.4f}, ρ={rho:.4f}')

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(OUTPUT / 'predictions' / 'all_predictions.csv', index=False)
    r = pearsonr(df_all['ddG'], df_all['ddG_pred'])[0]
    rho = spearmanr(df_all['ddG'], df_all['ddG_pred'])[0]
    print(f'\nOverall: R={r:.4f}, ρ={rho:.4f} (target ~0.632/0.527)')
    print(f'N entries: {len(df_all)}')


if __name__ == '__main__':
    main()
