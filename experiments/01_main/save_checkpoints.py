"""
Save CIR-DDG checkpoints for all 6 backbones × 5 folds.

Reproduces the exact training from Step 01 (generate_tables.py) and saves:
  - Model state_dict
  - z_mu, z_std (geometry normalization from training fold)

Checkpoints are saved to:
  experiments/01_main/checkpoints/{backbone}/fold{i}.pt

Each checkpoint contains:
  - model_state_dict
  - z_mu, z_std
  - config
  - train_info (n_entries, n_interface, fold)
"""

import sys
import os
import json
import math
import random
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, '/data/yuwl/paper1/CIR-DDG/cir_ddg/src')
from model import CrossChainResidual

PROJECT = Path('/data/yuwl/paper1')
COMMON = PROJECT / 'experiments' / 'common'
BASE_DIR = PROJECT / 'experiments' / '00_base_retrain'
OUTPUT = PROJECT / 'experiments' / '01_main'
GEOM_PATH = PROJECT / 'CIR-DDG' / 'cir_ddg' / 'data' / 'interface_geometry_all.npz'
ANNOT_PATH = PROJECT / 'CIR-DDG' / 'cir_ddg' / 'data' / 'd5b2_all_skempi_annotated_zero_shot.csv'

MODEL_ORDER = ['ProteinMPNN', 'ESM-IF', 'RDE-Network', 'DiffAffinity', 'DDAffinity', 'Vanilla Pythia-PPI']
BASE_PATHS = {
    'ProteinMPNN': BASE_DIR / 'proteinmpnn/predictions/predictions.csv',
    'ESM-IF': BASE_DIR / 'esmif/predictions/predictions.csv',
    'RDE-Network': BASE_DIR / 'rde/predictions/all_predictions.csv',
    'DiffAffinity': BASE_DIR / 'diffaffinity/predictions/predictions.csv',
    'DDAffinity': BASE_DIR / 'ddaffinity/predictions/phase_b_all_predictions.csv',
    'Vanilla Pythia-PPI': BASE_DIR / 'pythia/predictions/predictions.csv',
}

CIR_CONFIG = {
    'seed': 2026, 'n_runs': 1, 'n_outer_folds': 5, 'n_inner_folds': 3,
    'max_epochs': 150, 'patience': 30, 'lr': 5e-4,
    'weight_decay': 1e-4, 'hidden_dim': 64, 'dropout': 0.1,
    'pearson_weight': 0.2, 'l1_weight': 0.0,
    'pearson_guard': 3.0, 'grad_clip': 1.0,
}


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def complex_level_inner_split(complexes, n_folds, seed):
    unique = sorted(set(complexes.tolist()))
    random.Random(seed).shuffle(unique)
    c2f = {}
    chunk = math.ceil(len(unique) / n_folds)
    for fi in range(n_folds):
        for c in unique[fi * chunk:(fi + 1) * chunk]:
            c2f[c] = fi
    return np.array([c2f[c] for c in complexes])

def pearson_loss(pred, target):
    vx = pred - pred.mean()
    vy = target - target.mean()
    return 1.0 - (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)


def train_and_save_one_fold(train_base, train_geom, train_y, train_mask, train_cplx,
                             config, device, run_seed):
    """Train CIR-DDG on one fold's training data. Returns (model, z_mu, z_std)."""
    seed_all(run_seed)
    inner_folds = complex_level_inner_split(train_cplx, config['n_inner_folds'], run_seed)
    z_mu = train_geom.mean(0)
    z_std = train_geom.std(0) + 1e-6
    train_z = (train_geom - z_mu) / z_std

    it = inner_folds != 0; iv = inner_folds == 0
    tz = torch.from_numpy(train_z[it].astype(np.float32)).to(device)
    tb = torch.from_numpy(train_base[it].astype(np.float32)).to(device)
    ty = torch.from_numpy(train_y[it].astype(np.float32)).to(device)
    tm = torch.from_numpy(train_mask[it].astype(np.float32)).to(device)
    ivz = torch.from_numpy(train_z[iv].astype(np.float32)).to(device)
    ivb = torch.from_numpy(train_base[iv].astype(np.float32)).to(device)
    ivy = train_y[iv]; ivm = train_mask[iv]

    model = CrossChainResidual(22, config['hidden_dim'], config['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    best_score, best_state, bad = -1e9, None, 0

    for ep in range(1, config['max_epochs'] + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        p = model(tb, tz, tm, 1.0)
        loss = F.mse_loss(p, ty)
        if tm.any():
            loss = loss + F.mse_loss(p[tm.bool()], ty[tm.bool()])
        pw = config.get('pearson_weight', 0.0)
        if pw > 0:
            loss = loss + pw * pearson_loss(p, ty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
        opt.step()

        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                ivp = model(ivb, ivz, torch.from_numpy(ivm.astype(np.float32)).to(device), 1.0).cpu().numpy()
            masked = ivm.astype(bool)
            if masked.sum() >= 3 and np.std(ivy[masked]) > 1e-10 and np.std(ivp[masked]) > 1e-10:
                sc, _ = spearmanr(ivy[masked], ivp[masked])
            else:
                sc = -1.0
            pg = config.get('pearson_guard', 0.0)
            if pg > 0:
                all_r, _ = pearsonr(ivy, ivp)
                base_r, _ = pearsonr(ivy, train_base[iv])
                sc = sc - max(0, base_r - all_r) * pg
            if sc > best_score + 1e-4:
                best_score = sc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 5
                if bad >= config['patience']:
                    break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, z_mu, z_std, best_score


def main():
    device = torch.device('cpu')

    # Load shared data
    with open(COMMON / 'complex_to_fold.json') as f:
        complex_to_fold = json.load(f)
    meta = pd.read_csv(COMMON / 'entries_meta.csv')
    annot = pd.read_csv(ANNOT_PATH)
    geom_data = np.load(GEOM_PATH, allow_pickle=True)
    geom_dict = {str(k): geom_data['geom'][i] for i, k in enumerate(geom_data['keys'])}
    ab_keys = set(annot[(annot['cohort'] == 'antibody_expanded') &
                        (annot['interface_bin'] == 'interface')]['key'])
    pdb_to_fold = {}
    for cplx, fold in complex_to_fold.items():
        pdb = cplx.split('_')[0]
        if pdb not in pdb_to_fold:
            pdb_to_fold[pdb] = fold

    ckpt_dir = OUTPUT / 'checkpoints'

    for model_name in MODEL_ORDER:
        print(f'\n{"=" * 60}\n  {model_name}\n{"=" * 60}')

        path = BASE_PATHS[model_name]
        if not path.exists():
            print(f'  SKIPPED: {path} not found')
            continue

        df = pd.read_csv(path).dropna(subset=['ddG_pred'])
        if 'key' not in df.columns:
            if 'pdbcode' in df.columns and 'mutstr' in df.columns:
                df['key'] = df['pdbcode'] + '_' + df['mutstr']
            elif 'complex' in df.columns and 'mutstr' in df.columns:
                df['pdbcode'] = df['complex'].str.split('_').str[0]
                df['key'] = df['pdbcode'] + '_' + df['mutstr']
        if 'num_muts' not in df.columns:
            df['num_muts'] = df['key'].apply(lambda k: k.split('_', 1)[1].count(',') + 1)
        if 'complex' not in df.columns:
            meta_cplx = meta[['pdbcode', 'mutstr', 'complex']].drop_duplicates()
            meta_cplx['key'] = meta_cplx['pdbcode'] + '_' + meta_cplx['mutstr']
            df = df.merge(meta_cplx[['key', 'complex']], on='key', how='left')

        df['fold'] = df['complex'].map(complex_to_fold)
        no_fold = df['fold'].isna()
        if no_fold.any():
            df.loc[no_fold, 'fold'] = df.loc[no_fold, 'key'].apply(
                lambda k: pdb_to_fold.get(k.split('_')[0], np.nan))
        df = df.dropna(subset=['fold'])
        df['fold'] = df['fold'].astype(int)
        df['ab_interface'] = df['key'].isin(ab_keys)

        # Filter to entries with geometry
        df_with_geom = df[df['key'].isin(geom_dict)].copy()
        if len(df_with_geom) < 50:
            print(f'  SKIPPED: too few entries with geometry ({len(df_with_geom)})')
            continue

        keys_arr = df_with_geom['key'].values
        bp = df_with_geom['ddG_pred'].values.astype(np.float32)
        y = df_with_geom['ddG'].values.astype(np.float32)
        geom = np.array([geom_dict[k] for k in keys_arr], dtype=np.float32)
        cplx = df_with_geom['complex'].values
        folds = df_with_geom['fold'].values.astype(int)
        ab_mask = df_with_geom['ab_interface'].values.astype(bool)

        print(f'  Entries: {len(df_with_geom)} with geometry, {ab_mask.sum()} ab-interface')

        # Save dir for this backbone
        backbone_slug = model_name.lower().replace(' ', '_').replace('-', '_')
        save_dir = ckpt_dir / backbone_slug
        save_dir.mkdir(parents=True, exist_ok=True)

        rseed = CIR_CONFIG['seed']
        for fold in range(CIR_CONFIG['n_outer_folds']):
            ti = folds != fold
            if ti.sum() == 0:
                continue

            model, z_mu, z_std, score = train_and_save_one_fold(
                bp[ti], geom[ti], y[ti], ab_mask[ti], cplx[ti],
                CIR_CONFIG, device, rseed + fold * 100)

            ckpt = {
                'model_state_dict': model.state_dict(),
                'z_mu': z_mu,
                'z_std': z_std,
                'config': CIR_CONFIG,
                'fold': fold,
                'backbone': model_name,
                'n_train': int(ti.sum()),
                'n_interface_train': int(ab_mask[ti].sum()),
                'best_val_score': float(score),
            }
            save_path = save_dir / f'fold{fold}.pt'
            torch.save(ckpt, save_path)
            print(f'    Fold {fold}: saved (score={score:.4f}, n_train={ti.sum()})')

    print(f'\nAll checkpoints saved to: {ckpt_dir}/')


if __name__ == '__main__':
    main()
