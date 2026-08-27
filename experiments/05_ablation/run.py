"""
Step 05: Ablation & Controls

05a: Equal Capacity — geometry input zeroed, same MLP structure (proves gain from geometry not params)
05b: Feature Ablation — which of 3 feature groups contributes most
     - Distance (dims 0-8, 9d)
     - Contact counts (dims 9-17, 9d)
     - Chemical type (dims 18-21, 4d)

Evaluated on ab-interface Spearman ρ. All 6 models.

Usage: python run.py
"""
import sys
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, '/data/yuwl/paper1/CIR-DDG/cir_ddg/src')
from model import CrossChainResidual

COMMON = Path('/data/yuwl/paper1/experiments/common')
BASE_DIR = Path('/data/yuwl/paper1/experiments/00_base_retrain')
GEOM_PATH = Path('/data/yuwl/paper1/CIR-DDG/cir_ddg/data/interface_geometry_all.npz')
ANNOT_PATH = Path('/data/yuwl/paper1/CIR-DDG/cir_ddg/data/d5b2_all_skempi_annotated_zero_shot.csv')
OUTPUT = Path('/data/yuwl/paper1/experiments/05_ablation')

ALL_MODELS = {
    'ProteinMPNN': BASE_DIR / 'proteinmpnn/predictions/predictions.csv',
    'ESM-IF': BASE_DIR / 'esmif/predictions/predictions.csv',
    'RDE-Network': BASE_DIR / 'rde/predictions/all_predictions.csv',
    'DiffAffinity': BASE_DIR / 'diffaffinity/predictions/predictions.csv',
    'DDAffinity': BASE_DIR / 'ddaffinity/predictions/phase_b_all_predictions.csv',
    'Vanilla Pythia-PPI': BASE_DIR / 'pythia/predictions/predictions.csv',
}


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def cplx_split(complexes, n_folds, seed):
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
    return 1 - (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8)


def run_ablation(bp, geom_input, y, ab_mask, folds, cplx):
    """Train CIR-DDG with given geometry input, return ab-interface Spearman."""
    seed_all(2026)
    cv_pred = np.copy(bp)
    for fold in range(5):
        ti = folds != fold; vi = folds == fold
        inner = cplx_split(cplx[ti], 3, 2026)
        z_mu = geom_input[ti].mean(0); z_std = geom_input[ti].std(0) + 1e-6
        it = inner != 0

        tz = torch.from_numpy(((geom_input[ti] - z_mu) / z_std)[it].astype(np.float32))
        tb = torch.from_numpy(bp[ti][it])
        ty_t = torch.from_numpy(y[ti][it])
        tm = torch.from_numpy(ab_mask[ti][it].astype(np.float32))
        ivz = torch.from_numpy(((geom_input[ti] - z_mu) / z_std)[inner == 0].astype(np.float32))
        ivb = torch.from_numpy(bp[ti][inner == 0])
        ivy = y[ti][inner == 0]; ivm = ab_mask[ti][inner == 0]

        gdim = geom_input.shape[1]
        model = CrossChainResidual(gdim, 64, 0.1)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        best_sc, best_st, bad = -1e9, None, 0
        for ep in range(1, 151):
            model.train(); opt.zero_grad()
            p = model(tb, tz, tm, 1.0)
            loss = F.mse_loss(p, ty_t)
            if tm.any():
                loss += F.mse_loss(p[tm.bool()], ty_t[tm.bool()])
            loss += 0.2 * pearson_loss(p, ty_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ep % 5 == 0:
                model.eval()
                with torch.no_grad():
                    ivp = model(ivb, ivz, torch.from_numpy(ivm.astype(np.float32)), 1.0).numpy()
                m = ivm.astype(bool)
                sc = spearmanr(ivy[m], ivp[m])[0] if m.sum() >= 3 and np.std(ivy[m]) > 1e-10 and np.std(ivp[m]) > 1e-10 else -1
                ar, _ = pearsonr(ivy, ivp); br, _ = pearsonr(ivy, bp[ti][inner == 0])
                sc -= max(0, br - ar) * 3.0
                if sc > best_sc + 1e-4:
                    best_sc = sc
                    best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 5
                    if bad >= 30:
                        break
        if best_st:
            model.load_state_dict(best_st)
        model.eval()
        with torch.no_grad():
            vz = torch.from_numpy(((geom_input[vi] - z_mu) / z_std).astype(np.float32))
            vp = model(torch.from_numpy(bp[vi]), vz,
                       torch.from_numpy(ab_mask[vi].astype(np.float32)), 1.0).numpy()
        cv_pred[np.where(vi)[0][ab_mask[vi]]] = vp[ab_mask[vi]]

    rho_ab, _ = spearmanr(y[ab_mask], cv_pred[ab_mask])
    return float(rho_ab)


def main():
    annot = pd.read_csv(ANNOT_PATH)
    ab_keys = set(annot[(annot['cohort'] == 'antibody_expanded') &
                        (annot['interface_bin'] == 'interface')]['key'])
    geom_data = np.load(GEOM_PATH, allow_pickle=True)
    geom_dict = {str(k): geom_data['geom'][i] for i, k in enumerate(geom_data['keys'])}
    with open(COMMON / 'complex_to_fold.json') as f:
        complex_to_fold = json.load(f)
    pdb_to_fold = {}
    for c, f in complex_to_fold.items():
        p = c.split('_')[0]
        if p not in pdb_to_fold:
            pdb_to_fold[p] = f

    results_all = {}

    for model_name, path in ALL_MODELS.items():
        print(f'\n{"=" * 60}\n  {model_name}\n{"=" * 60}')
        df = pd.read_csv(path).dropna(subset=['ddG_pred'])
        if 'key' not in df.columns:
            if 'pdbcode' in df.columns:
                df['key'] = df['pdbcode'] + '_' + df['mutstr']
            elif 'complex' in df.columns:
                df['key'] = df['complex'].str.split('_').str[0] + '_' + df['mutstr']
        df = df[df['key'].isin(geom_dict)].copy()

        # Uniform fold assignment via pdb_to_fold
        df['pdb'] = df['key'].str.split('_').str[0]
        df['fold'] = df['pdb'].map(pdb_to_fold)
        df = df.dropna(subset=['fold'])
        df['fold'] = df['fold'].astype(int)
        df['ab_interface'] = df['key'].isin(ab_keys)

        bp = df['ddG_pred'].values.astype(np.float32)
        y_arr = df['ddG'].values.astype(np.float32)
        geom_arr = np.array([geom_dict[k] for k in df['key'].values], np.float32)
        cplx = df['pdb'].values
        folds = df['fold'].values.astype(int)
        ab_mask = df['ab_interface'].values.astype(bool)

        rho_base, _ = spearmanr(y_arr[ab_mask], bp[ab_mask])
        print(f'  Base: {rho_base:.4f}')

        rho_full = run_ablation(bp, geom_arr, y_arr, ab_mask, folds, cplx)
        print(f'  Full 22d:        {rho_full:.4f} (Δ={rho_full - rho_base:+.4f})')

        rho_zeros = run_ablation(bp, np.zeros_like(geom_arr), y_arr, ab_mask, folds, cplx)
        print(f'  Equal cap (0):   {rho_zeros:.4f} (Δ={rho_zeros - rho_base:+.4f})')

        rho_dist = run_ablation(bp, geom_arr[:, :9], y_arr, ab_mask, folds, cplx)
        print(f'  Distance (9d):   {rho_dist:.4f} (Δ={rho_dist - rho_base:+.4f})')

        rho_contact = run_ablation(bp, geom_arr[:, 9:18], y_arr, ab_mask, folds, cplx)
        print(f'  Contact (9d):    {rho_contact:.4f} (Δ={rho_contact - rho_base:+.4f})')

        rho_chem = run_ablation(bp, geom_arr[:, 18:22], y_arr, ab_mask, folds, cplx)
        print(f'  Chemical (4d):   {rho_chem:.4f} (Δ={rho_chem - rho_base:+.4f})')

        rho_no_dist = run_ablation(bp, geom_arr[:, 9:], y_arr, ab_mask, folds, cplx)
        print(f'  Contact+Chem:    {rho_no_dist:.4f} (Δ={rho_no_dist - rho_base:+.4f})')

        results_all[model_name] = {
            'base': float(rho_base),
            'full_22d': float(rho_full),
            'equal_capacity_zeros': float(rho_zeros),
            'distance_9d': float(rho_dist),
            'contact_9d': float(rho_contact),
            'chemical_4d': float(rho_chem),
            'contact_chem_13d': float(rho_no_dist),
        }

    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(results_all, f, indent=2)

    print(f'\n{"=" * 80}')
    print('ABLATION SUMMARY (Ab-Interface Spearman Δρ)')
    print(f'{"=" * 80}')
    print(f'{"Model":<20} {"Base":>6} {"Full":>6} {"Zeros":>6} {"Dist":>6} {"Contact":>7} {"Chem":>6} {"Cont+Ch":>7}')
    print('-' * 68)
    for m in ALL_MODELS:
        r = results_all[m]; b = r['base']
        print(f'{m:<20} {b:>6.3f} {r["full_22d"] - b:>+6.3f} {r["equal_capacity_zeros"] - b:>+6.3f} '
              f'{r["distance_9d"] - b:>+6.3f} {r["contact_9d"] - b:>+7.3f} '
              f'{r["chemical_4d"] - b:>+6.3f} {r["contact_chem_13d"] - b:>+7.3f}')

    print(f'\nSaved: {OUTPUT}/results.json')


if __name__ == '__main__':
    main()

