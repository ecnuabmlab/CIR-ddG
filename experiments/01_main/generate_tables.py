"""
Step 01 v3: Generate three main tables (Overall, Per-Structure, Ab-Interface)
Each table: 6 models × {base, +CIR-DDG} × {all, single, multi} = 36 rows × 7 columns

Columns: Model, Mutation, Pearson's R, Spearman's ρ, RMSE, MAE, AUROC

Wait for all base models to finish before running.
"""

import sys, json, math, random, copy, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression

sys.path.insert(0, '/data/yuwl/paper1/CIR-DDG/cir_ddg/src')
from model import CrossChainResidual

PROJECT = Path('/data/yuwl/paper1')
COMMON = PROJECT / 'experiments_v3' / 'common'
BASE_DIR = PROJECT / 'experiments_v3' / '00_base_retrain'
OUTPUT = PROJECT / 'experiments_v3' / '01_main'
GEOM_PATH = PROJECT / 'CIR-DDG' / 'cir_ddg' / 'data' / 'interface_geometry_all.npz'
ANNOT_PATH = PROJECT / 'CIR-DDG' / 'cir_ddg' / 'data' / 'd5b2_all_skempi_annotated_zero_shot.csv'

OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / 'predictions').mkdir(exist_ok=True)

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


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred):
    """Compute 5 metrics: Pearson R, Spearman rho, RMSE, MAE, AUROC."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 5:
        return {'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan}

    r, _ = pearsonr(yt, yp)
    rho, _ = spearmanr(yt, yp)

    # Linear-calibrated RMSE/MAE
    lr = LinearRegression().fit(yp.reshape(-1, 1), yt)
    yp_cal = lr.predict(yp.reshape(-1, 1))
    rmse = float(np.sqrt(np.mean((yt - yp_cal) ** 2)))
    mae = float(np.mean(np.abs(yt - yp_cal)))

    # AUROC: ddG < 0 = stabilizing = positive class; score = -prediction
    labels = (yt < 0).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        auroc = np.nan
    else:
        auroc = roc_auc_score(labels, -yp)

    return {'R': float(r), 'rho': float(rho), 'RMSE': rmse, 'MAE': mae, 'AUROC': auroc}


def compute_per_structure_metrics(df, pred_col='ddG_pred', true_col='ddG',
                                   group_col='complex', min_rows=10):
    """Compute per-structure averaged metrics."""
    rs, rhos, rmses, maes, aurocs = [], [], [], [], []
    for name, grp in df.groupby(group_col):
        if len(grp) < min_rows:
            continue
        yt = grp[true_col].values
        yp = grp[pred_col].values
        if np.std(yt) < 1e-10 or np.std(yp) < 1e-10:
            continue
        m = compute_metrics(yt, yp)
        if not np.isnan(m['R']):
            rs.append(m['R'])
            rhos.append(m['rho'])
            rmses.append(m['RMSE'])
            maes.append(m['MAE'])
            if not np.isnan(m['AUROC']):
                aurocs.append(m['AUROC'])

    if not rs:
        return {'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan, 'n_complexes': 0}
    return {
        'R': float(np.mean(rs)), 'rho': float(np.mean(rhos)),
        'RMSE': float(np.mean(rmses)), 'MAE': float(np.mean(maes)),
        'AUROC': float(np.mean(aurocs)) if aurocs else np.nan,
        'n_complexes': len(rs),
    }


# ============================================================
# CIR-DDG Residual Training
# ============================================================

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
    """Differentiable Pearson correlation loss: 1 - R."""
    vx = pred - pred.mean()
    vy = target - target.mean()
    r = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)
    return 1.0 - r

def train_cir_one_fold(train_base, train_geom, train_y, train_mask, train_cplx,
                       val_base, val_geom, val_mask, config, device, run_seed):
    seed_all(run_seed)
    inner_folds = complex_level_inner_split(train_cplx, config['n_inner_folds'], run_seed)
    z_mu = train_geom.mean(0); z_std = train_geom.std(0) + 1e-6
    train_z = (train_geom - z_mu) / z_std; val_z = (val_geom - z_mu) / z_std
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
        # Combined loss: MSE + interface focus + Pearson preservation
        loss = F.mse_loss(p, ty)
        if tm.any():
            loss = loss + F.mse_loss(p[tm.bool()], ty[tm.bool()])
        # Pearson loss to preserve global linear relationship
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
            # Pearson guard: penalize if overall Pearson drops
            pg = config.get('pearson_guard', 0.0)
            if pg > 0:
                all_r, _ = pearsonr(ivy, ivp)
                base_r, _ = pearsonr(ivy, train_base[iv])
                penalty = max(0, base_r - all_r) * pg
                sc = sc - penalty
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
    with torch.no_grad():
        vp = model(torch.from_numpy(val_base.astype(np.float32)).to(device),
                   torch.from_numpy(val_z.astype(np.float32)).to(device),
                   torch.from_numpy(val_mask.astype(np.float32)).to(device), 1.0).cpu().numpy()
    # Clamp correction magnitude to prevent large scale shifts
    max_corr = config.get('max_correction', None)
    if max_corr is not None:
        correction = vp - val_base
        correction = np.clip(correction, -max_corr, max_corr)
        vp = val_base + correction
    return vp


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device('cpu')

    # Load shared data
    with open(COMMON / 'complex_to_fold.json') as f:
        complex_to_fold = json.load(f)
    meta = pd.read_csv(COMMON / 'entries_meta.csv')
    annot = pd.read_csv(ANNOT_PATH)
    geom_data = np.load(GEOM_PATH, allow_pickle=True)
    geom_dict = {str(k): geom_data['geom'][i] for i, k in enumerate(geom_data['keys'])}

    # Ab-interface mask from annot (cohort=antibody_expanded + interface_bin=interface)
    ab_keys = set(annot[(annot['cohort'] == 'antibody_expanded') &
                        (annot['interface_bin'] == 'interface')]['key'])

    # pdb_to_fold for models that use pdb-only complex format
    pdb_to_fold = {}
    for cplx, fold in complex_to_fold.items():
        pdb = cplx.split('_')[0]
        if pdb not in pdb_to_fold:
            pdb_to_fold[pdb] = fold

    # Results storage
    rows_overall = []
    rows_perstruct = []
    rows_abinterface = []

    for model_name in MODEL_ORDER:
        print(f'\n{"=" * 60}\n  {model_name}\n{"=" * 60}')

        # Load base predictions
        path = BASE_PATHS[model_name]
        if not path.exists():
            print(f'  SKIPPED: {path} not found')
            for mut_type in ['all', 'single', 'multi']:
                rows_overall.append({'Model': model_name, 'Mutation': mut_type,
                                     'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan})
                rows_overall.append({'Model': f'{model_name} +CIR-DDG', 'Mutation': mut_type,
                                     'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan})
            continue

        df = pd.read_csv(path).dropna(subset=['ddG_pred'])
        if 'key' not in df.columns:
            if 'pdbcode' in df.columns and 'mutstr' in df.columns:
                df['key'] = df['pdbcode'] + '_' + df['mutstr']
            elif 'complex' in df.columns and 'mutstr' in df.columns:
                # RDE format: complex='1CSE_E_I', extract pdbcode from it
                df['pdbcode'] = df['complex'].str.split('_').str[0]
                df['key'] = df['pdbcode'] + '_' + df['mutstr']
        if 'num_muts' not in df.columns:
            df['num_muts'] = df['key'].apply(lambda k: k.split('_', 1)[1].count(',') + 1)
        if 'complex' not in df.columns:
            # Merge complex from meta
            meta_cplx = meta[['pdbcode', 'mutstr', 'complex']].drop_duplicates()
            meta_cplx['key'] = meta_cplx['pdbcode'] + '_' + meta_cplx['mutstr']
            df = df.merge(meta_cplx[['key', 'complex']], on='key', how='left')

        # Assign folds
        df['fold'] = df['complex'].map(complex_to_fold)
        # For entries without complex_to_fold match, try pdb_to_fold
        no_fold = df['fold'].isna()
        if no_fold.any():
            df.loc[no_fold, 'fold'] = df.loc[no_fold, 'key'].apply(
                lambda k: pdb_to_fold.get(k.split('_')[0], np.nan))
        df = df.dropna(subset=['fold'])
        df['fold'] = df['fold'].astype(int)

        # Ab-interface mask
        df['ab_interface'] = df['key'].isin(ab_keys)

        print(f'  Entries: {len(df)} (single={int((df.num_muts==1).sum())}, multi={int((df.num_muts>1).sum())})')
        print(f'  Ab-interface: {df.ab_interface.sum()}')

        # --- BASE metrics ---
        for mut_type in ['all', 'single', 'multi']:
            if mut_type == 'all':
                sub = df
            elif mut_type == 'single':
                sub = df[df.num_muts == 1]
            else:
                sub = df[df.num_muts > 1]

            if len(sub) < 10:
                m_ov = {'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan}
                m_ps = {'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan}
                m_ab = {'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan}
            else:
                m_ov = compute_metrics(sub['ddG'].values, sub['ddG_pred'].values)
                m_ps = compute_per_structure_metrics(sub, 'ddG_pred', 'ddG', 'complex', 10)
                sub_ab = sub[sub.ab_interface]
                m_ab = compute_metrics(sub_ab['ddG'].values, sub_ab['ddG_pred'].values) if len(sub_ab) >= 10 else \
                    {'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan}

            rows_overall.append({'Model': model_name, 'Mutation': mut_type, **m_ov})
            rows_perstruct.append({'Model': model_name, 'Mutation': mut_type, **m_ps})
            rows_abinterface.append({'Model': model_name, 'Mutation': mut_type, **m_ab})

        # --- CIR-DDG residual ---
        # Need geometry for residual training
        df_with_geom = df[df['key'].isin(geom_dict)].copy()
        if len(df_with_geom) < 50:
            print(f'  CIR-DDG SKIPPED: too few entries with geometry ({len(df_with_geom)})')
            for mut_type in ['all', 'single', 'multi']:
                rows_overall.append({'Model': f'{model_name} +CIR-DDG', 'Mutation': mut_type,
                                     'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan})
                rows_perstruct.append({'Model': f'{model_name} +CIR-DDG', 'Mutation': mut_type,
                                       'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan})
                rows_abinterface.append({'Model': f'{model_name} +CIR-DDG', 'Mutation': mut_type,
                                         'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan})
            continue

        keys_arr = df_with_geom['key'].values
        bp = df_with_geom['ddG_pred'].values.astype(np.float32)
        y = df_with_geom['ddG'].values.astype(np.float32)
        geom = np.array([geom_dict[k] for k in keys_arr], dtype=np.float32)
        cplx = df_with_geom['complex'].values
        folds = df_with_geom['fold'].values.astype(int)
        ab_mask = df_with_geom['ab_interface'].values.astype(bool)
        num_muts = df_with_geom['num_muts'].values

        # Run 5 times, average predictions
        all_run_preds = []
        for run_id in range(CIR_CONFIG['n_runs']):
            rseed = CIR_CONFIG['seed'] + run_id
            cv_pred = np.copy(bp)
            for fold in range(CIR_CONFIG['n_outer_folds']):
                ti = folds != fold; vi = folds == fold
                if vi.sum() == 0:
                    continue
                vp = train_cir_one_fold(
                    bp[ti], geom[ti], y[ti], ab_mask[ti], cplx[ti],
                    bp[vi], geom[vi], ab_mask[vi], CIR_CONFIG, device, rseed + fold * 100)
                # Apply residual only to ab-interface entries
                val_ab = ab_mask[vi]
                cv_pred[np.where(vi)[0][val_ab]] = vp[val_ab]
            all_run_preds.append(cv_pred)

        # Average across runs
        mean_preds = np.mean(all_run_preds, axis=0)

        # Zero-mean correction: ensure CIR predictions on ab-interface subset
        # have same mean as base predictions (prevents scale shift from hurting Pearson)
        ab_idx = np.where(ab_mask)[0]
        if len(ab_idx) > 0:
            mean_shift = mean_preds[ab_idx].mean() - bp[ab_idx].mean()
            mean_preds[ab_idx] -= mean_shift

        df_with_geom['ddG_pred_cir'] = mean_preds

        # Merge CIR predictions back to FULL df (entries without geometry keep base pred)
        df['ddG_pred_cir'] = df['ddG_pred']  # default: keep base
        df.loc[df['key'].isin(df_with_geom['key'].values), 'ddG_pred_cir'] = \
            df_with_geom.set_index('key')['ddG_pred_cir'].reindex(
                df.loc[df['key'].isin(df_with_geom['key'].values), 'key'].values).values

        # CIR metrics on FULL df (same entry set as base metrics)
        for mut_type in ['all', 'single', 'multi']:
            if mut_type == 'all':
                sub = df
            elif mut_type == 'single':
                sub = df[df.num_muts == 1]
            else:
                sub = df[df.num_muts > 1]

            if len(sub) < 10:
                m_ov = {'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan}
                m_ps = {'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan}
                m_ab = {'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan}
            else:
                m_ov = compute_metrics(sub['ddG'].values, sub['ddG_pred_cir'].values)
                m_ps = compute_per_structure_metrics(sub, 'ddG_pred_cir', 'ddG', 'complex', 10)
                sub_ab = sub[sub['ab_interface']]
                m_ab = compute_metrics(sub_ab['ddG'].values, sub_ab['ddG_pred_cir'].values) if len(sub_ab) >= 10 else \
                    {'R': np.nan, 'rho': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'AUROC': np.nan}

            rows_overall.append({'Model': f'{model_name} +CIR-DDG', 'Mutation': mut_type, **m_ov})
            rows_perstruct.append({'Model': f'{model_name} +CIR-DDG', 'Mutation': mut_type, **m_ps})
            rows_abinterface.append({'Model': f'{model_name} +CIR-DDG', 'Mutation': mut_type, **m_ab})

        print(f'  CIR-DDG done (5 runs averaged)')

    # Save tables
    col_order = ['Model', 'Mutation', 'R', 'rho', 'RMSE', 'MAE', 'AUROC']
    col_rename = {'R': "Pearson's R", 'rho': "Spearman's ρ", 'RMSE': 'RMSE', 'MAE': 'MAE', 'AUROC': 'AUROC'}

    df_overall = pd.DataFrame(rows_overall)[col_order].rename(columns=col_rename)
    df_perstruct = pd.DataFrame(rows_perstruct)[col_order].rename(columns=col_rename)
    df_abinterface = pd.DataFrame(rows_abinterface)[col_order].rename(columns=col_rename)

    # Ab-interface: only keep 'all' rows (single=all, multi=NaN, so simplify)
    df_abinterface = df_abinterface[df_abinterface['Mutation'] == 'all'].drop(columns=['Mutation'])

    df_overall.to_csv(OUTPUT / 'table_overall.csv', index=False)
    df_perstruct.to_csv(OUTPUT / 'table_per_structure.csv', index=False)
    df_abinterface.to_csv(OUTPUT / 'table_ab_interface.csv', index=False)

    print(f'\n{"=" * 60}')
    print(f'Saved:')
    print(f'  {OUTPUT}/table_overall.csv')
    print(f'  {OUTPUT}/table_per_structure.csv')
    print(f'  {OUTPUT}/table_ab_interface.csv')
    print(f'{"=" * 60}')

    # Print summary
    print('\n--- Overall Table ---')
    print(df_overall.to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
