"""
Step 04: R² Mechanism — Interface Geometry Under-encoding Evidence

For each backbone on ab-interface subset:
  R²(base_pred → ΔΔG): base's explanatory power
  R²(geom → ΔΔG): 22d geometry's explanatory power (model-agnostic)
  R²(base_pred + geom → ΔΔG): combined
  ΔR² = R²(combined) - R²(base): geometry's incremental value beyond base
  R²(base_pred → geom): how much geometry the base already encodes

Usage: python run.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

COMMON = Path('/data/yuwl/paper1/experiments/common')
BASE_DIR = Path('/data/yuwl/paper1/experiments/00_base_retrain')
GEOM_PATH = Path('/data/yuwl/paper1/CIR-DDG/cir_ddg/data/interface_geometry_all.npz')
ANNOT_PATH = Path('/data/yuwl/paper1/CIR-DDG/cir_ddg/data/d5b2_all_skempi_annotated_zero_shot.csv')
OUTPUT = Path('/data/yuwl/paper1/experiments/04_r2_mechanism')

MODELS = {
    'ProteinMPNN': BASE_DIR / 'proteinmpnn/predictions/predictions.csv',
    'ESM-IF': BASE_DIR / 'esmif/predictions/predictions.csv',
    'RDE-Network': BASE_DIR / 'rde/predictions/all_predictions.csv',
    'DiffAffinity': BASE_DIR / 'diffaffinity/predictions/predictions.csv',
    'DDAffinity': BASE_DIR / 'ddaffinity/predictions/phase_b_all_predictions.csv',
    'Vanilla Pythia-PPI': BASE_DIR / 'pythia/predictions/predictions.csv',
}


def main():
    annot = pd.read_csv(ANNOT_PATH)
    ab_keys = set(annot[(annot['cohort'] == 'antibody_expanded') &
                        (annot['interface_bin'] == 'interface')]['key'])
    geom_data = np.load(GEOM_PATH, allow_pickle=True)
    geom_dict = {str(k): geom_data['geom'][i] for i, k in enumerate(geom_data['keys'])}

    print('=' * 80)
    print('STEP 04: R² Mechanism — Incremental Geometry Contribution')
    print('=' * 80)

    results = {}
    print(f"\n{'Model':<20} {'N':>5} {'R²(base)':>9} {'R²(geom)':>9} {'R²(both)':>9} "
          f"{'ΔR²':>7} {'R²(pred→geom)':>14}")
    print('-' * 80)

    for name, path in MODELS.items():
        df = pd.read_csv(path).dropna(subset=['ddG_pred'])
        if 'key' not in df.columns:
            if 'pdbcode' in df.columns:
                df['key'] = df['pdbcode'] + '_' + df['mutstr']
            elif 'complex' in df.columns:
                df['key'] = df['complex'].str.split('_').str[0] + '_' + df['mutstr']

        common = sorted(set(df['key']) & set(geom_dict.keys()) & ab_keys)
        if len(common) < 50:
            print(f'{name:<20} {len(common):>5} — insufficient entries')
            continue

        pred_map = dict(zip(df['key'], df['ddG_pred']))
        ddg_map = dict(zip(annot['key'], annot['ddG']))

        base_pred = np.array([pred_map[k] for k in common], dtype=np.float32).reshape(-1, 1)
        geom_arr = np.array([geom_dict[k] for k in common], dtype=np.float32)
        ddg_arr = np.array([ddg_map[k] for k in common], dtype=np.float32)

        # (A) R²(base_pred → ΔΔG)
        pred_a = cross_val_predict(Ridge(alpha=1.0), base_pred, ddg_arr, cv=5)
        r2_base = r2_score(ddg_arr, pred_a)

        # (B) R²(geometry → ΔΔG)
        pred_b = cross_val_predict(Ridge(alpha=1.0), geom_arr, ddg_arr, cv=5)
        r2_geom = r2_score(ddg_arr, pred_b)

        # (C) R²(base_pred + geometry → ΔΔG)
        combined = np.hstack([base_pred, geom_arr])
        pred_c = cross_val_predict(Ridge(alpha=1.0), combined, ddg_arr, cv=5)
        r2_combined = r2_score(ddg_arr, pred_c)

        delta_r2 = r2_combined - r2_base

        # (D) R²(base_pred → geometry)
        r2_decode_per_dim = []
        for d in range(22):
            pred_d = cross_val_predict(Ridge(alpha=1.0), base_pred, geom_arr[:, d], cv=5)
            r2_d = r2_score(geom_arr[:, d], pred_d)
            r2_decode_per_dim.append(max(r2_d, 0.0))
        r2_decode_mean = np.mean(r2_decode_per_dim)

        results[name] = {
            'n': len(common),
            'r2_base': float(r2_base),
            'r2_geom': float(r2_geom),
            'r2_combined': float(r2_combined),
            'delta_r2': float(delta_r2),
            'r2_pred_to_geom': float(r2_decode_mean),
            'r2_pred_to_geom_per_dim': r2_decode_per_dim,
        }
        print(f'{name:<20} {len(common):>5} {r2_base:>9.4f} {r2_geom:>9.4f} '
              f'{r2_combined:>9.4f} {delta_r2:>+7.4f} {r2_decode_mean:>14.4f}')

    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUTPUT}/results.json')


if __name__ == '__main__':
    main()
