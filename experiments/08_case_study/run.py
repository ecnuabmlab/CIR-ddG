"""
Step 08: Case Study — 3HFM (HyHEL-10 Fab + Hen Egg-White Lysozyme)

Visualize CIR-DDG's geometry-guided correction on a classic antibody-antigen
complex. Generates:
  1. Per-residue correction analysis (which positions, how much, direction)
  2. PyMOL script for 3D visualization
  3. Summary statistics and scatter plots

Complex: 3HFM
  Chain H = Heavy chain (antibody)
  Chain L = Light chain (antibody)
  Chain Y = Lysozyme (antigen)
  96 ab-interface mutations, CIR Δρ = +0.101

Usage:
    python run.py --device cpu
"""

import sys
import os
import json
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, '/data/yuwl/paper1/CIR-DDG/baselines/RDE-Network')
sys.path.insert(0, '/data/yuwl/paper1/CIR-DDG/cir_ddg/src')

from model import CrossChainResidual
from geometry import compute_geometry_from_pdb

PROJECT = Path('/data/yuwl/paper1')
CKPT_DIR = PROJECT / 'experiments' / '01_main' / 'checkpoints'
BASE_DIR = PROJECT / 'experiments' / '00_base_retrain'
COMMON = PROJECT / 'experiments' / 'common'
GEOM_PATH = PROJECT / 'CIR-DDG' / 'cir_ddg' / 'data' / 'interface_geometry_all.npz'
ANNOT_PATH = PROJECT / 'CIR-DDG' / 'cir_ddg' / 'data' / 'd5b2_all_skempi_annotated_zero_shot.csv'
PDB_PATH = PROJECT / 'CIR-DDG' / 'data' / 'SKEMPI2' / 'PDBs' / '3HFM.pdb'
OUTPUT = PROJECT / 'experiments' / '08_case_study'

COMPLEX = '3HFM_HL_Y'
ANTIBODY_CHAINS = ['H', 'L']
ANTIGEN_CHAINS = ['Y']


# ============================================================
# Data Loading & CIR Application
# ============================================================

def load_and_apply_cir(device):
    """Load 3HFM ab-interface entries, apply RDE + CIR-DDG."""
    # Load SKEMPI data
    with open(COMMON / 'complex_to_fold.json') as f:
        c2f = json.load(f)
    annot = pd.read_csv(ANNOT_PATH)
    ab_keys = set(annot[(annot['cohort'] == 'antibody_expanded') &
                        (annot['interface_bin'] == 'interface')]['key'])
    geom_data = np.load(GEOM_PATH, allow_pickle=True)
    geom_dict = {str(k): geom_data['geom'][i] for i, k in enumerate(geom_data['keys'])}

    # RDE predictions for 3HFM
    rde = pd.read_csv(BASE_DIR / 'rde/predictions/all_predictions.csv').dropna(subset=['ddG_pred'])
    rde['pdbcode'] = rde['complex'].str.split('_').str[0]
    rde['key'] = rde['pdbcode'] + '_' + rde['mutstr']
    rde_3hfm = rde[(rde['complex'] == COMPLEX) & rde['key'].isin(ab_keys) & rde['key'].isin(geom_dict)].copy()

    # Fold assignment
    fold = c2f[COMPLEX]
    print(f'3HFM: fold={fold}, {len(rde_3hfm)} ab-interface entries with geometry')

    # Base predictions and ground truth
    keys = rde_3hfm['key'].values
    bp = rde_3hfm['ddG_pred'].values.astype(np.float32)
    y = rde_3hfm['ddG'].values.astype(np.float32)
    geom = np.array([geom_dict[k] for k in keys], dtype=np.float32)
    mutstrs = rde_3hfm['mutstr'].values

    # Apply CIR-DDG (use the correct fold)
    ckpt = torch.load(str(CKPT_DIR / 'rde_network' / f'fold{fold}.pt'), map_location=device)
    cir_model = CrossChainResidual(22, 64, 0.1).to(device)
    cir_model.load_state_dict(ckpt['model_state_dict'])
    cir_model.eval()
    z_mu, z_std = ckpt['z_mu'], ckpt['z_std']

    z = np.clip(((geom - z_mu) / z_std).astype(np.float32), -5.0, 5.0)
    imask = (geom[:, 0] <= 5.0).astype(np.float32)
    with torch.no_grad():
        cir_pred = cir_model(
            torch.from_numpy(bp).to(device),
            torch.from_numpy(z).to(device),
            torch.from_numpy(imask).to(device),
            1.0
        ).cpu().numpy()

    # Build result DataFrame
    result = pd.DataFrame({
        'key': keys,
        'mutstr': mutstrs,
        'ddG_true': y,
        'ddG_base': bp,
        'ddG_cir': cir_pred,
        'correction': cir_pred - bp,
        'error_base': y - bp,
        'error_cir': y - cir_pred,
        'min_dist': geom[:, 0],
        'interface': imask.astype(bool),
    })

    # Parse mutation details
    result['chain'] = result['mutstr'].apply(lambda m: m.split(',')[0][1] if len(m) > 1 else '')
    result['position'] = result['mutstr'].apply(
        lambda m: int(''.join(c for c in m.split(',')[0][2:-1] if c.isdigit())) if len(m) > 2 else 0)
    result['wt'] = result['mutstr'].apply(lambda m: m.split(',')[0][0] if m else '')
    result['mt'] = result['mutstr'].apply(lambda m: m.split(',')[0][-1] if m else '')

    rho_base, _ = spearmanr(y, bp)
    rho_cir, _ = spearmanr(y, cir_pred)
    print(f'  Spearman ρ: base={rho_base:.4f}, +CIR={rho_cir:.4f}, Δ={rho_cir-rho_base:+.4f}')

    return result


# ============================================================
# Per-Position Analysis
# ============================================================

def analyze_positions(df):
    """Aggregate corrections per residue position."""
    pos_df = df.groupby(['chain', 'position']).agg(
        n_muts=('ddG_true', 'count'),
        wt=('wt', 'first'),
        mean_correction=('correction', 'mean'),
        abs_correction=('correction', lambda x: np.abs(x).mean()),
        mean_error_base=('error_base', 'mean'),
        mean_min_dist=('min_dist', 'mean'),
        rho_improvement=('ddG_true', lambda x: None),  # placeholder
    ).reset_index()

    # Per-position Spearman improvement (where n >= 5)
    improvements = []
    for (ch, pos), grp in df.groupby(['chain', 'position']):
        if len(grp) < 5:
            improvements.append(np.nan)
            continue
        rb, _ = spearmanr(grp['ddG_true'], grp['ddG_base'])
        rc, _ = spearmanr(grp['ddG_true'], grp['ddG_cir'])
        improvements.append(rc - rb)
    pos_df['rho_improvement'] = improvements

    # Correct direction: correction goes same direction as base error
    pos_df['correct_direction'] = (pos_df['mean_correction'] * pos_df['mean_error_base']) > 0

    return pos_df


# ============================================================
# PyMOL Script Generation
# ============================================================

def generate_pymol_script(pos_df, df):
    """Generate PyMOL script for 3D visualization."""
    script = f"""
# PyMOL script for CIR-DDG Case Study: 3HFM (HyHEL-10 + Lysozyme)
# Generated by Step 08

# Load structure
load {PDB_PATH}, complex

# Basic setup
bg_color white
hide everything
show cartoon, complex

# Color scheme: antibody=blue, antigen=red
color marine, chain H
color slate, chain L
color firebrick, chain Y

# Show interface residues as sticks
"""

    # Add interface residues
    interface_positions = pos_df[pos_df['mean_min_dist'] <= 5.0]
    for _, row in interface_positions.iterrows():
        ch, pos = row['chain'], int(row['position'])
        script += f"show sticks, chain {ch} and resi {pos}\n"

    # Color by correction magnitude (blue=negative correction, red=positive)
    script += "\n# Color interface residues by CIR correction magnitude\n"
    script += "# Red = positive correction (CIR increases prediction)\n"
    script += "# Blue = negative correction (CIR decreases prediction)\n\n"

    max_corr = interface_positions['abs_correction'].max()
    for _, row in interface_positions.iterrows():
        ch, pos = row['chain'], int(row['position'])
        corr = row['mean_correction']
        # Normalize to [0, 1]: 0.5 = no correction, 1 = max positive, 0 = max negative
        norm = 0.5 + 0.5 * corr / (max_corr + 1e-6)
        norm = max(0, min(1, norm))
        r, g, b = plt.cm.RdBu_r(norm)[:3]
        script += f"set_color corr_{ch}{pos}, [{r:.3f}, {g:.3f}, {b:.3f}]\n"
        script += f"color corr_{ch}{pos}, chain {ch} and resi {pos}\n"

    # Highlight top corrected positions
    top5 = interface_positions.nlargest(5, 'abs_correction')
    script += "\n# Top 5 most corrected positions (label)\n"
    for _, row in top5.iterrows():
        ch, pos, wt = row['chain'], int(row['position']), row['wt']
        script += f'label chain {ch} and resi {pos} and name CA, "{wt}{pos}({ch})"\n'

    # Show antigen contact residues near interface
    script += """
# Show antigen surface near interface
show surface, chain Y
set surface_color, firebrick, chain Y
set transparency, 0.7, chain Y

# View settings
set cartoon_fancy_helices, 1
set stick_radius, 0.15
set label_size, 14
set label_color, black

# Orient view to show interface
orient chain H + chain L + (chain Y within 8 of (chain H or chain L))
zoom chain H + chain L + (chain Y within 8 of (chain H or chain L)), 5

# Ray trace
set ray_shadow, 0
set antialias, 2
"""

    script_path = OUTPUT / 'figures' / 'pymol_3hfm.pml'
    with open(script_path, 'w') as f:
        f.write(script)
    print(f'  PyMOL script: {script_path}')
    return script_path


def render_pymol_3d(pos_df):
    """Render 3D structure as PDF using headless PyMOL."""
    import pymol
    from pymol import cmd

    pymol.finish_launching(['pymol', '-cq'])

    cmd.load(str(PDB_PATH), 'complex')
    cmd.hide('everything')
    cmd.show('cartoon', 'complex')

    # Color: antibody blue/slate, antigen red
    cmd.color('marine', 'chain H')
    cmd.color('slate', 'chain L')
    cmd.color('firebrick', 'chain Y')

    # Show interface residues as sticks
    interface_positions = pos_df[pos_df['mean_min_dist'] <= 5.0]
    for _, row in interface_positions.iterrows():
        ch, pos = row['chain'], int(row['position'])
        cmd.show('sticks', f'chain {ch} and resi {pos}')

    # Color interface residues by correction magnitude (RdBu colormap)
    max_corr = interface_positions['abs_correction'].max()
    for _, row in interface_positions.iterrows():
        ch, pos = row['chain'], int(row['position'])
        corr = row['mean_correction']
        norm = 0.5 + 0.5 * corr / (max_corr + 1e-6)
        norm = max(0.0, min(1.0, norm))
        r, g, b = plt.cm.RdBu_r(norm)[:3]
        color_name = f'corr_{ch}{int(pos)}'
        cmd.set_color(color_name, [r, g, b])
        cmd.color(color_name, f'chain {ch} and resi {int(pos)}')

    # Show antigen surface (transparent)
    cmd.show('surface', 'chain Y')
    cmd.set('surface_color', 'firebrick', 'chain Y')
    cmd.set('transparency', 0.7, 'chain Y')

    # Settings
    cmd.set('cartoon_fancy_helices', 1)
    cmd.set('stick_radius', 0.15)
    cmd.set('ray_shadow', 0)
    cmd.set('antialias', 2)
    cmd.bg_color('white')

    # Orient to show interface
    cmd.orient('chain H + chain L + (chain Y within 8 of (chain H or chain L))')
    cmd.zoom('chain H + chain L + (chain Y within 8 of (chain H or chain L))', 5)

    # Render
    png_path = str(OUTPUT / 'figures' / 'structure_3d.png')
    cmd.ray(1600, 1200)
    cmd.png(png_path, dpi=300)
    cmd.quit()

    # Convert PNG to PDF
    from PIL import Image
    img = Image.open(png_path)
    pdf_path = str(OUTPUT / 'figures' / 'structure_3d.pdf')
    img.save(pdf_path, 'PDF', resolution=300)
    print(f'  3D structure: {pdf_path}')
    return pdf_path


# ============================================================
# Figures
# ============================================================

def plot_correction_scatter(df):
    """Scatter: base error vs CIR correction (per mutation)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    interface = df[df['interface']].copy()

    ax.scatter(interface['error_base'], interface['correction'],
               alpha=0.5, s=20, c=interface['min_dist'], cmap='viridis_r',
               edgecolors='none')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

    # Quadrants: correct direction = Q1 (both +) or Q3 (both -)
    n_correct = ((interface['error_base'] * interface['correction']) > 0).sum()
    n_total = len(interface)
    ax.set_xlabel('Base prediction error (truth − base)', fontsize=11)
    ax.set_ylabel('CIR-DDG correction', fontsize=11)
    ax.set_title(f'3HFM: Correction vs Error ({n_correct}/{n_total} correct direction)', fontsize=12)

    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Min distance to antigen (Å)')

    plt.tight_layout()
    fig_path = OUTPUT / 'figures' / 'correction_scatter.pdf'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Scatter plot: {fig_path}')


def plot_per_position_bar(pos_df):
    """Bar chart: per-position correction magnitude, colored by direction."""
    interface = pos_df[pos_df['mean_min_dist'] <= 5.0].sort_values('abs_correction', ascending=False).head(20)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = ['#2166AC' if c < 0 else '#B2182B' for c in interface['mean_correction']]
    labels = [f"{row['wt']}{int(row['position'])}({row['chain']})" for _, row in interface.iterrows()]

    bars = ax.bar(range(len(interface)), interface['mean_correction'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(interface)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Mean CIR correction', fontsize=11)
    ax.set_title('3HFM: Top 20 interface positions by |correction|', fontsize=12)

    # Mark correct direction with checkmarks
    for i, (_, row) in enumerate(interface.iterrows()):
        if row['correct_direction']:
            ax.text(i, row['mean_correction'] + 0.02 * np.sign(row['mean_correction']),
                    '✓', ha='center', fontsize=8, color='green')

    plt.tight_layout()
    fig_path = OUTPUT / 'figures' / 'position_bar.pdf'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Bar chart: {fig_path}')


def plot_distance_vs_correction(df):
    """Scatter: min_distance vs |correction| (shows geometry drives correction)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    interface = df[df['interface']]

    ax.scatter(interface['min_dist'], np.abs(interface['correction']),
               alpha=0.4, s=15, color='#D32F2F', edgecolors='none')
    ax.set_xlabel('Min distance to antigen (Å)', fontsize=11)
    ax.set_ylabel('|CIR correction|', fontsize=11)
    ax.set_title('3HFM: Closer to antigen → larger correction', fontsize=12)

    # Add trend line
    from numpy.polynomial.polynomial import polyfit
    x, y_val = interface['min_dist'].values, np.abs(interface['correction'].values)
    b, m = polyfit(x, y_val, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b + m * xs, 'k--', linewidth=1.5, alpha=0.7)

    rho, pval = spearmanr(x, y_val)
    ax.text(0.95, 0.95, f'ρ = {rho:.3f}\np = {pval:.1e}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig_path = OUTPUT / 'figures' / 'distance_vs_correction.pdf'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Distance plot: {fig_path}')


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = args.device

    (OUTPUT / 'figures').mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'logs').mkdir(exist_ok=True)
    print(f'Device: {device}\n')

    # 1. Load data and apply CIR
    print('=== Loading 3HFM data & applying CIR-DDG ===')
    df = load_and_apply_cir(device)

    # 2. Per-position analysis
    print('\n=== Per-position analysis ===')
    pos_df = analyze_positions(df)
    n_correct = pos_df['correct_direction'].sum()
    n_total = len(pos_df)
    print(f'  Positions with correct correction direction: {n_correct}/{n_total} ({100*n_correct/n_total:.0f}%)')

    # 3. Generate PyMOL script + render 3D
    print('\n=== Generating PyMOL visualization ===')
    generate_pymol_script(pos_df, df)
    render_pymol_3d(pos_df)

    # 4. Generate figures
    print('\n=== Generating figures ===')
    plot_correction_scatter(df)
    plot_per_position_bar(pos_df)
    plot_distance_vs_correction(df)

    # 5. Save analysis data
    df.to_csv(OUTPUT / 'mutation_analysis.csv', index=False)
    pos_df.to_csv(OUTPUT / 'position_analysis.csv', index=False)

    # 6. Summary
    rho_base, _ = spearmanr(df['ddG_true'], df['ddG_base'])
    rho_cir, _ = spearmanr(df['ddG_true'], df['ddG_cir'])
    mae_base = np.abs(df['error_base']).mean()
    mae_cir = np.abs(df['error_cir']).mean()

    summary = {
        'complex': COMPLEX,
        'pdb': '3HFM',
        'system': 'HyHEL-10 Fab + Hen Egg-White Lysozyme',
        'n_mutations': len(df),
        'n_interface': int(df['interface'].sum()),
        'rho_base': float(rho_base),
        'rho_cir': float(rho_cir),
        'delta_rho': float(rho_cir - rho_base),
        'mae_base': float(mae_base),
        'mae_cir': float(mae_cir),
        'correction_direction_accuracy': float(n_correct / n_total),
    }
    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n=== Summary ===')
    print(f'  System: HyHEL-10 Fab + Lysozyme (3HFM)')
    print(f'  Mutations: {len(df)} (interface: {int(df["interface"].sum())})')
    print(f'  Spearman ρ: base={rho_base:.4f}, +CIR={rho_cir:.4f}, Δ={rho_cir-rho_base:+.4f}')
    print(f'  MAE: base={mae_base:.3f}, +CIR={mae_cir:.3f}')
    print(f'  Correction direction accuracy: {100*n_correct/n_total:.0f}%')
    print(f'\n  Outputs:')
    print(f'    {OUTPUT}/figures/pymol_3hfm.pml')
    print(f'    {OUTPUT}/figures/correction_scatter.pdf')
    print(f'    {OUTPUT}/figures/position_bar.pdf')
    print(f'    {OUTPUT}/figures/distance_vs_correction.pdf')
    print(f'    {OUTPUT}/mutation_analysis.csv')
    print(f'    {OUTPUT}/position_analysis.csv')
    print(f'    {OUTPUT}/results.json')


if __name__ == '__main__':
    main()
