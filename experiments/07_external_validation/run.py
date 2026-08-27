"""
Step 07: External Validation on SARS-CoV-2 RBD-ACE2 (R3669)

Evaluate CIR-DDG generalization on a completely independent complex:
  - PDB 6M0J (not in SKEMPI): SARS-CoV-2 RBD bound to human ACE2
  - 3669 single-point mutations on RBD (chain E) at the binding interface
  - Experimental binding effects (Starr et al. 2022 DMS)

Protocol (per backbone):
  For each fold i (0..4):
    1. Run fold i's backbone model → base predictions
    2. Apply fold i's CIR-DDG checkpoint → corrected predictions
    3. Compute Spearman ρ for both
  Average metrics across 5 folds.

  Exception: ProteinMPNN is zero-shot (1 model), so base is the same for all folds.
  Only CIR varies per fold.

Backbones: ProteinMPNN, RDE-Network, DiffAffinity*, Pythia-PPI
  * DiffAffinity predictions pre-computed via JAX env (diffaffinity_r3669_preds.npz)

Usage:
    python run.py --device cuda:0
"""

import sys
import os
import json
import re
import copy
import argparse

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, '/data/yuwl/paper1/CIR-DDG/baselines/ProteinMPNN')
sys.path.insert(0, '/data/yuwl/paper1/CIR-DDG/baselines/RDE-Network')
sys.path.insert(0, '/data/yuwl/paper1/CIR-DDG/cir_ddg/src')

from model import CrossChainResidual
from geometry import compute_geometry_from_pdb

PROJECT = Path('/data/yuwl/paper1')
DATA_DIR = PROJECT / 'CIR-DDG' / 'data' / 'R3669_RBD'
CIR_CKPT_DIR = PROJECT / 'experiments' / '01_main' / 'checkpoints'
BASE_CKPT_DIR = PROJECT / 'experiments' / '00_base_retrain'
MPNN_DIR = PROJECT / 'CIR-DDG' / 'baselines' / 'ProteinMPNN'
RDE_DIR = PROJECT / 'CIR-DDG' / 'baselines' / 'RDE-Network'
PYTHIA_DIR = PROJECT / 'CIR-DDG' / 'baselines' / 'Pythia-PPI'
OUTPUT = PROJECT / 'experiments' / '07_external_validation'

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

CIR_SLUGS = {
    'ProteinMPNN': 'proteinmpnn',
    'RDE-Network': 'rde_network',
    'DiffAffinity': 'diffaffinity',
    'Vanilla Pythia-PPI': 'vanilla_pythia_ppi',
}


# ============================================================
# Data & Geometry
# ============================================================

def load_data():
    df = pd.read_csv(DATA_DIR / 'r3669_processed.csv')
    print(f'R3669: {len(df)} entries, {df["pdb_resnum"].nunique()} positions')
    return df

def compute_geometry(df):
    pdb_path = str(DATA_DIR / 'PDBs' / '6M0J.pdb')
    pos_cache = {}
    for pos in df['pdb_resnum'].unique():
        pos_cache[pos] = compute_geometry_from_pdb(pdb_path, 'E', str(pos), ['A'])
    geom = np.array([pos_cache[p] for p in df['pdb_resnum']], dtype=np.float32)
    interface = geom[:, 0] <= 5.0
    print(f'  Geometry: {interface.sum()} interface (≤5Å) / {len(geom)} total')
    return geom, interface


# ============================================================
# CIR-DDG Application
# ============================================================

def apply_cir(fold, backbone_slug, base_preds, geom, interface, device):
    """Apply fold-specific CIR-DDG. Returns corrected predictions."""
    ckpt = torch.load(str(CIR_CKPT_DIR / backbone_slug / f'fold{fold}.pt'),
                      map_location=device)
    model = CrossChainResidual(22, 64, 0.1).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    z_mu, z_std = ckpt['z_mu'], ckpt['z_std']

    valid = np.isfinite(base_preds)
    cir_preds = base_preds.copy()
    z = np.clip(((geom[valid] - z_mu) / z_std).astype(np.float32), -5.0, 5.0)
    imask = interface[valid].astype(np.float32)
    with torch.no_grad():
        cir_preds[valid] = model(
            torch.from_numpy(base_preds[valid].astype(np.float32)).to(device),
            torch.from_numpy(z).to(device),
            torch.from_numpy(imask).to(device),
            1.0
        ).cpu().numpy()
    return cir_preds


# ============================================================
# ProteinMPNN (zero-shot, 1 model)
# ============================================================

def run_proteinmpnn(df, device):
    from protein_mpnn_utils import ProteinMPNN, parse_PDB
    from Bio.PDB import PDBParser

    ckpt = torch.load(str(MPNN_DIR / 'vanilla_model_weights/v_48_020.pt'),
                      map_location=device)
    mpnn = ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        k_neighbors=ckpt['num_edges'], augment_eps=0.0, vocab=21)
    mpnn.load_state_dict(ckpt['model_state_dict'])
    mpnn.to(device).eval()

    pdb_path = str(DATA_DIR / 'PDBs' / '6M0J.pdb')
    parsed = parse_PDB(pdb_path)
    d = parsed[0]
    chains = sorted([k.replace('seq_chain_', '') for k in d.keys() if k.startswith('seq_chain_')])
    coords_list, chain_enc, res_idx_list, chain_starts, chain_seqs = [], [], [], {}, {}
    offset, global_idx = 0, 0
    for ch_idx, ch in enumerate(chains):
        cd = d[f'coords_chain_{ch}']
        n, ca, c, o = [np.array(cd[f'{a}_chain_{ch}']) for a in ['N','CA','C','O']]
        L_ch = len(n)
        coords_list.append(np.stack([n, ca, c, o], axis=1))
        chain_enc.extend([ch_idx + 1] * L_ch)
        res_idx_list.extend(range(offset, offset + L_ch))
        chain_starts[ch] = global_idx
        chain_seqs[ch] = d[f'seq_chain_{ch}']
        global_idx += L_ch; offset += L_ch

    coords_all = np.concatenate(coords_list, axis=0)
    seq_all = ''.join(d[f'seq_chain_{ch}'] for ch in chains)
    L = len(seq_all)
    gap_mask = np.any(np.isnan(coords_all.reshape(L, -1)), axis=1)
    coords_all = np.nan_to_num(coords_all, nan=0.0)

    bio_parser = PDBParser(QUIET=True)
    structure = bio_parser.get_structure('x', pdb_path)
    resseq_to_global = {}
    for ch in chains:
        if ch not in structure[0]: continue
        nongap = [chain_starts[ch] + i for i, c in enumerate(chain_seqs[ch]) if c != '-']
        residues = [(r.get_id()[1], r.get_id()[2].strip())
                    for r in structure[0][ch].get_residues() if r.get_id()[0] == ' ']
        if len(residues) == len(nongap):
            for (resseq, icode), gp in zip(residues, nongap):
                resseq_to_global[(ch, f"{resseq}{icode}" if icode else str(resseq))] = gp

    X = torch.tensor(coords_all, dtype=torch.float32).unsqueeze(0).to(device)
    S = torch.tensor([ALPHABET.index(aa) if aa in ALPHABET else 20 for aa in seq_all],
                     dtype=torch.long).unsqueeze(0).to(device)
    mask_t = torch.ones(1, L, dtype=torch.float32, device=device)
    mask_t[0, gap_mask] = 0.0
    chain_M = torch.ones(1, L, dtype=torch.float32, device=device)
    ridx = torch.tensor(res_idx_list, dtype=torch.long).unsqueeze(0).to(device)
    cenc = torch.tensor(chain_enc, dtype=torch.long).unsqueeze(0).to(device)
    randn = torch.zeros(1, L, dtype=torch.float32, device=device)
    with torch.no_grad():
        log_probs = mpnn(X, S, mask_t, chain_M, ridx, cenc, randn)

    preds = []
    for _, row in df.iterrows():
        wt, mt, pos = row['wt'], row['mt'], str(row['pdb_resnum'])
        gp = resseq_to_global.get(('E', pos))
        if gp is None or seq_all[gp] != wt:
            preds.append(np.nan); continue
        wi, mi = ALPHABET.index(wt), ALPHABET.index(mt)
        if wi < 0 or mi < 0:
            preds.append(np.nan); continue
        preds.append((log_probs[0, gp, wi] - log_probs[0, gp, mi]).item())
    return np.array(preds)


# ============================================================
# RDE-Network (5 fold models)
# ============================================================

def run_rde_single_fold(df, fold, device):
    """Run one fold of RDE on R3669."""
    old_cwd = os.getcwd()
    os.chdir(str(RDE_DIR))
    from rde.utils.misc import load_config
    from rde.models.rde_ddg import DDG_RDE_Network
    from rde.utils.protein.parsers import parse_biopython_structure
    from rde.utils.transforms import SelectAtom
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import one_to_index

    config, _ = load_config('configs/train/rde_ddg_skempi.yml')
    m = DDG_RDE_Network(config.model).to(device)
    ckpt = torch.load(str(BASE_CKPT_DIR / 'rde' / 'checkpoints' / f'fold{fold}_best.pt'),
                      map_location=device)
    m.load_state_dict(ckpt['model'])
    m.eval()

    pdb_path = str(DATA_DIR / 'PDBs' / '6M0J.pdb')
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(None, pdb_path)
    base_data, seq_map = parse_biopython_structure(structure[0])
    base_data = SelectAtom('backbone+CB')(base_data)

    group_id = [1 if ch == 'E' else 2 if ch == 'A' else 0 for ch in base_data['chain_id']]

    preds = []
    for _, row in df.iterrows():
        data = copy.deepcopy(base_data)
        data['group_id'] = torch.LongTensor(group_id)
        pos_key = ('E', row['pdb_resnum'], ' ')
        if pos_key not in seq_map:
            preds.append(np.nan); continue
        aa_mut = data['aa'].clone()
        try:
            aa_mut[seq_map[pos_key]] = one_to_index(row['mt'])
        except:
            preds.append(np.nan); continue
        data['aa_mut'] = aa_mut
        data['mut_flag'] = (data['aa'] != data['aa_mut'])
        data['ddG'] = torch.tensor(0.0)
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor)
                 else [v] if isinstance(v, list) else v
                 for k, v in data.items()}
        with torch.no_grad():
            try:
                _, out = m(batch)
                preds.append(out['ddG_pred'].item())
            except:
                preds.append(np.nan)

    os.chdir(old_cwd)
    return np.array(preds)


# ============================================================
# Pythia-PPI (5 fold models)
# ============================================================

def run_pythia_single_fold(df, fold, device):
    """Run one fold of Pythia on R3669."""
    old_cwd = os.getcwd()
    os.chdir(str(PYTHIA_DIR))
    sys.path.insert(0, str(PYTHIA_DIR))
    import importlib
    utils_pythia_model = importlib.import_module('utils.pythia.model')
    utils_pythia_pdb = importlib.import_module('utils.pythia.pdb_utils')
    from utils.model import Pythia_PPI, get_torch_model
    from utils.dataset import inference_process

    pdb_path = str(DATA_DIR / 'PDBs' / '6M0J.pdb')
    pythia_encoder = get_torch_model('./utils/pythia/pythia-p.pt', device)
    model = Pythia_PPI(pythia_encoder)
    ckpt = torch.load(str(BASE_CKPT_DIR / 'pythia' / 'checkpoints' / f'fold{fold}.pt'),
                      map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    feats, info = inference_process(pdb_path, '6M0J', device=device)
    with torch.no_grad():
        pred_1, _ = model(feats['wt_id'], feats['mt_id'], feats['node_in'], feats['edge_in'])
    all_preds = pred_1.cpu().numpy().flatten()

    # Match to R3669 entries using mt_position from original CSV
    r3669_orig = pd.read_csv('/data/yuwl/R3669/RBD.csv')
    chain_E_indices = [i for i, c in enumerate(info['chain']) if c == 'E']
    lookup = {(info['wt'][i], info['mt_position'][i], info['mt'][i]): i for i in chain_E_indices}

    matched = []
    for _, row in r3669_orig.iterrows():
        idx = lookup.get((row['wt'], row['mt_position'], row['mt']))
        matched.append(all_preds[idx] if idx is not None else np.nan)

    os.chdir(old_cwd)
    return np.array(matched)


# ============================================================
# DiffAffinity (pre-computed per-fold predictions from JAX)
# ============================================================

def get_diffaffinity_fold_preds():
    """Load pre-computed DiffAffinity 5-fold predictions."""
    npz_path = OUTPUT / 'diffaffinity_r3669_preds.npz'
    if not npz_path.exists():
        raise FileNotFoundError(
            f'{npz_path} not found. Run run_diffaffinity.py in env-diffaffinity first.')
    data = np.load(str(npz_path))
    return data['fold_preds']  # shape (5, N)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = args.device

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'logs').mkdir(exist_ok=True)
    print(f'Device: {device}\n')

    # Load data & geometry
    print('=== Loading R3669 ===')
    df = load_data()
    y_true = df['ddG'].values
    print('\n=== Computing geometry ===')
    geom, interface = compute_geometry(df)

    all_results = {}

    # --- ProteinMPNN (zero-shot: 1 base, 5 CIR folds) ---
    print(f'\n{"="*60}\n  ProteinMPNN\n{"="*60}')
    mpnn_base = run_proteinmpnn(df, device)
    print(f'  Scored: {np.isfinite(mpnn_base).sum()}/{len(mpnn_base)}')

    fold_metrics = []
    for fold in range(5):
        cir_preds = apply_cir(fold, 'proteinmpnn', mpnn_base, geom, interface, device)
        valid = np.isfinite(cir_preds)
        rho_ov, _ = spearmanr(y_true[valid], cir_preds[valid])
        rho_if, _ = spearmanr(y_true[interface & valid], cir_preds[interface & valid])
        fold_metrics.append({'overall_rho': rho_ov, 'interface_rho': rho_if})

    base_valid = np.isfinite(mpnn_base)
    rho_base, _ = spearmanr(y_true[base_valid], mpnn_base[base_valid])
    rho_base_if, _ = spearmanr(y_true[interface & base_valid], mpnn_base[interface & base_valid])
    avg_ov = np.mean([fm['overall_rho'] for fm in fold_metrics])
    avg_if = np.mean([fm['interface_rho'] for fm in fold_metrics])
    print(f'  Overall ρ:   base={rho_base:.4f}, +CIR={avg_ov:.4f}, Δ={avg_ov-rho_base:+.4f}')
    print(f'  Interface ρ: base={rho_base_if:.4f}, +CIR={avg_if:.4f}, Δ={avg_if-rho_base_if:+.4f}')
    for i, fm in enumerate(fold_metrics):
        print(f'    Fold {i}: ov={fm["overall_rho"]:.4f}, if={fm["interface_rho"]:.4f}')
    all_results['ProteinMPNN'] = {
        'base': {'overall_rho': rho_base, 'interface_rho': rho_base_if},
        'cir_avg': {'overall_rho': avg_ov, 'interface_rho': avg_if},
        'cir_per_fold': fold_metrics,
    }

    # --- RDE-Network (5 fold backbone + 5 fold CIR, paired) ---
    print(f'\n{"="*60}\n  RDE-Network\n{"="*60}')
    fold_metrics = []
    base_rhos_ov, base_rhos_if = [], []
    for fold in range(5):
        print(f'  Fold {fold}...', end=' ', flush=True)
        base = run_rde_single_fold(df, fold, device)
        valid = np.isfinite(base)
        rb, _ = spearmanr(y_true[valid], base[valid])
        rb_if, _ = spearmanr(y_true[interface & valid], base[interface & valid])
        base_rhos_ov.append(rb)
        base_rhos_if.append(rb_if)

        cir_preds = apply_cir(fold, 'rde_network', base, geom, interface, device)
        valid_c = np.isfinite(cir_preds)
        rc, _ = spearmanr(y_true[valid_c], cir_preds[valid_c])
        rc_if, _ = spearmanr(y_true[interface & valid_c], cir_preds[interface & valid_c])
        fold_metrics.append({'base_ov': rb, 'base_if': rb_if, 'cir_ov': rc, 'cir_if': rc_if})
        print(f'base_ov={rb:.4f}, cir_ov={rc:.4f}, base_if={rb_if:.4f}, cir_if={rc_if:.4f}')

    avg_base_ov = np.mean(base_rhos_ov)
    avg_base_if = np.mean(base_rhos_if)
    avg_cir_ov = np.mean([fm['cir_ov'] for fm in fold_metrics])
    avg_cir_if = np.mean([fm['cir_if'] for fm in fold_metrics])
    print(f'  Overall ρ:   base={avg_base_ov:.4f}, +CIR={avg_cir_ov:.4f}, Δ={avg_cir_ov-avg_base_ov:+.4f}')
    print(f'  Interface ρ: base={avg_base_if:.4f}, +CIR={avg_cir_if:.4f}, Δ={avg_cir_if-avg_base_if:+.4f}')
    all_results['RDE-Network'] = {
        'base': {'overall_rho': avg_base_ov, 'interface_rho': avg_base_if},
        'cir_avg': {'overall_rho': avg_cir_ov, 'interface_rho': avg_cir_if},
        'cir_per_fold': fold_metrics,
    }

    # --- Pythia-PPI (5 fold backbone + 5 fold CIR, paired) ---
    print(f'\n{"="*60}\n  Vanilla Pythia-PPI\n{"="*60}')
    fold_metrics = []
    base_rhos_ov, base_rhos_if = [], []
    for fold in range(5):
        print(f'  Fold {fold}...', end=' ', flush=True)
        base = run_pythia_single_fold(df, fold, device)
        valid = np.isfinite(base)
        rb, _ = spearmanr(y_true[valid], base[valid])
        rb_if, _ = spearmanr(y_true[interface & valid], base[interface & valid])
        base_rhos_ov.append(rb)
        base_rhos_if.append(rb_if)

        cir_preds = apply_cir(fold, 'vanilla_pythia_ppi', base, geom, interface, device)
        valid_c = np.isfinite(cir_preds)
        rc, _ = spearmanr(y_true[valid_c], cir_preds[valid_c])
        rc_if, _ = spearmanr(y_true[interface & valid_c], cir_preds[interface & valid_c])
        fold_metrics.append({'base_ov': rb, 'base_if': rb_if, 'cir_ov': rc, 'cir_if': rc_if})
        print(f'base_ov={rb:.4f}, cir_ov={rc:.4f}, base_if={rb_if:.4f}, cir_if={rc_if:.4f}')

    avg_base_ov = np.mean(base_rhos_ov)
    avg_base_if = np.mean(base_rhos_if)
    avg_cir_ov = np.mean([fm['cir_ov'] for fm in fold_metrics])
    avg_cir_if = np.mean([fm['cir_if'] for fm in fold_metrics])
    print(f'  Overall ρ:   base={avg_base_ov:.4f}, +CIR={avg_cir_ov:.4f}, Δ={avg_cir_ov-avg_base_ov:+.4f}')
    print(f'  Interface ρ: base={avg_base_if:.4f}, +CIR={avg_cir_if:.4f}, Δ={avg_cir_if-avg_base_if:+.4f}')
    all_results['Vanilla Pythia-PPI'] = {
        'base': {'overall_rho': avg_base_ov, 'interface_rho': avg_base_if},
        'cir_avg': {'overall_rho': avg_cir_ov, 'interface_rho': avg_cir_if},
        'cir_per_fold': fold_metrics,
    }

    # --- DiffAffinity (pre-computed per-fold from JAX) ---
    print(f'\n{"="*60}\n  DiffAffinity\n{"="*60}')
    try:
        diff_fold_preds = get_diffaffinity_fold_preds()  # (5, 3669)
        fold_metrics = []
        base_rhos_ov, base_rhos_if = [], []
        for fold in range(5):
            base = diff_fold_preds[fold]
            valid = np.isfinite(base)
            rb, _ = spearmanr(y_true[valid], base[valid])
            rb_if, _ = spearmanr(y_true[interface & valid], base[interface & valid])
            base_rhos_ov.append(rb)
            base_rhos_if.append(rb_if)

            cir_preds = apply_cir(fold, 'diffaffinity', base, geom, interface, device)
            valid_c = np.isfinite(cir_preds)
            rc, _ = spearmanr(y_true[valid_c], cir_preds[valid_c])
            rc_if, _ = spearmanr(y_true[interface & valid_c], cir_preds[interface & valid_c])
            fold_metrics.append({'base_ov': rb, 'base_if': rb_if, 'cir_ov': rc, 'cir_if': rc_if})
            print(f'  Fold {fold}: base_ov={rb:.4f}, cir_ov={rc:.4f}, base_if={rb_if:.4f}, cir_if={rc_if:.4f}')

        avg_base_ov = np.mean(base_rhos_ov)
        avg_base_if = np.mean(base_rhos_if)
        avg_cir_ov = np.mean([fm['cir_ov'] for fm in fold_metrics])
        avg_cir_if = np.mean([fm['cir_if'] for fm in fold_metrics])
        print(f'  Overall ρ:   base={avg_base_ov:.4f}, +CIR={avg_cir_ov:.4f}, Δ={avg_cir_ov-avg_base_ov:+.4f}')
        print(f'  Interface ρ: base={avg_base_if:.4f}, +CIR={avg_cir_if:.4f}, Δ={avg_cir_if-avg_base_if:+.4f}')
        all_results['DiffAffinity'] = {
            'base': {'overall_rho': avg_base_ov, 'interface_rho': avg_base_if},
            'cir_avg': {'overall_rho': avg_cir_ov, 'interface_rho': avg_cir_if},
            'cir_per_fold': fold_metrics,
        }
    except FileNotFoundError as e:
        print(f'  SKIPPED: {e}')

    # Save
    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n{"="*60}')
    print(f'Saved: {OUTPUT}/results.json')


if __name__ == '__main__':
    main()
