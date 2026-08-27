"""
ProteinMPNN & ESM-IF zero-shot inference for ALL 6706 SKEMPI entries (v3).
Handles both single and multi-mutations.

For multi-mut: score = sum of per-position logit_delta
"""
import os, sys, re, torch, numpy as np, pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

MPNN_DIR = Path('/data/yuwl/paper1/CIR-DDG/baselines/ProteinMPNN')
DATA_DIR = Path('/data/yuwl/paper1/CIR-DDG/data/SKEMPI2/PDBs')
OUTPUT_DIR = Path('/data/yuwl/paper1/experiments_v3/00_base_retrain')

sys.path.insert(0, str(MPNN_DIR))
from protein_mpnn_utils import ProteinMPNN, parse_PDB

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def load_mpnn_model(device):
    ckpt = torch.load(str(MPNN_DIR / 'vanilla_model_weights/v_48_020.pt'), map_location=device)
    model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                        k_neighbors=ckpt['num_edges'], augment_eps=0.0, vocab=21)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model


def parse_pdb_structure(pdb_path):
    """Parse PDB, return (input_tensors, seq_all, chain_residue_map)."""
    parsed = parse_PDB(str(pdb_path))
    if not parsed:
        return None
    d = parsed[0]
    chains = sorted([k.replace('seq_chain_', '') for k in d.keys() if k.startswith('seq_chain_')])

    coords_list, chain_enc, res_idx = [], [], []
    # Build (chain, local_resnum) → global_index mapping
    # local_resnum is 0-based sequential index within chain
    chain_residue_map = {}  # (chain_letter, pdb_resnum) → global_idx — we'll build this from sequence
    offset = 0
    global_idx = 0

    for ch_idx, ch in enumerate(chains):
        cd = d[f'coords_chain_{ch}']
        n = np.array(cd[f'N_chain_{ch}'])
        ca = np.array(cd[f'CA_chain_{ch}'])
        c = np.array(cd[f'C_chain_{ch}'])
        o = np.array(cd[f'O_chain_{ch}'])
        L_ch = len(n)
        coords_list.append(np.stack([n, ca, c, o], axis=1))
        chain_enc.extend([ch_idx + 1] * L_ch)
        res_idx.extend(range(offset, offset + L_ch))
        # Map: (chain, sequential_index) → global_index
        for local_i in range(L_ch):
            chain_residue_map[(ch, local_i)] = global_idx
            global_idx += 1
        offset += L_ch

    coords_all = np.concatenate(coords_list, axis=0)
    seq_all = ''.join(d[f'seq_chain_{ch}'] for ch in chains)
    L = len(seq_all)

    return {
        'coords': coords_all, 'seq': seq_all, 'L': L,
        'chains': chains, 'chain_residue_map': chain_residue_map,
        'chain_enc': chain_enc, 'res_idx': res_idx,
        'chain_lengths': {ch: len(np.array(d[f'coords_chain_{ch}'][f'N_chain_{ch}'])) for ch in chains},
    }


def resolve_mutation_positions(pdb_info, mutstr):
    """
    Resolve mutation string to list of (global_pos, wt_aa, mt_aa).
    mutstr format: 'LI38G' or 'RI48A,RI46A'
    Token format: {chain}{wt}{resnum}{mt}

    Resolution: find the position in the chain where the residue matches wt_aa.
    PDB residue numbering may not be sequential, so we search by matching.
    """
    mutations = []
    tokens = mutstr.split(',')
    chains = pdb_info['chains']
    seq = pdb_info['seq']

    # Build chain_start offsets
    chain_starts = {}
    offset = 0
    for ch in chains:
        chain_starts[ch] = offset
        offset += pdb_info['chain_lengths'][ch]

    for token in tokens:
        m = re.match(r'([A-Z])([A-Z])(\d+)([A-Z])', token)
        if not m:
            return None
        chain, wt, resnum, mt = m.groups()
        resnum = int(resnum)

        if chain not in chain_starts:
            return None

        # Search for the position: within this chain, find where seq matches wt
        # PDB residue numbers may not be 0-indexed, so we try:
        # 1. Direct index (resnum as 0-based within chain)
        # 2. Search for wt_aa near expected position
        ch_start = chain_starts[chain]
        ch_len = pdb_info['chain_lengths'][chain]

        # Try resnum-1 as 0-based index
        if 0 <= resnum - 1 < ch_len:
            global_pos = ch_start + resnum - 1
            if global_pos < len(seq) and seq[global_pos] == wt:
                mutations.append((global_pos, wt, mt))
                continue

        # Try resnum directly as 0-based
        if 0 <= resnum < ch_len:
            global_pos = ch_start + resnum
            if global_pos < len(seq) and seq[global_pos] == wt:
                mutations.append((global_pos, wt, mt))
                continue

        # Search within chain for matching wt_aa
        found = False
        for local_i in range(ch_len):
            global_pos = ch_start + local_i
            if global_pos < len(seq) and seq[global_pos] == wt:
                # Check if this is close to expected position
                if abs(local_i - resnum) < 5 or abs(local_i - (resnum-1)) < 5:
                    mutations.append((global_pos, wt, mt))
                    found = True
                    break
        if not found:
            return None

    return mutations


def score_mpnn(model, pdb_info, mutations, device):
    """Score mutations using ProteinMPNN. Returns sum of logit_deltas."""
    L = pdb_info['L']
    X = torch.tensor(pdb_info['coords'], dtype=torch.float32).unsqueeze(0).to(device)
    S = torch.tensor([ALPHABET.index(aa) if aa in ALPHABET else 20 for aa in pdb_info['seq']],
                     dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.ones(1, L, dtype=torch.float32).to(device)
    chain_M = torch.ones(1, L, dtype=torch.float32).to(device)
    residue_idx = torch.tensor(pdb_info['res_idx'], dtype=torch.long).unsqueeze(0).to(device)
    chain_encoding = torch.tensor(pdb_info['chain_enc'], dtype=torch.long).unsqueeze(0).to(device)
    randn = torch.zeros(1, L, dtype=torch.float32).to(device)

    with torch.no_grad():
        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding, randn)

    total_delta = 0.0
    for pos, wt, mt in mutations:
        if pos >= L:
            return None
        wt_idx = ALPHABET.index(wt) if wt in ALPHABET else -1
        mt_idx = ALPHABET.index(mt) if mt in ALPHABET else -1
        if wt_idx < 0 or mt_idx < 0:
            return None
        total_delta += (log_probs[0, pos, wt_idx] - log_probs[0, pos, mt_idx]).item()

    return total_delta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = args.device

    meta = pd.read_csv('/data/yuwl/paper1/experiments_v3/common/entries_meta.csv')
    print(f'Total entries: {len(meta)} (single={len(meta[meta.num_muts==1])}, multi={len(meta[meta.num_muts>1])})')

    model = load_mpnn_model(device)
    print('ProteinMPNN model loaded')

    results = []
    failed = 0
    pdb_cache = {}

    for pdb_id, group in tqdm(meta.groupby('pdbcode'), desc='Scoring'):
        pdb_path = DATA_DIR / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            failed += len(group)
            for _, row in group.iterrows():
                results.append({'key': f"{row['pdbcode']}_{row['mutstr']}", 'ddG': row['ddG'], 'ddG_pred': np.nan,
                                'num_muts': row['num_muts'], 'complex': row['complex']})
            continue

        if pdb_id not in pdb_cache:
            pdb_info = parse_pdb_structure(pdb_path)
            pdb_cache[pdb_id] = pdb_info
        else:
            pdb_info = pdb_cache[pdb_id]

        if pdb_info is None:
            failed += len(group)
            for _, row in group.iterrows():
                results.append({'key': f"{row['pdbcode']}_{row['mutstr']}", 'ddG': row['ddG'], 'ddG_pred': np.nan,
                                'num_muts': row['num_muts'], 'complex': row['complex']})
            continue

        for _, row in group.iterrows():
            mutations = resolve_mutation_positions(pdb_info, row['mutstr'])
            if mutations is None:
                results.append({'key': f"{row['pdbcode']}_{row['mutstr']}", 'ddG': row['ddG'], 'ddG_pred': np.nan,
                                'num_muts': row['num_muts'], 'complex': row['complex']})
                failed += 1
                continue

            score = score_mpnn(model, pdb_info, mutations, device)
            results.append({'key': f"{row['pdbcode']}_{row['mutstr']}", 'ddG': row['ddG'],
                            'ddG_pred': score if score is not None else np.nan,
                            'num_muts': row['num_muts'], 'complex': row['complex']})
            if score is None:
                failed += 1

    df = pd.DataFrame(results)
    valid = df['ddG_pred'].notna().sum()
    print(f'\nDone: {valid}/{len(df)} scored, {failed} failed')

    outpath = OUTPUT_DIR / 'proteinmpnn' / 'predictions' / 'predictions.csv'
    df.to_csv(outpath, index=False)
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()
