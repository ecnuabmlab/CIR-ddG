"""
DiffAffinity v2: Train on OUR unified complex-level 5-fold split (seed 2026).

Key difference from previous version:
- Uses our fold_assignments.json (not SkempiDatasetManager's internal split)
- Loads all data via SkempiDatasetManager, then manually splits by our fold
- Ensures all 6 models use identical fold assignments
"""

import scipy.linalg
import numpy as np
scipy.linalg.tril = np.tril
scipy.linalg.triu = np.triu

import jax
import jax.numpy as jnp
jnp.DeviceArray = jax.Array

import sys
import os
import json
import time
from typing import NamedTuple
from pathlib import Path

import haiku as hk
import optax
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

class Freezable_TrainState(NamedTuple):
    trainable_params: hk.Params
    non_trainable_params: hk.Params
    opt_state: optax.OptState

sys.path.insert(0, '.')
sys.path.insert(0, '/data/yuwl/paper1')

from score_sde.utils import restore
from context_generator.utils.misc import load_config
from context_generator.utils.skempi import SkempiDatasetManager
from context_generator.models.da_ddg import DDG_RDE_Network
from context_generator.models.da_encoder import RDEEncoder
from context_generator.modules.encoders.single import PerResidueEncoder
from context_generator.modules.encoders.pair import ResiduePairEncoder
from context_generator.modules.encoders.attn import GAEncoder

OUTPUT_DIR = Path('/data/yuwl/paper1/experiments_v3/00_base_retrain/diffaffinity')
COMMON_DIR = Path('/data/yuwl/paper1/experiments_v3/common')

print(f'=== DiffAffinity v2 (unified split, GPU): {jax.default_backend()} ===')
print(f'Devices: {jax.devices()}')

config, _ = load_config('context_generator/configs/train/da_ddg_skempi.yml')
rde_config, _ = load_config('context_generator/configs/train/diff.yml')

# Load complex_to_fold (the sole source of fold assignments)
with open(COMMON_DIR / 'complex_to_fold.json') as f:
    complex_to_fold = json.load(f)
print(f'Loaded complex_to_fold: {len(complex_to_fold)} complexes')

# Load ALL data using SkempiDatasetManager (across all its internal folds)
dataset_mgr = SkempiDatasetManager(config, num_cvfolds=5, num_workers=0)

# Collect ALL entries from the dataset manager
all_entries = []
for mgr_fold in range(5):
    for batch in dataset_mgr.get_val_loader(mgr_fold):
        pdbcodes = batch.pop('pdbcode', [])
        mutstrs = batch.pop('mutstr', [])
        for k, v in list(batch.items()):
            if isinstance(v, (list, int)):
                batch.pop(k)
        for i in range(len(pdbcodes)):
            key = f"{pdbcodes[i]}_{mutstrs[i]}"
            entry_batch = {k: v[i:i+1] for k, v in batch.items() if hasattr(v, 'shape')}
            entry_batch['_key'] = key
            entry_batch['_pdbcode'] = pdbcodes[i]
            entry_batch['_mutstr'] = mutstrs[i]
            all_entries.append(entry_batch)

print(f'Loaded {len(all_entries)} total entries from SkempiDatasetManager')

# Assign entries to folds by complex (NO dedup — all entries including repeated measurements)
# SkempiDatasetManager's internal complex field matches our complex_to_fold keys
# We need to figure out which complex each entry belongs to.
# The batch from SkempiDatasetManager has 'pdbcode' which maps to complex via entries.pkl
# Simpler: use the _pdbcode + chain info to find complex.
# Actually the SkempiDatasetManager's get_val_loader iterates over its own entries,
# which already have a 'complex' field internally. But batches don't expose it.
# Workaround: the 'complex' in complex_to_fold is format "1CSE_E_I" (= #Pdb field).
# The _pdbcode from batches is just "1CSE". We need the full complex name.
# Let's load entries.pkl to get pdbcode→complex mapping.

import pickle as _pkl
with open('SKEMPI_v2/SKEMPI_v2_cache/entries.pkl', 'rb') as _f:
    _rde_entries = _pkl.load(_f)
# Build (pdbcode, mutstr) → complex mapping
_key_to_complex = {}
for _e in _rde_entries:
    _k = f"{_e['pdbcode']}_{_e['mutstr']}"
    _key_to_complex[_k] = _e['complex']

entries_by_fold = {i: [] for i in range(5)}
unmapped = 0
for entry in all_entries:
    key = entry['_key']
    cplx = _key_to_complex.get(key)
    if cplx and cplx in complex_to_fold:
        entries_by_fold[complex_to_fold[cplx]].append(entry)
    else:
        unmapped += 1

total_mapped = sum(len(v) for v in entries_by_fold.values())
print(f'Mapped to folds: {total_mapped} entries (unmapped: {unmapped})')
for fi in range(5):
    print(f'  Fold {fi}: {len(entries_by_fold[fi])} entries')


def collate_entries(entries):
    """Stack individual entry batches into a batch."""
    batch = {}
    for k in entries[0]:
        if k.startswith('_'):
            continue
        batch[k] = jnp.concatenate([e[k] for e in entries], axis=0)
    return batch


# Define model (same as before)
def _forward(batch):
    batch_wt = {k: v for k, v in batch.items()}
    batch_mt = {k: v for k, v in batch.items()}
    batch_mt['aa'] = batch_mt['aa_mut']
    ddg_rde = DDG_RDE_Network(
        context_encoder=RDEEncoder(
            single_encoder=PerResidueEncoder(feat_dim=rde_config.model.encoder.node_feat_dim, max_num_atoms=5),
            masked_bias=hk.Embed(vocab_size=2, embed_dim=rde_config.model.encoder.node_feat_dim),
            pair_encoder=ResiduePairEncoder(feat_dim=rde_config.model.encoder.pair_feat_dim, max_num_atoms=5),
            attn_encoder=GAEncoder(**rde_config.model.encoder)),
        cfg=config,
        single_encoder_ddg=PerResidueEncoder(feat_dim=config.model.encoder.node_feat_dim, max_num_atoms=5, name='trainable_per_residue_encoder'),
        pair_encoder_ddg=ResiduePairEncoder(feat_dim=config.model.encoder.pair_feat_dim, max_num_atoms=5, name='trainable_residue_pair_encoder'),
        attn_encoder_ddg=GAEncoder(**config.model.encoder, name='trainable_ga_encoder'))
    h_wt = ddg_rde(batch_wt)
    h_mt = ddg_rde(batch_mt)
    H_mt, H_wt = h_mt.max(axis=1), h_wt.max(axis=1)
    dim = config.model.encoder.node_feat_dim
    ddg_readout = hk.Sequential([
        hk.Linear(dim, name='trainable_linear'), jax.nn.relu,
        hk.Linear(dim, name='trainable_linear_1'), jax.nn.relu,
        hk.Linear(1, name='trainable_linear_2')])
    ddg_pred = ddg_readout(H_mt - H_wt).squeeze(-1)
    ddg_pred_inv = ddg_readout(H_wt - H_mt).squeeze(-1)
    return ddg_pred, ddg_pred_inv

model = hk.transform(_forward)

# Init with first batch
first_batch = collate_entries(entries_by_fold[0][:32])
rng = jax.random.PRNGKey(2026)
init_params = model.init(rng=rng, batch=first_batch)

# Load encoder
encoder_state = restore('SidechainDiff_ckpt')
encoder_params = encoder_state.params_ema
for k in list(encoder_params.keys()):
    if k.startswith('torus_generator'):
        encoder_params.pop(k)
full_params = hk.data_structures.merge(init_params, encoder_params)
trainable_params, non_trainable_params = hk.data_structures.partition(
    lambda m, n, p: "trainable" in m or "ddg_rde__network" in m, full_params)

print(f'Trainable: {len(trainable_params)}, Frozen: {len(non_trainable_params)}')

# Loss and train step
def loss_fn(tp, ntp, batch):
    params = hk.data_structures.merge(tp, ntp)
    pred, pred_inv = model.apply(params, None, batch)
    return (jnp.mean(optax.l2_loss(pred, batch['ddG'])) +
            jnp.mean(optax.l2_loss(pred_inv, -batch['ddG']))) / 2

optimiser = optax.chain(optax.adam(3e-4), optax.clip_by_global_norm(1.0))

@jax.jit
def train_step(tp, ntp, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(tp, ntp, batch)
    updates, new_os = optimiser.update(grads, opt_state)
    new_tp = optax.apply_updates(tp, updates)
    return new_tp, new_os, loss

@jax.jit
def predict(params, batch):
    pred, _ = model.apply(params, None, batch)
    return pred

# Train per fold
MAX_ITERS = 30000
VAL_FREQ = 1000
PATIENCE = 10
BATCH_SIZE = 32

all_results = []
for test_fold in range(5):
    print(f'\n{"="*60}')
    print(f'Fold {test_fold}/5')
    print(f'{"="*60}')

    train_entries = []
    for fi in range(5):
        if fi != test_fold:
            train_entries.extend(entries_by_fold[fi])
    val_entries = entries_by_fold[test_fold]
    print(f'  Train: {len(train_entries)}, Val: {len(val_entries)}')

    # Re-init trainable params
    opt_state = optimiser.init(trainable_params)
    tp = trainable_params
    ntp = non_trainable_params

    best_val_loss = float('inf')
    best_tp = None
    patience_counter = 0
    rng_local = np.random.RandomState(2026 + test_fold)
    t0 = time.time()

    for it in range(1, MAX_ITERS + 1):
        # Random batch from train
        idx = rng_local.choice(len(train_entries), BATCH_SIZE, replace=False)
        batch = collate_entries([train_entries[i] for i in idx])
        tp, opt_state, loss = train_step(tp, ntp, opt_state, batch)

        if it % VAL_FREQ == 0:
            # Validate on all val entries (in batches)
            val_losses = []
            for vi in range(0, len(val_entries), BATCH_SIZE):
                vbatch = collate_entries(val_entries[vi:vi+BATCH_SIZE])
                vl = loss_fn(tp, ntp, vbatch)
                val_losses.append(float(vl))
            avg_vl = np.mean(val_losses)
            print(f'  iter {it}: val_loss={avg_vl:.4f} ({time.time()-t0:.0f}s)')

            if avg_vl < best_val_loss:
                best_val_loss = avg_vl
                best_tp = jax.tree_util.tree_map(lambda x: x.copy(), tp)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f'  Early stopping at iter {it}')
                    break

    # Predict val with best model
    if best_tp is not None:
        tp = best_tp
    best_params = hk.data_structures.merge(tp, ntp)

    fold_preds = []
    for vi in range(0, len(val_entries), BATCH_SIZE):
        vbatch_entries = val_entries[vi:vi+BATCH_SIZE]
        vbatch = collate_entries(vbatch_entries)
        preds = predict(best_params, vbatch)
        for i, entry in enumerate(vbatch_entries):
            fold_preds.append({
                'key': entry['_key'],
                'pdbcode': entry['_pdbcode'],
                'mutstr': entry['_mutstr'],
                'ddG': float(vbatch['ddG'][i]),
                'ddG_pred': float(preds[i]),
                'fold': test_fold,
            })

    all_results.extend(fold_preds)
    fold_df = pd.DataFrame(fold_preds)
    r = pearsonr(fold_df['ddG'], fold_df['ddG_pred'])[0]
    rho = spearmanr(fold_df['ddG'], fold_df['ddG_pred'])[0]
    print(f'  Fold {test_fold}: n={len(fold_preds)}, R={r:.4f}, rho={rho:.4f}')

    # Save checkpoint (trainable params for this fold)
    ckpt_dir = OUTPUT_DIR / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'fold{test_fold}_best.npz'
    flat_tp, tree_def = jax.tree_util.tree_flatten(best_tp if best_tp is not None else tp)
    np.savez(str(ckpt_path),
             *[np.array(x) for x in flat_tp],
             tree_def_str=repr(tree_def))
    print(f'  Saved checkpoint: {ckpt_path}')

# Save
df = pd.DataFrame(all_results)
r_all = pearsonr(df['ddG'], df['ddG_pred'])[0]
rho_all = spearmanr(df['ddG'], df['ddG_pred'])[0]
print(f'\n=== Final (unified complex-level 5-fold) ===')
print(f'  Overall: R={r_all:.4f}, rho={rho_all:.4f} (target ~0.669/0.556)')
print(f'  N entries: {len(df)}')

(OUTPUT_DIR / 'predictions').mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_DIR / 'predictions' / 'predictions.csv', index=False)
print(f'Saved to {OUTPUT_DIR}/predictions/predictions.csv')
