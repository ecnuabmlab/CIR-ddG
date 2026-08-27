"""DiffAffinity inference on R3669 (6M0J). Run with env-diffaffinity."""
import scipy.linalg
import numpy as np
scipy.linalg.tril = np.tril
scipy.linalg.triu = np.triu

import sys, os, json, time
import pandas as pd
import jax
import jax.numpy as jnp
jnp.DeviceArray = jax.Array
import haiku as hk
from pathlib import Path
from scipy.stats import spearmanr

os.chdir('/data/yuwl/paper1/CIR-DDG/baselines/DiffAffinity')
sys.path.insert(0, '.')

sys.modules['tensorboard'] = type(sys)('tensorboard')
sys.modules['torch.utils.tensorboard'] = type(sys)('torch.utils.tensorboard')

from context_generator.utils.misc import load_config
from context_generator.models.da_ddg import DDG_RDE_Network
from context_generator.models.da_encoder import RDEEncoder
from context_generator.modules.encoders.single import PerResidueEncoder
from context_generator.modules.encoders.pair import ResiduePairEncoder
from context_generator.modules.encoders.attn import GAEncoder
from score_sde.utils import restore
from prediction import _6M0JDataset
from torch.utils.data import DataLoader

CKPT_DIR = Path('/data/yuwl/paper1/experiments/00_base_retrain/diffaffinity/checkpoints')
OUTPUT = Path('/data/yuwl/paper1/experiments/07_external_validation')

config, _ = load_config('context_generator/configs/inference/6m0j.yml')
rde_config, _ = load_config('context_generator/configs/train/diff.yml')

print(f'JAX backend: {jax.default_backend()}, devices: {jax.devices()}')

dataset = _6M0JDataset(
    pdb_path='/data/yuwl/paper1/CIR-DDG/data/R3669_RBD/PDBs/6M0J.pdb',
    mutations='/data/yuwl/paper1/CIR-DDG/data/R3669_RBD/r3669_diffaffinity.csv',
)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
print(f'Dataset: {len(dataset)} entries')

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

ibatch = next(iter(loader))
ibatch_jax = {k: jnp.array(v.numpy()) for k, v in ibatch.items() if hasattr(v, 'numpy')}
rng = jax.random.PRNGKey(2026)
init_params = model.init(rng=rng, batch=ibatch_jax)

encoder_state = restore('SidechainDiff_ckpt')
encoder_params = encoder_state.params_ema
for k in list(encoder_params.keys()):
    if k.startswith('torus_generator'):
        encoder_params.pop(k)
full_params = hk.data_structures.merge(init_params, encoder_params)
trainable_params, non_trainable_params = hk.data_structures.partition(
    lambda m, n, p: "trainable" in m or "ddg_rde__network" in m, full_params)

print(f'Encoder loaded')

all_fold_preds = []
for fold in range(5):
    ckpt_path = CKPT_DIR / f'fold{fold}_best.npz'
    data = np.load(str(ckpt_path), allow_pickle=True)
    flat_arrays = [jnp.array(data[f'arr_{i}']) for i in range(len(data.files) - 1)]
    _, tree_def = jax.tree_util.tree_flatten(trainable_params)
    tp_restored = jax.tree_util.tree_unflatten(tree_def, flat_arrays)
    params = hk.data_structures.merge(tp_restored, non_trainable_params)

    preds = []
    for batch in loader:
        batch_jax = {k: jnp.array(v.numpy()) for k, v in batch.items() if hasattr(v, 'numpy')}
        pred, _ = model.apply(params, None, batch_jax)
        preds.extend(pred.tolist())
    all_fold_preds.append(np.array(preds))
    print(f'  Fold {fold}: {len(preds)} predictions')

r3669_diff = pd.read_csv('/data/yuwl/paper1/CIR-DDG/data/R3669_RBD/r3669_diffaffinity.csv')
y_true = r3669_diff['delta_bind'].values[:len(all_fold_preds[0])]

for fold in range(5):
    rho, _ = spearmanr(y_true, all_fold_preds[fold])
    print(f'  Fold {fold} ρ = {rho:.4f}')

np.savez(str(OUTPUT / 'diffaffinity_r3669_preds.npz'),
         fold_preds=np.array(all_fold_preds), y_true=y_true)
print(f'Saved: {OUTPUT}/diffaffinity_r3669_preds.npz')
