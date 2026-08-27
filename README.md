# CIR-DDG

**Backbone-agnostic residual correction of antibody–antigen affinity changes (ΔΔG) with explicit cross-chain geometry.**

![CIR-DDG framework](assets/framework.png)

CIR-DDG treats ΔΔG prediction as *output-level repair*: a frozen base predictor is combined with a masked additive residual driven by 22 explicit physical descriptors of the mutation-centered interface (9 cross-chain distance features, 9 contact-count features, 4 partner-chemistry features). The correction is gated by a scope mask so that it activates only on single-point mutations at antibody–antigen interfaces (cross-chain distance ≤ 5 Å); predictions outside this cohort are conserved by construction.

The module reads only a scalar base prediction and a fixed descriptor vector, so it can be attached to predictors whose internals are unavailable, retrains in minutes, and can be audited feature by feature.

This repository contains the code and experiment scripts for the paper:

> W. Yu and X. Chen, *CIR-DDG: backbone-agnostic residual correction of antibody–antigen affinity changes with explicit cross-chain geometry* (manuscript under review).

## Repository layout

```
src/                        CIR-DDG core library
  geometry.py               22-dim cross-chain interface descriptor (pure structural, no learned parameters)
  model.py                  residual adapter (22→64→1 MLP, GELU, dropout 0.1) and gating
  train.py                  training with complex-level CV, Pearson guard, early stopping
  evaluate.py               metrics (Pearson/Spearman, calibrated RMSE/MAE, AUROC)
data/
  d5b2_all_skempi_annotated_zero_shot.csv   curated SKEMPI 2.0 entries (6,706 measurements)
  interface_geometry_all.npz                precomputed 22-dim descriptors for all entries
experiments/
  common/                   complex→fold assignment (343 complexes, 5 folds, seed 2026) and entry metadata
  00_base_retrain/          training/inference scripts for the six backbones
                            (ProteinMPNN, ESM-IF, RDE-Network, DiffAffinity, DDAffinity, Vanilla Pythia-PPI)
  01_main/                  main tables: train CIR-DDG per backbone × fold, evaluate overall /
                            per-structure / antibody–antigen interface cohorts
  04_r2_mechanism/          incremental-R² probing analysis (does geometry carry information
                            not encoded by the backbone?)
  05_ablation/              equal-capacity zero-input control and feature-group ablations
                            (distance 9 / contact 9 / chemistry 4)
  07_external_validation/   zero-shot transfer to the SARS-CoV-2 RBD–ACE2 DMS benchmark
                            (3,669 substitutions, PDB 6M0J; standardized descriptors clipped to [−5, 5])
  08_case_study/            3HFM (HyHEL-10–lysozyme) case study and figures
```

Trained checkpoints (≈280 MB) are included: retrained backbone weights under `experiments/00_base_retrain/*/checkpoints/` and the fold-specific CIR adapters under `experiments/01_main/checkpoints/`. Raw training logs and result files (predictions, tables, JSON metrics) are not tracked; each experiment script regenerates them (see **Data availability** below).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Core experiments need only PyTorch, NumPy, pandas, SciPy and scikit-learn. Two backbones have extra requirements (see `requirements.txt` comments): DiffAffinity needs a JAX/Haiku environment, and ESM-IF needs `fair-esm`. Neither is needed to train or evaluate CIR-DDG itself on precomputed backbone predictions.

## Quick start

Train the CIR-DDG residual module on top of precomputed backbone predictions:

```bash
python src/train.py \
    --base-predictions <backbone_predictions>.npz \   # keys: keys, ddg, ddg_pred, complexes
    --geometry data/interface_geometry_all.npz \      # keys: keys, geom
    --output results.json
```

The interface mask is auto-detected from the descriptor (minimum cross-chain distance ≤ 5 Å); a custom mask can be passed with `--interface-mask`. Training uses complex-level cross-validation so that no complex appears in both training and evaluation.

The experiment folders mirror the paper's analysis chain: main results (`01_main`), mechanism probing (`04_r2_mechanism`), ablations (`05_ablation`), external zero-shot validation (`07_external_validation`) and the case study (`08_case_study`). Each folder contains a runnable script (`run.py` / `generate_tables.py`) that regenerates the exact result files reported in the paper.

## Data availability

- SKEMPI 2.0: Jankauskaite et al. (2019), publicly available from the source described in the paper.
- External benchmark: DMS measurements from Starr et al. (2022); structure PDB 6M0J.
- Trained checkpoints: included in this repository (see Repository layout). Raw logs: available on request.

## License

MIT (see `LICENSE`).

## Citation

If you use CIR-DDG, please cite:

```bibtex
@article{yu2026cirddg,
  title   = {CIR-DDG: backbone-agnostic residual correction of antibody--antigen affinity changes with explicit cross-chain geometry},
  journal = {manuscript under review},
  year    = {2026}
}
```
