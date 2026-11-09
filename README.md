# Generative Design and GNN Screening of Bimetallic HER Catalysts

Code and data accompanying the paper:

> **Generative Design and Graph Neural Network Screening of Novel Bimetallic
> Alloy Catalysts for the Hydrogen Evolution Reaction**
> V. Kurian Elavunkel, S. Le Tonquesse, H. Vroylandt.

This repository contains the end-to-end pipeline that couples diffusion-based
generative modelling (MatterGen), machine-learning force-field relaxation
(MatterSim), and two graph neural network surrogates (MEGNet and ALIGNN) to
discover novel bimetallic catalysts for the hydrogen evolution reaction (HER).

---

## Pipeline overview

```
MatterGen fine-tuning + generation   (1,569 bulk labels  -> 3,000 structures)
        |
Standardise + novelty selection      (1,151 stable+unique -> 730 database-novel
                                       binary alloys; both elements in the Mamun
                                       37-metal set, stoichiometry not pure/A3B/AB,
                                       no Mamun structural match)
        |
Surface-energy screen (MatterSim)    (rank 100/110/111 terminations by gamma;
                                       keep the 3 lowest-gamma per bulk)
        |
Slab + adsorption-site construction  (top/bridge/hollow, up to 0.25 ML)
   + MatterSim relaxation            (-> 39,132 slab+H configurations)
        |
GNN screening (MEGNet + ALIGNN)      (best site per bulk)
   + dual-model consensus            (GNN dE + 0.24 eV; r = 0.847 -> 77 consensus,
                                       50 sign-agree, 49 after removing non-converged)
        |
MatterSim CHE energy-difference      (orthogonal 3-way triangulation, top 25)
        |
DFT validation (Quantum ESPRESSO)    (8 candidates, 4 within +/-0.2 eV)
   + MatterSim phonons               (all 8 dynamically stable)
```

## Repository layout

| Directory | Stage | Key scripts |
|-----------|-------|-------------|
| `01_generation/`       | Standardise generated bulks, test Mamun overlap, select the 730 novel binaries | `stage1_standardize.py`, `mamun_overlap.py`, `make_binary_manifest.py` |
| `02_surface_screen/`   | MatterSim surface-energy screen; keep 3 lowest-gamma facets per bulk | `stage1c_surface_screen.py`, `slab_common.py` |
| `03_slab_construction/`| Slab + H enumeration (up to 0.25 ML) and MatterSim relaxation | `stage2_slabH.py`, `slab_common.py` |
| `04_gnn_training/`     | Fine-tuning / training of the GNN surrogates | `finetune_megnet_her.py`, `train_alignn_her.py`, `prepare_alignn_dataset.py`, `make_fig_gnn_benchmark.py` |
| `05_screening/`        | GNN prediction, dual-model consensus, MatterSim 3-way triangulation | `predict_megnet_v3.py`, `predict_alignn_v3.py`, `consensus_v3_dG.py`, `compute_mattersim_dG_v3dG.py`, `make_fig_screening_v3dG.py` |
| `data/training/`       | Training datasets (derived from Mamun, BEEF-vdW) | `finetuning_bulk_1569.csv` (MatterGen), `MamunHighT2019_SlabH_only_cleaned.csv` (surrogate source data), `gnn_training_id_prop.csv` + `gnn_training_poscars.tar.gz` (surrogates) |
| `data/structures/`     | Generated + relaxed structures (see below) | `generated_crystals_3000.extxyz` (all 3,000 generated), `bulk_novel_cifs/` (730), `relaxed_slabH_v3.tar.gz` (39,132), `bulk_manifest_binary.csv`, `surface_screen.csv`, `slabH_metadata_v3.csv` |
| `data/predictions/`    | Full GNN predictions, consensus, 3-way tables | `alignn_predictions_all.csv`, `megnet_predictions_all.csv`, `consensus_all.csv`, `consensus_sign_agree.csv`, `consensus_top25_3way.csv`, `best_per_bulk_*.csv` |
| `06_validation/`       | DFT adsorption free energies from the Quantum ESPRESSO outputs, MatterSim phonons of the 8 DFT candidates | `compute_dft_dG.py`, `compute_phonons.py` |
| `data/gnn_test/`       | GNN test-set predictions (1,330 held-out structures) and training curves | `alignn_test_preds.csv`, `megnet_test_preds.csv`, `alignn_history.pkl`, `megnet_metrics.csv` |
| `data/dft/`            | Quantum ESPRESSO input and output files, DFT adsorption free energies | `<compound>/slab_H/`, `<compound>/clean_slab/`, `H2/`, `dft_energies_dG.csv`, `dft_vs_ml_dG.csv` |
| `data/phonons/`        | MatterSim phonons of the 8 DFT candidates | `phonon_summary.json`, `phonon_bands_dos.json` |
| `models/`              | Trained GNN checkpoints used for screening | `megnet_best_model/`, `alignn_best_model.pt` |

Structure generation (stage 0) is done with the [MatterGen](https://github.com/microsoft/mattergen)
package, fine-tuned on `data/training/finetuning_bulk_1569.csv` (no custom script needed).
The generated bulks are then standardised and filtered to the 730 novel binaries by the
`01_generation/` scripts.

## Structures, predictions & reproducibility

Everything needed to reproduce the screening is included in the repo.

**Training data** (both derived from the Mamun high-throughput dataset, BEEF-vdW; cite
Mamun et al., *Sci. Data* 6, 76 (2019) and Catalysis-Hub.org):
- `data/training/finetuning_bulk_1569.csv` — the **MatterGen fine-tuning set**: 1,569 unique
  bulk alloy compositions, each with the embedded `structure` and `reactionEnergy` = the
  **minimum ΔE_H\*** over all sites for that bulk (the property MatterGen was
  conditioned on). Mamun reports raw electronic reaction energies without zero-point or
  entropy corrections (Mamun et al., Eq. 1 and Table 5).
- `data/training/MamunHighT2019_SlabH_only_cleaned.csv` — the **surrogate source data**: 8,864
  Mamun slab+H entries from Catalysis-Hub (columns `id`, `Equation`, `chemicalComposition`,
  `facet`, `reactionEnergy`, `sites`, `dftCode`, `dftFunctional`, `formulas`, `structure`).
  `reactionEnergy` is the raw BEEF-vdW electronic reaction energy (eV) for the reaction given in
  `Equation`. This is the file read by `04_gnn_training/prepare_alignn_dataset.py` and
  `finetune_megnet_her.py`; keeping entries with −5 ≤ `reactionEnergy` ≤ 5 eV and a structure gives
  the 8,863 training entries below.
- `data/training/gnn_training_id_prop.csv` + `gnn_training_poscars.tar.gz` — the **GNN
  surrogate training set** (slab+H POSCARs with BEEF-vdW electronic ΔE_H\*). The train/val/test split is
  reproduced from a fixed seed inside the training scripts. The trained checkpoints in
  `models/` are unchanged from the surrogates used throughout (MEGNet MAE 0.174 eV, R² 0.850;
  ALIGNN MAE 0.161 eV, R² 0.879 on a held-out set of 1,330 slab+H structures).

**Pipeline outputs:**
- `data/structures/bulk_novel_cifs/` — the **730 database-novel** generated bulk alloy CIFs,
  named `structure_<id>_E<hull>.cif`.
- `data/structures/bulk_manifest_binary.csv` — one row per novel bulk: formula, standardised
  space group, prototype, novelty tier, `in_mamun` flag, Mamun facet used for slab enumeration.
- `data/structures/surface_screen.csv` — the MatterSim surface-energy screen: every
  (100)/(110)/(111) termination with its surface energy (`gamma_eV_A2`, `gamma_J_m2`); the
  three lowest-gamma per bulk are carried forward.
- `data/structures/relaxed_slabH_v3.tar.gz` — the **39,132 MatterSim-relaxed slab+H** CIFs
  (the exact geometries the GNNs scored). Filenames encode the origin:
  `structure_<id>_E<hull>__<miller>__t<term>__<site>__<index>.cif`
  (e.g. `structure_0001_E0.0654__100__t3__ontop__0.cif`).
- `data/structures/slabH_metadata_v3.csv` — one row per slab+H config: filename → bulk
  formula, space group, Miller facet, termination, surface energy, site type, atom counts,
  number of fixed atoms, convergence.
- `data/predictions/{alignn,megnet}_predictions_all.csv` — raw GNN outputs (`predicted_dG` column = ΔE_H\*,
  the Mamun label scale) for all 39,132 configs; add +0.24 eV for ΔG_H\*; the **`relaxed_file` column is the CIF filename** above, so any structure is
  directly searchable. `best_per_bulk_*.csv` give the best-site table per bulk.
- `data/predictions/consensus_all.csv` — the merged ALIGNN+MEGNet best-site table on the
  ΔG scale (ΔE + 0.24 eV) for all
  730 bulks with `mean_abs_dG`, `max_abs_dG`, `sign_agree`, and `rank`;
  `consensus_sign_agree.csv` and `consensus_top.csv` are the filtered shortlists.
- `data/predictions/consensus_top25_3way.csv` — the top-25 consensus candidates with the
  orthogonal MatterSim CHE ΔG (`dG_mattersim`) alongside `dG_alignn`/`dG_megnet`.
- `models/` — trained MEGNet and ALIGNN checkpoints used for screening.

**Look up a structure by filename** (e.g. from a predictions row):
```
tar xzf data/structures/relaxed_slabH_v3.tar.gz relaxed_slabH_v3/<filename>.cif
```
**Regenerate from scratch:** the slab+H set can be rebuilt from the 730 novel bulk CIFs with
`02_surface_screen/stage1c_surface_screen.py` then `03_slab_construction/stage2_slabH.py`, and
scored with `05_screening/predict_{alignn,megnet}_v3.py` followed by `consensus_v3_dG.py`.

## DFT validation and phonons

- `data/dft/<compound>/slab_H/` and `data/dft/<compound>/clean_slab/`: Quantum ESPRESSO (pw.x)
  input and output files of the slab with adsorbed H and of the clean slab for the eight DFT
  candidates. `data/dft/H2/` holds the gas-phase H2 reference. All settings (functional, cutoffs,
  k-points, smearing, fixed atoms) are in the input files. The pseudopotentials are the pslibrary
  1.0.0 PBE PAW files named under `ATOMIC_SPECIES`; put them in `./pseudo` to rerun a calculation.
  Cluster-specific absolute paths were replaced by `./pseudo` and `./tmp`. Nothing else was changed.
- `data/dft/dft_energies_dG.csv`: final total energies and
  ΔG_H\* = E(slab+H) − E(slab) − ½E(H2) + 0.24 eV, written by `06_validation/compute_dft_dG.py`
  from the output files.
- `data/dft/dft_vs_ml_dG.csv`: DFT ΔG_H\* next to the ALIGNN, MEGNet and MatterSim values for the
  same site. `site_group` gives the two groups used in the paper (ΔE ≈ 0 sites and ΔG best sites).
- `data/phonons/phonon_summary.json` and `phonon_bands_dos.json`: MatterSim + Phonopy phonons of
  the eight bulk alloys (supercell, minimum band and mesh frequencies, band structure along the
  seekpath high-symmetry path, atom-projected DOS), computed with the settings in
  `06_validation/compute_phonons.py`.
- `data/gnn_test/`: reference and predicted ΔE_H\* for the 1,330 held-out structures
  (`alignn_test_preds.csv` and `megnet_test_preds.csv`, rows aligned) and the training curves read
  by `04_gnn_training/make_fig_gnn_benchmark.py`.
- `data/structures/generated_crystals_3000.extxyz`: all 3,000 structures generated by the
  fine-tuned MatterGen model.

## Methods summary

- **Generation.** MatterGen is fine-tuned on 1,569 unique bulk alloy compositions from the
  Mamun dataset, each labelled with the minimum hydrogen adsorption energy ΔE_H\* across all
  sites, and conditioned toward ΔE ≈ 0 eV. Of 3,000 generated structures, 1,151
  are stable and unique; 730 are **database-novel binary alloys** built from the training-set
  metals (both elements in the Mamun 37-metal set, stoichiometry not pure/A₃B/AB, no Mamun
  structural match).
- **Surface-energy screen.** For each novel bulk, (100)/(110)/(111) terminations are cut and
  relaxed with MatterSim, ranked by surface energy γ = (E_slab − N·E_bulk)/2A, and the three
  lowest-γ terminations per bulk are retained.
- **Slab + H.** Hydrogen is placed at symmetry-inequivalent on-top, bridge and hollow sites at
  up to 0.25 ML on the retained terminations, giving 39,132 configurations. Each is relaxed with
  MatterSim; the bottom of the slab is held fixed by depth (atoms below 2 Å from the top are
  frozen).
- **Surrogates.** MEGNet (pairwise) and ALIGNN (three-body, line-graph) are trained on the
  Mamun slab+H subset (BEEF-vdW). Their mutual agreement is used as an architecture-independent
  reliability filter, not to rank one above the other.
- **Consensus.** A candidate is retained only if both models place its best site within the
  thermoneutral window (max|ΔG_H*| ≤ 0.2 eV, with ΔG = GNN ΔE + 0.24 eV). Inter-model Pearson
  r = 0.847 across all 730 novel alloys; 77 consensus candidates (10.5%), 50 additionally
  sign-agreeing, 49 after removing non-converged surfaces. The top 25 are triangulated with
  orthogonal MatterSim CHE energy differences (13 of 25 within the window).

## Data availability

All code and data of the paper are in this repository: the 3,000 generated structures and the
730 novel bulk CIFs, the surface-energy screen, the 39,132 relaxed slab+H CIFs (compressed
archive), the GNN training data, test-set predictions and trained checkpoints, the full
ALIGNN/MEGNet predictions and consensus tables, the Quantum ESPRESSO input and output files of
the DFT calculations, and the MatterSim phonon results (see *Structures, predictions &
reproducibility* and *DFT validation and phonons*). External resources:

- **Mamun training dataset:** Catalysis-Hub.org (<https://www.catalysis-hub.org>).
- **MatterGen:** <https://github.com/microsoft/mattergen>.
- **MatterSim:** <https://github.com/microsoft/mattersim>.
- **MatGL (MEGNet):** <https://github.com/materialsvirtuallab/matgl>;
  **ALIGNN:** <https://github.com/usnistgov/alignn>.

## Requirements

See `requirements.txt`. The core dependencies are `mattergen`, `mattersim`, `matgl`, `alignn`,
`pymatgen`, `ase`, `torch`, `dgl`, `matminer`, and standard scientific-Python packages.

## Citation

If you use this code or data, please cite the paper:

> V. Kurian Elavunkel, S. Le Tonquesse, and H. Vroylandt,
> *Generative Design and Graph Neural Network Screening of Novel Bimetallic Alloy
> Catalysts for the Hydrogen Evolution Reaction*, arXiv:2610.41932 (2026).

```bibtex
@article{elavunkel2026her,
  title   = {Generative Design and Graph Neural Network Screening of Novel
             Bimetallic Alloy Catalysts for the Hydrogen Evolution Reaction},
  author  = {Kurian Elavunkel, Vipin and Le Tonquesse, Sylvain and Vroylandt, Hadrien},
  year    = {2026},
  journal = {arXiv preprint arXiv:2610.41932},
  doi     = {}
}
```

See `LICENSE` for terms.
