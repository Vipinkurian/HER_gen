#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent MatterSim CHE Delta-G_H for the top 25 consensus candidates.

  dG_H = E(slab+H) - E(slab) - 1/2 E(H2) + 0.24 eV

Slab+H and clean slab are relaxed with the same 2.0 A depth constraint used to
build the slabs. MatterSim is a general force field trained on PBE data, not an
adsorption model, so the result is an independent check of the sign and of the
distance from thermoneutral, with a possible systematic offset from the
BEEF-vdW-trained GNNs.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.build import molecule
import torch
from mattersim.forcefield import MatterSimCalculator

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from slab_common import fix_bottom_by_depth

CALC = MatterSimCalculator(device="cuda" if torch.cuda.is_available() else "cpu")
FREE_DEPTH, FMAX, STEPS, CORR = 2.0, 0.05, 200, 0.24
CIFDIR = BASE / "data/relaxed_slabH_v3"
N = 25


def energy(struct, freeze=True):
    at = AseAtomsAdaptor().get_atoms(struct)
    if freeze:
        fix_bottom_by_depth(at, FREE_DEPTH)
    at.calc = CALC
    BFGS(at, logfile=os.devnull).run(fmax=FMAX, steps=STEPS)
    return at.get_potential_energy()


h2 = molecule("H2"); h2.set_cell([12, 12, 12]); h2.center(); h2.calc = CALC
BFGS(h2, logfile=os.devnull).run(fmax=FMAX, steps=200)
E_H2 = h2.get_potential_energy()
print(f"E(H2) = {E_H2:.4f} eV  (1/2 = {E_H2/2:.4f})", flush=True)

cons = pd.read_csv(BASE / "predictions/predictions_consensus_v3_dG/consensus_sign_agree.csv")
cons = (cons[cons["both_converged"] & (cons["min_gamma"] >= 0.3)]
        .sort_values("mean_abs_dG").head(N).reset_index(drop=True))

rows = []
for _, c in cons.iterrows():
    try:
        sh = Structure.from_file(str(CIFDIR / c["relaxed_file_al"]))
        E_shH = energy(sh, True)
        slab = Structure.from_sites([s for s in sh if s.specie.symbol != "H"])
        E_s = energy(slab, True)
        dG = E_shH - E_s - 0.5 * E_H2 + CORR
    except Exception as e:
        print(f"  {c['formula']:10s} FAIL: {str(e)[:60]}", flush=True)
        continue
    rows.append(dict(bulk_file=c["bulk_file"], formula=c["formula"], tier=c["tier"],
                     miller=c["miller_al"], site=c["site_type_al"],
                     slab_atoms=int(c["slab_atoms_al"]),
                     dG_alignn=round(float(c["predicted_dG_al"]), 6),
                     dG_megnet=round(float(c["predicted_dG_mg"]), 6),
                     dG_mattersim=round(float(dG), 6), slabH_file=c["relaxed_file_al"]))
    print(f"{c['formula']:10s} al={c['predicted_dG_al']:+.3f} "
          f"mg={c['predicted_dG_mg']:+.3f} ms={dG:+.3f}", flush=True)

out = pd.DataFrame(rows)
out["all_sign_agree"] = ((np.sign(out["dG_alignn"]) == np.sign(out["dG_megnet"])) &
                         (np.sign(out["dG_megnet"]) == np.sign(out["dG_mattersim"])))
out["max3_abs"] = out[["dG_alignn", "dG_megnet", "dG_mattersim"]].abs().max(axis=1)
out.to_csv(BASE / "predictions/predictions_consensus_v3_dG/consensus_top25_3way.csv", index=False)

print("\n=== 3-way table (top consensus GNN candidates) ===")
print(out.drop(columns=["slabH_file"]).to_string(index=False))
print(f"\nMatterSim |dG| <= 0.2 : {(out['dG_mattersim'].abs() <= 0.2).sum()}/{len(out)}")
print(f"all-3 sign agree      : {int(out['all_sign_agree'].sum())}/{len(out)}")
print(f"all-3 within 0.2 eV   : {int((out['max3_abs'] <= 0.2).sum())}/{len(out)}")
print(f"MatterSim dG range    : {out['dG_mattersim'].min():+.3f} .. "
      f"{out['dG_mattersim'].max():+.3f}  mean {out['dG_mattersim'].mean():+.3f}")
