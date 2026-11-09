#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consensus selection on the free-energy scale.

Mamun labels (and hence GNN predictions) are electronic adsorption energies
Delta E_H (Mamun et al., Sci. Data 6, 76, Eq. 1 and Table 5: no ZPE/entropy).
They are converted to Delta G_H = Delta E_H + 0.24 eV (CHE) before selection.

For each model take the min-|dG| site per bulk, merge on bulk_file, and keep
bulks where BOTH models' best site is within the thermoneutral window.
The `converged` and `gamma_J_m2` columns are carried over so that structures
with non-converged relaxations can be excluded.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ALIGNN_CSV = BASE / "predictions/predictions_alignn_v3/predictions_all.csv"
MEGNET_CSV = BASE / "predictions/predictions_megnet_v3/predictions_all.csv"
OUT = BASE / "predictions/predictions_consensus_v3_dG"
CHE = 0.24   # eV, ZPE + entropy correction for H* (Norskov 2005)
OUT.mkdir(parents=True, exist_ok=True)

COMMON = ["bulk_file", "formula", "tier"]
MODEL  = ["miller", "site_type", "site_index", "coverage_ML", "slab_atoms",
          "gamma_J_m2", "converged", "relaxed_file", "predicted_dG", "abs_dG"]


def best_per_bulk(path, tag):
    df = pd.read_csv(path)
    df["predicted_dG"] = df["predicted_dG"] + CHE
    df["abs_dG"] = df["predicted_dG"].abs()
    b = df.loc[df.groupby("bulk_file")["abs_dG"].idxmin()].reset_index(drop=True)
    b = b[COMMON + MODEL].rename(columns={c: f"{c}_{tag}" for c in MODEL})
    return b


al = best_per_bulk(ALIGNN_CSV, "al")
mg = best_per_bulk(MEGNET_CSV, "mg").drop(columns=["formula", "tier"])
m = pd.merge(al, mg, on="bulk_file", how="inner")

m["mean_abs_dG"] = (m["abs_dG_al"] + m["abs_dG_mg"]) / 2
m["max_abs_dG"]  = m[["abs_dG_al", "abs_dG_mg"]].max(axis=1)
m["sign_agree"]  = np.sign(m["predicted_dG_al"]) == np.sign(m["predicted_dG_mg"])
m["both_converged"] = (m["converged_al"] == "OK") & (m["converged_mg"] == "OK")
m["min_gamma"] = m[["gamma_J_m2_al", "gamma_J_m2_mg"]].min(axis=1)
m = m.sort_values("mean_abs_dG").reset_index(drop=True)
m["rank"] = m.index + 1
m.to_csv(OUT / "consensus_all.csv", index=False)

r = float(np.corrcoef(m["predicted_dG_al"], m["predicted_dG_mg"])[0, 1])
print(f"bulks in both models   : {len(m)}")
print(f"inter-model Pearson r  : {r:.3f}")
print("consensus hit rates (max|dG| over both models):")
for thr in (0.1, 0.2, 0.3):
    n = int((m["max_abs_dG"] <= thr).sum())
    print(f"  max|dG| <= {thr} eV : {n:3d} ({100*n/len(m):.1f}%)")

top = m[m["max_abs_dG"] <= 0.2].copy()
top.to_csv(OUT / "consensus_top.csv", index=False)
sa = top[top["sign_agree"]].copy()
sa.to_csv(OUT / "consensus_sign_agree.csv", index=False)

print(f"\nconsensus (max|dG| <= 0.2) : {len(top)}")
print(f"  ...sign-agreeing        : {len(sa)}")
print(f"  ...+ both converged     : {int(sa['both_converged'].sum())}")
print(f"  ...+ gamma >= 0.3 J/m2  : {int((sa['both_converged'] & (sa['min_gamma'] >= 0.3)).sum())}")

flag = sa[~sa["both_converged"] | (sa["min_gamma"] < 0.3)]
if len(flag):
    print("\nFLAGGED sign-agreeing candidates (NOT_CONVERGED or gamma<0.3):")
    for _, x in flag.iterrows():
        why = []
        if not x["both_converged"]:
            why.append("NOT_CONVERGED")
        if x["min_gamma"] < 0.3:
            why.append(f"gamma={x['min_gamma']:.3f}")
        print(f"  {x['formula']:12s} al={x['predicted_dG_al']:+.3f} "
              f"mg={x['predicted_dG_mg']:+.3f}  [{', '.join(why)}]")

cols = ["rank", "formula", "tier", "predicted_dG_al", "predicted_dG_mg",
        "mean_abs_dG", "both_converged", "min_gamma",
        "miller_al", "site_type_al", "slab_atoms_al"]
clean = sa[sa["both_converged"] & (sa["min_gamma"] >= 0.3)]
print(f"\nClean sign-agreeing consensus candidates ({len(clean)}), by mean|dG|:")
print(clean[cols].to_string(index=False))
