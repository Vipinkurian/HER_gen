#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare the ALIGNN dataset from the Mamun CSV.
Converts CIF strings -> POSCAR files + id_prop.csv

Output structure:
  data/alignn_dataset/
      id_prop.csv          (filename, reaction_energy)
      POSCAR-0001.vasp
      POSCAR-0002.vasp
      ...
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar
from tqdm import tqdm

# CONFIG

DATA_CSV   = "data/training/MamunHighT2019_SlabH_only_cleaned.csv"   # run from the repository root
OUT_DIR    = Path("data/alignn_dataset")
ENERGY_MIN = -5.0
ENERGY_MAX =  5.0

# MAIN

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("ALIGNN Dataset Preparation")
    print("=" * 55)

    # 1. Load CSV
    print("\nLoading dataset ...")
    try:
        df = pd.read_csv(DATA_CSV, sep="\t")
        if df.shape[1] < 3:
            raise ValueError
    except Exception:
        df = pd.read_csv(DATA_CSV)

    print("  Raw rows    : %d" % len(df))
    df = df.dropna(subset=["structure", "reactionEnergy"])
    df = df[(df["reactionEnergy"] >= ENERGY_MIN) &
            (df["reactionEnergy"] <= ENERGY_MAX)]
    df = df.reset_index(drop=True)
    print("  After filter: %d rows" % len(df))

    # 2. Convert each CIF string -> POSCAR file
    print("\nConverting CIF strings to POSCAR files ...")
    print("Output directory: %s" % OUT_DIR)

    records = []
    failed  = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Converting"):
        try:
            parser  = CifParser(StringIO(str(row["structure"])))
            structs = parser.get_structures(primitive=False)
            if not structs:
                failed += 1
                continue
            struct = structs[0]

            fname    = "POSCAR-%05d.vasp" % idx
            out_path = OUT_DIR / fname
            poscar   = Poscar(struct)
            poscar.write_file(str(out_path))

            records.append({
                "filename"       : fname,
                "reactionEnergy" : float(row["reactionEnergy"]),
            })

        except Exception as e:
            failed += 1
            continue

    print("\n  Converted : %d" % len(records))
    print("  Failed    : %d" % failed)

    # 3. Write id_prop.csv  (ALIGNN format: filename, target)
    id_prop = pd.DataFrame(records)
    id_prop.to_csv(str(OUT_DIR / "id_prop.csv"),
                   index=False, header=False)

    print("\nid_prop.csv written: %d entries" % len(id_prop))
    print("Target stats:")
    print("  Min  : %.4f eV" % id_prop["reactionEnergy"].min())
    print("  Max  : %.4f eV" % id_prop["reactionEnergy"].max())
    print("  Mean : %.4f eV" % id_prop["reactionEnergy"].mean())
    print("  Std  : %.4f eV" % id_prop["reactionEnergy"].std())

    # 4. Quick sanity check: verify 3 random files parse in jarvis-tools
    print("\nSanity check (3 random structures) ...")
    from jarvis.core.atoms import Atoms

    sample = id_prop.sample(3, random_state=42)
    for _, row in sample.iterrows():
        fpath = OUT_DIR / row["filename"]
        try:
            atoms = Atoms.from_poscar(str(fpath))
            print("  OK  %s  |  %s  |  %d atoms" % (
                row["filename"], atoms.composition, atoms.num_atoms))
        except Exception as e:
            print("  FAIL %s : %s" % (row["filename"], str(e)))

    # 5. Summary
    print("\n" + "=" * 55)
    print("DONE")
    print("=" * 55)
    print("Dataset ready at: %s/" % OUT_DIR)
    print("  POSCAR files : %d" % len(records))
    print("  id_prop.csv  : %d entries (no header, 2 columns)" % len(records))
    print("\nNext step: run the ALIGNN training script")


if __name__ == "__main__":
    main()
