#!/usr/bin/env python3
"""MatterSim + Phonopy finite-displacement phonons for the eight DFT candidates.

Each bulk cell is relaxed with MatterSim (cell and positions, fmax = 0.01 eV/A). Atoms are
displaced by 0.03 A in a supercell of about 8.5 A per side (at most 340 atoms), and MatterSim
forces give the force constants. The script computes the band structure along the seekpath
high-symmetry path, the atom-projected DOS on a 16x16x16 mesh and the minimum frequencies, and
writes
    data/phonons/phonon_bands_dos.json   (band structure and projected DOS)
    data/phonons/phonon_summary.json     (supercell and minimum/maximum frequencies)

Usage: python 06_validation/compute_phonons.py [--out DIR]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "data" / "structures" / "bulk_novel_cifs"
CAND = {
    "Pd5Au": "structure_2612_E0.0048.cif",
    "Pd13Ru3": "structure_2245_E0.0874.cif",
    "TiPd5": "structure_0750_E-0.0103.cif",
    "Cu3Pd13": "structure_0577_E0.0136.cif",
    "Pt5Rh": "structure_1091_E0.0097.cif",
    "Cu5Pt3": "structure_1278_E0.0070.cif",
    "Cu2Pt3": "structure_0659_E0.0214.cif",
    "Pt7Pb": "structure_1858_E0.0449.cif",
}
SUMMARY_KEYS = ("bulk_cif", "supercell", "n_atoms_supercell",
                "min_freq_band_THz", "min_freq_mesh_THz", "max_freq_THz")


def run(cif, calc, target=8.5, max_atoms=340):
    from ase import Atoms
    from ase.filters import FrechetCellFilter
    from ase.io import read
    from ase.optimize import BFGS
    from phonopy import Phonopy
    from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
    from phonopy.structure.atoms import PhonopyAtoms

    at = read(str(cif))
    at.calc = calc
    BFGS(FrechetCellFilter(at), logfile=os.devnull).run(fmax=0.01, steps=400)
    n = [max(1, int(np.ceil(target / length))) for length in at.cell.lengths()]
    while len(at) * int(np.prod(n)) > max_atoms and max(n) > 1:
        n[int(np.argmax(n))] -= 1
    unit = PhonopyAtoms(symbols=at.get_chemical_symbols(), cell=at.cell.array,
                        scaled_positions=at.get_scaled_positions())
    ph = Phonopy(unit, supercell_matrix=np.diag(n))
    ph.generate_displacements(distance=0.03)
    forces = []
    for sc in ph.supercells_with_displacements:
        a = Atoms(symbols=sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
    ph.forces = forces
    ph.produce_force_constants()
    qpts, labels, conns = get_band_qpoints_by_seekpath(unit, 101)
    ph.run_band_structure(qpts, path_connections=conns, labels=labels)
    bsd = ph.get_band_structure_dict()
    ph.run_mesh([16, 16, 16], with_eigenvectors=True, is_mesh_symmetry=False)
    ph.run_projected_dos()
    pd = ph.get_projected_dos_dict()
    mesh = ph.get_mesh_dict()
    return dict(distances=[np.asarray(d) for d in bsd["distances"]],
                frequencies=[np.asarray(f) for f in bsd["frequencies"]],
                labels=list(labels), connections=list(conns),
                pdos_fp=np.asarray(pd["frequency_points"]),
                pdos=np.asarray(pd["projected_dos"]),
                syms=at.get_chemical_symbols(),
                minf=float(min(f.min() for f in bsd["frequencies"])),
                minf_mesh=float(mesh["frequencies"].min()),
                fmax=float(max(f.max() for f in bsd["frequencies"])),
                sc=list(n), natoms=len(at) * int(np.prod(n)))


def to_record(cif_name, r, nd=5):
    """Convert the result of run() into a JSON-serialisable record."""
    rnd = lambda a: np.round(np.asarray(a, dtype=float), nd).tolist()
    return {
        "bulk_cif": cif_name,
        "supercell": [int(x) for x in r["sc"]],
        "n_atoms_supercell": int(r["natoms"]),
        "min_freq_band_THz": round(float(r["minf"]), 6),
        "min_freq_mesh_THz": round(float(r["minf_mesh"]), 6),
        "max_freq_THz": round(float(r["fmax"]), 6),
        "band_structure": {
            "labels": list(r["labels"]),
            "path_connections": [bool(c) for c in r["connections"]],
            "distances": [rnd(d) for d in r["distances"]],
            "frequencies_THz": [rnd(f) for f in r["frequencies"]],
        },
        "projected_dos": {
            "frequency_points_THz": rnd(r["pdos_fp"]),
            "atom_symbols": list(r["syms"]),
            "dos_per_atom": rnd(r["pdos"]),
        },
    }


def write_outputs(records, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phonon_bands_dos.json").write_text(json.dumps(records))
    summary = {k: {key: v[key] for key in SUMMARY_KEYS} for k, v in records.items()}
    (out_dir / "phonon_summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "phonons")
    args = parser.parse_args()

    import torch
    from mattersim.forcefield import MatterSimCalculator
    calc = MatterSimCalculator(device="cuda" if torch.cuda.is_available() else "cpu")

    records = {}
    for name, cif in CAND.items():
        r = run(BULK / cif, calc)
        records[name] = to_record(cif, r)
        print(f"{name:8s} supercell {r['sc']} ({r['natoms']} atoms) "
              f"min band {r['minf']:+.4f} THz, min mesh {r['minf_mesh']:+.4f} THz", flush=True)
    write_outputs(records, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
