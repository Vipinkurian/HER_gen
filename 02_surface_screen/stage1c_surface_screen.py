#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage1c_surface_screen.py -- rank (facet, termination) pairs by surface energy.

WHY: SlabGenerator.get_slabs() returns every symmetrically distinct cut along c.
For a low-symmetry intermetallic that can be 20 per facet, most of them slicing
through dense atomic planes at high surface energy -- surfaces that would never
form. Enumerating H sites on all of them gave ~290k slab+H structures with site
counts varying 100x between compositions (Ga3Pt 1037 vs AlCu3 9), which biases
any min(dG_H) screening toward whichever composition happened to enumerate more.

This script relaxes each CLEAN slab (no adsorbate) with MatterSim and ranks by
approximate surface energy:

    gamma ~ (E_slab - N_slab * E_bulk_per_atom) / (2 * A)

CAVEAT: get_slabs(symmetrize=False) builds every slab from whole oriented unit
cells, so each slab has the bulk composition and no chemical potentials are
needed (the `stoichiometric` column is 1 for all 9,128 rows of the screen).
The approximation is that the two faces of a slab are generally inequivalent
(78% of the kept surfaces), so gamma is their mean, and the bottom face is not
relaxed. Used here purely as a RANKING HEURISTIC to prune obviously
unfavourable cuts, not as a reported surface energy.

Termination indices depend on the pymatgen version. Run this script and
stage2_slabH.py with the same version (pinned in requirements.txt), or the
(miller, term_index) keys will not refer to the same slabs.

Output: data/surface_screen.csv, consumed by stage2_slabH.py.

Usage:
    python 02_surface_screen/stage1c_surface_screen.py [--limit N] [--keep-per-bulk 3]
"""
import argparse
import csv
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.optimize import BFGS
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

sys.path.insert(0, str(Path(__file__).parent))
from slab_common import (layer_groups, n_top_atoms, slab_thickness,
                         supercell_to_coverage, fix_bottom_layers,
                         fix_bottom_by_depth)

warnings.filterwarnings("ignore")

STD_DIR = Path("data/structures/bulk_std")
MANIFEST = Path("data/bulk_manifest_binary.csv")
OUT_CSV = Path("data/surface_screen.csv")
CHKPT = Path("data/surface_screen_checkpoint.csv")

MILLER_INDICES = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
MIN_SLAB_A = 8.0
VACUUM_A = 15.0
MIN_TOP_ATOMS = 4
MAX_REP = 4
FREE_DEPTH = 2.0         # A: relax metal atoms within this depth of the top surface
FMAX = 0.05
MAX_STEPS = 300
MAX_SLAB_ATOMS = 400
EV_PER_A2_TO_J_M2 = 16.0218

HEADER = ["bulk_file", "formula", "tier", "in_mamun", "sg_stable", "miller",
          "term_index", "n_terminations", "supercell", "slab_atoms",
          "top_layer_atoms", "coverage_ML", "surface_area", "thickness_A",
          "surface_composition", "stoichiometric", "E_slab", "E_bulk_per_atom",
          "gamma_eV_A2", "gamma_J_m2", "converged"]

_calc = None


def get_calc():
    global _calc
    if _calc is None:
        import torch
        from mattersim.forcefield import MatterSimCalculator
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _calc = MatterSimCalculator(device=dev)
        print("MatterSim ready on {}".format(dev), flush=True)
    return _calc


def bulk_energy_per_atom(struct):
    atoms = AseAtomsAdaptor().get_atoms(struct)
    atoms.calc = get_calc()
    return atoms.get_potential_energy() / len(atoms)


def relax_clean(slab):
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(slab)
    fix_bottom_by_depth(atoms, FREE_DEPTH)
    atoms.calc = get_calc()
    conv = BFGS(atoms, logfile=os.devnull).run(fmax=FMAX, steps=MAX_STEPS)
    return atoms.get_potential_energy(), bool(conv)


def is_stoichiometric(slab, bulk):
    a = slab.composition.reduced_composition.as_dict()
    b = bulk.composition.reduced_composition.as_dict()
    if set(a) != set(b):
        return 0
    return int(all(abs(a[k] - b[k]) < 1e-6 for k in a))


def surface_composition(slab):
    top = layer_groups(slab)[0][-1]
    syms = sorted(slab[i].specie.symbol for i in top)
    return "".join("{}{}".format(s, syms.count(s)) for s in sorted(set(syms)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-per-bulk", type=int, default=3,
                    help="written to the summary; stage2 applies the cut")
    args = ap.parse_args()

    done = set()
    if CHKPT.exists():
        done = set(open(CHKPT).read().split())

    new = not OUT_CSV.exists()
    fh = open(OUT_CSV, "a", newline="", buffering=1)
    w = csv.DictWriter(fh, fieldnames=HEADER, extrasaction="ignore")
    if new:
        w.writeheader()

    bulks = [r for r in csv.DictReader(open(MANIFEST)) if r["status"] == "ok"]
    if args.limit:
        bulks = bulks[: args.limit]
    print("screening {} bulks\n".format(len(bulks)), flush=True)

    t0 = time.time()
    n_done = n_skip = 0

    for bi, row in enumerate(bulks, 1):
        stem = Path(row["std_file"]).stem
        try:
            bulk = Structure.from_file(str(STD_DIR / row["std_file"]))
        except Exception as e:
            print("  load fail {}: {}".format(stem, e), flush=True)
            continue

        facets = ([tuple(int(c) for c in row["mamun_facet"])]
                  if row.get("tier") == "A" and row.get("mamun_facet")
                  else MILLER_INDICES)

        e_bulk = None
        for miller in facets:
            ms = "".join(map(str, miller))
            try:
                slabs = SlabGenerator(bulk, miller, MIN_SLAB_A, VACUUM_A,
                                      center_slab=True, in_unit_planes=False,
                                      primitive=True).get_slabs(symmetrize=False)
            except Exception:
                continue
            if not slabs:
                continue

            for ti, raw in enumerate(slabs):
                sid = "{}__{}__t{}".format(stem, ms, ti)
                if sid in done:
                    n_skip += 1
                    continue
                slab, rep, ntop = supercell_to_coverage(raw, MIN_TOP_ATOMS, MAX_REP)
                if slab is None or len(slab) > MAX_SLAB_ATOMS:
                    continue

                if e_bulk is None:
                    try:
                        e_bulk = bulk_energy_per_atom(bulk)
                    except Exception as e:
                        print("  bulk energy fail {}: {}".format(stem, e), flush=True)
                        break

                try:
                    e_slab, conv = relax_clean(slab)
                except Exception as e:
                    print("  relax fail {}: {}".format(sid, str(e)[:50]), flush=True)
                    continue

                area = slab.surface_area
                gamma = (e_slab - len(slab) * e_bulk) / (2.0 * area)
                w.writerow({
                    "bulk_file": row["bulk_file"], "formula": row["formula"],
                    "tier": row.get("tier", ""), "in_mamun": row.get("in_mamun", ""),
                    "sg_stable": row.get("sg_stable", ""), "miller": ms,
                    "term_index": ti, "n_terminations": len(slabs),
                    "supercell": rep, "slab_atoms": len(slab),
                    "top_layer_atoms": ntop, "coverage_ML": round(1.0 / ntop, 4),
                    "surface_area": round(area, 3),
                    "thickness_A": round(slab_thickness(slab), 2),
                    "surface_composition": surface_composition(slab),
                    "stoichiometric": is_stoichiometric(slab, bulk),
                    "E_slab": round(e_slab, 6),
                    "E_bulk_per_atom": round(e_bulk, 6),
                    "gamma_eV_A2": round(gamma, 6),
                    "gamma_J_m2": round(gamma * EV_PER_A2_TO_J_M2, 4),
                    "converged": "OK" if conv else "NOT_CONVERGED"})
                fh.flush()
                with open(CHKPT, "a") as f:
                    f.write(sid + "\n")
                n_done += 1

        if bi % 25 == 0:
            el = time.time() - t0
            print("  [{}/{}] {} surfaces, {:.0f}/hr".format(
                bi, len(bulks), n_done, n_done / el * 3600 if el else 0), flush=True)

    fh.close()
    print("\ndone: {} surfaces relaxed, {} skipped, {:.2f} hr".format(
        n_done, n_skip, (time.time() - t0) / 3600))

    rows = list(csv.DictReader(open(OUT_CSV)))
    by_bulk = defaultdict(list)
    for r in rows:
        try:
            by_bulk[r["bulk_file"]].append(float(r["gamma_J_m2"]))
        except ValueError:
            pass
    if by_bulk:
        spread = [max(v) - min(v) for v in by_bulk.values() if len(v) > 1]
        print("\nbulks screened      : {}".format(len(by_bulk)))
        print("surfaces per bulk   : mean {:.1f}, max {}".format(
            np.mean([len(v) for v in by_bulk.values()]),
            max(len(v) for v in by_bulk.values())))
        if spread:
            print("gamma spread per bulk (J/m2): median {:.2f}, max {:.2f}".format(
                float(np.median(spread)), max(spread)))
        print("keeping {} lowest-gamma per bulk -> {} surfaces "
              "(from {})".format(args.keep_per_bulk,
                                 sum(min(args.keep_per_bulk, len(v))
                                     for v in by_bulk.values()), len(rows)))
        ns = sum(1 for r in rows if r["stoichiometric"] == "0")
        print("non-stoichiometric terminations: {}/{} "
              "(gamma approximate for these)".format(ns, len(rows)))


if __name__ == "__main__":
    main()
