#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mamun_overlap.py -- flag generated bulks that reproduce a Mamun training bulk.

Mamun's bulk set is fully enumerable: 37 metals in three fcc-derived
prototypes -- A1 (pure), L1_2 (A3B), L1_0 (AB). So the reference set can be
built analytically; no download needed. pymatgen's StructureMatcher uses
scale=True by default, which normalises volume, so the DFT-vs-MatterSim
lattice-constant difference does not cause false negatives.

Composition is a hard pre-filter: only pure elements, A3B and AB over the 37
metals can possibly match. Fe13Cu cannot, so it is never even compared.

Adds `in_mamun` and `mamun_match` columns to the manifest.
Set DELETE_MATCHES = True to also drop them from the standardized directory.

Usage: python 01_generation/mamun_overlap.py
"""
import csv, shutil, warnings
from fractions import Fraction
from pathlib import Path

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

warnings.filterwarnings("ignore")

MANIFEST = Path("data/bulk_manifest.csv")
STD_DIR = Path("data/structures/bulk_std")
REMOVED_DIR = Path("data/structures/bulk_mamun_overlap")
DELETE_MATCHES = False       # False = flag only (recommended); True = move out

# The 37 metals of the Mamun dataset (the same set as in
# data/training/MamunHighT2019_SlabH_only_cleaned.csv).
MAMUN_METALS = [
    "La","Zr","Ti","Nb","Cr","W","Tc","Ru","Rh","Co","Mn","Pd","Ag","Zn","Hg",
    "Ga","Tl","Sn","Y","Sc","Hf","Ta","V","Mo","Re","Os","Ir","Ni","Fe","Pt",
    "Au","Cu","Cd","Al","In","Pb","Bi",
]
assert len(MAMUN_METALS) == 37, len(MAMUN_METALS)
MSET = set(MAMUN_METALS)
A = 3.9   # arbitrary; StructureMatcher(scale=True) normalises volume


def ref_A1(el):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(A), [el], [[0, 0, 0]])


def ref_L12(a_el, b_el):   # A3B, B at the corner
    return Structure(Lattice.cubic(A), [b_el, a_el, a_el, a_el],
                     [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])


def ref_L10(a_el, b_el):   # AB, alternating (001) layers
    return Structure(Lattice.tetragonal(A, A * 0.97), [a_el, a_el, b_el, b_el],
                     [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])


def candidate_refs(struct):
    """Return [(label, reference_structure)] worth comparing, else []."""
    comp = struct.composition.reduced_composition
    amts = comp.get_el_amt_dict()
    els = sorted(amts)
    if not set(els) <= MSET:
        return []
    if len(els) == 1:
        return [("A1|{}".format(els[0]), ref_A1(els[0]))]
    if len(els) != 2:
        return []
    e1, e2 = els
    r = Fraction(amts[e1] / amts[e2]).limit_denominator(12)
    out = []
    if r == 1:
        out.append(("L10|{}{}".format(e1, e2), ref_L10(e1, e2)))
    elif r == 3:
        out.append(("L12|{}3{}".format(e1, e2), ref_L12(e1, e2)))
    elif r == Fraction(1, 3):
        out.append(("L12|{}3{}".format(e2, e1), ref_L12(e2, e1)))
    return out


def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    matcher = StructureMatcher()   # defaults: primitive_cell=True, scale=True
    n_match = n_checked = 0

    for r in rows:
        r["in_mamun"], r["mamun_match"] = 0, ""
        if r.get("status") != "ok":
            continue
        s = Structure.from_file(str(STD_DIR / r["std_file"]))
        refs = candidate_refs(s)
        if not refs:
            continue
        n_checked += 1
        for label, ref in refs:
            if matcher.fit(s, ref):
                r["in_mamun"], r["mamun_match"] = 1, label
                n_match += 1
                print("  MATCH {:28s} -> {}".format(r["std_file"], label))
                break

    hdr = list(rows[0].keys())
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)

    print("\ncomposition-eligible: {}   confirmed Mamun duplicates: {}"
          .format(n_checked, n_match))

    if DELETE_MATCHES and n_match:
        REMOVED_DIR.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if r["in_mamun"] == 1:
                src = STD_DIR / r["std_file"]
                if src.exists():
                    shutil.move(str(src), str(REMOVED_DIR / r["std_file"]))
                r["status"] = "mamun_duplicate"
        with open(MANIFEST, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader(); w.writerows(rows)
        print("moved {} duplicates to {}".format(n_match, REMOVED_DIR))
    else:
        print("flagged only (DELETE_MATCHES=False); stage2 will still build them")


if __name__ == "__main__":
    main()
