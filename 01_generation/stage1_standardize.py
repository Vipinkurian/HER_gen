#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage1_standardize.py -- STAGE 1 of the slab+H pipeline.

Standardizes MatterGen/MatterSim-relaxed bulk CIFs and writes a manifest that
drives Stage 2.

WHY STANDARDIZE: Miller indices are defined against the input cell. Verified:
for fcc Cu, requesting (100) from the primitive cell returns the (111) surface
(area 5.643 A^2, d=2.084 A); from the conventional cell it returns the true
(100) (area 6.516 A^2, d=1.805 A). On a real Fe13Cu structure, cutting (100)
from the as-given P1 cell gives 7 terminations / 37.17 A^2, from the
standardized cell 4 terminations / 52.61 A^2. Without this step the `miller`
column does not identify a surface and facets are not comparable across
structures.

DOES NOT FILTER ANYTHING. Every structure is kept and labelled. MatterGen
produces stoichiometries outside the Mamun A1/L1_2/L1_0 grid by design; that is
the scientific contribution. Screening decisions happen later, from the tier
and prototype columns.

TIERS
  A  exact Mamun prototype (A1 / L1_2 / L1_0)  -> in-domain for the GNNs,
     close-packed facet known: (111) cubic, (101) tetragonal
  B  fcc-derived sublattice, different ordering -> facet enumerated
  C  non-fcc (bcc-derived, hcp, low-symmetry)   -> facet enumerated

Usage:
    python 01_generation/stage1_standardize.py
    python 01_generation/stage1_standardize.py --src data/structures/my_cifs --limit 50
"""
import argparse
import csv
import warnings
from collections import Counter
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

DEFAULT_SRC = Path("bulk_catalyst_optimal/grouped/stable_unique")
STD_DIR = Path("data/structures/bulk_std")
MANIFEST = Path("data/bulk_manifest.csv")

SYMPREC = 0.05                 # main tolerance for standardization
PROBE = (0.01, 0.05, 0.1)      # sweep; disagreement => genuinely low symmetry
SUBLATTICE_SYMPREC = 0.2       # loose: L1_0 is a tetragonally distorted fcc

HEADER = [
    "bulk_file", "std_file", "formula", "n_elements", "natoms_in", "natoms_std",
    "sg_number", "sg_symbol", "sg_stable", "sg_at_1e-3", "sublattice_sg",
    "prototype", "tier", "mamun_facet", "volume_per_atom", "status", "detail",
]


def classify_prototype(struct, sgnum):
    """Mamun prototype. Space group alone is NOT sufficient: #221 is L1_2 for
    A3B but B2 (CsCl) for AB; #225 is A1 for an element but B1 for AB."""
    amts = struct.composition.reduced_composition.get_el_amt_dict()
    vals = sorted(amts.values())
    n_el = len(amts)
    if sgnum == 225 and n_el == 1:
        return "A1"
    if sgnum == 221 and n_el == 2 and abs(vals[0] * 3 - vals[1]) < 1e-6:
        return "L12"
    if sgnum in (123, 129, 139) and n_el == 2 and abs(vals[0] - vals[1]) < 1e-6:
        return "L10"
    return "OTHER"


def is_fcc_derived(struct, symprec=SUBLATTICE_SYMPREC):
    """Strip chemical decoration; is the underlying metal sublattice fcc?"""
    dummy = struct.copy()
    dummy.replace_species({el: "Cu" for el in struct.composition.elements})
    try:
        sga = SpacegroupAnalyzer(dummy, symprec=symprec)
        return sga.get_space_group_number() == 225, sga.get_space_group_symbol()
    except Exception:
        return False, "err"


def assign_tier(conv, prototype):
    """Return (tier, sublattice_sg, mamun_facet)."""
    fcc, sub_sg = is_fcc_derived(conv)
    if prototype in ("A1", "L12", "L10"):
        a, b, c = conv.lattice.abc
        cubic = abs(a - b) < 1e-3 and abs(b - c) < 1e-3
        # Mamun: (111) for A1 and L1_2, (101) for L1_0. They note (101) is the
        # same physical surface as (111) cut from the cubic cell.
        return "A", sub_sg, "111" if cubic else "101"
    if fcc:
        # fcc-derived but a different ordering. The close-packed facet is not
        # reliably (101) here, so leave it to facet enumeration in Stage 2.
        return "B", sub_sg, ""
    return "C", sub_sg, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    STD_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    cifs = sorted(args.src.glob("*.cif"))
    if not cifs:
        raise SystemExit("no CIFs found in {}".format(args.src))
    if args.limit:
        cifs = cifs[: args.limit]
    print("found {} bulk CIFs in {}\n".format(len(cifs), args.src))

    rows = []
    for i, cif in enumerate(cifs, 1):
        try:
            s = Structure.from_file(str(cif))
        except Exception as e:
            rows.append(dict(bulk_file=cif.name, status="load_failed",
                             detail=str(e)[:80]))
            continue

        try:
            sga = SpacegroupAnalyzer(s, symprec=SYMPREC)
            sgnum = sga.get_space_group_number()
            sgsym = sga.get_space_group_symbol()
            conv = sga.get_conventional_standard_structure()
        except Exception as e:
            rows.append(dict(bulk_file=cif.name, status="symmetry_failed",
                             detail=str(e)[:80]))
            continue

        probes = []
        for t in PROBE:
            try:
                probes.append(SpacegroupAnalyzer(s, symprec=t).get_space_group_number())
            except Exception:
                probes.append(-1)
        sg_stable = int(len(set(probes)) == 1)

        try:
            tight = SpacegroupAnalyzer(s, symprec=1e-3).get_space_group_symbol()
        except Exception:
            tight = "?"

        proto = classify_prototype(conv, sgnum)
        tier, sub_sg, facet = assign_tier(conv, proto)

        out = STD_DIR / cif.name
        conv.to(filename=str(out), symprec=SYMPREC)

        rows.append({
            "bulk_file": cif.name, "std_file": out.name,
            "formula": conv.composition.reduced_formula,
            "n_elements": len(conv.composition.elements),
            "natoms_in": len(s), "natoms_std": len(conv),
            "sg_number": sgnum, "sg_symbol": sgsym,
            "sg_stable": sg_stable, "sg_at_1e-3": tight,
            "sublattice_sg": sub_sg, "prototype": proto,
            "tier": tier, "mamun_facet": facet,
            "volume_per_atom": round(conv.volume / len(conv), 4),
            "status": "ok", "detail": "",
        })

        if i % 50 == 0:
            print("  processed {}/{}".format(i, len(cifs)))

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("status") == "ok"]
    print("\n" + "=" * 62)
    print("wrote {}  ({} standardized CIFs in {})".format(
        MANIFEST, len(ok), STD_DIR))
    print("=" * 62)
    print("status     :", dict(Counter(r.get("status", "?") for r in rows)))
    if not ok:
        return

    tiers = Counter(r["tier"] for r in ok)
    print("\nTIER BREAKDOWN")
    labels = {"A": "Mamun prototype  (in-domain, close-packed facet known)",
              "B": "fcc-derived      (other ordering, enumerate facets)",
              "C": "non-fcc          (bcc/hcp/low-sym, enumerate facets)"}
    for t in ("A", "B", "C"):
        n = tiers.get(t, 0)
        print("  tier {}  {:5d}  ({:5.1f}%)  {}".format(
            t, n, 100.0 * n / len(ok), labels[t]))

    print("\nprototype  :", dict(Counter(r["prototype"] for r in ok)))
    print("sublattice :", dict(Counter(r["sublattice_sg"] for r in ok).most_common(6)))
    print("top space groups:",
          Counter(r["sg_symbol"] for r in ok).most_common(8))

    unstable = sum(1 for r in ok if not r["sg_stable"])
    print("\ntolerance-unstable space group: {}/{}".format(unstable, len(ok)))
    if unstable:
        print("  (these change space group between symprec 0.01 / 0.05 / 0.1;")
        print("   standardization may be imposing symmetry they do not have)")

    big = [r for r in ok if r["natoms_std"] > 60]
    if big:
        print("large standardized cells (>60 atoms): {} "
              "-- these will make expensive slabs".format(len(big)))

    print("\nNEXT: python 01_generation/mamun_overlap.py")


if __name__ == "__main__":
    main()
