#!/usr/bin/env python3
"""Derive data/bulk_manifest_binary.csv -- the 730 novel bimetallic target pool.

Filter (applied to data/bulk_manifest.csv rows with status == "ok"):
  * binary             : exactly 2 elements
  * not in Mamun       : in_mamun != "1" (no structural match to A1/L1_2/L1_0)
  * both metallic      : both elements pass pymatgen Element.is_metal
  * element in-domain  : both elements are in the Mamun 37-metal set
  * novel stoichiometry: reduced ratio is NOT 1:1 (AB) or 3:1 (A3B) -- the only
                         binary ratios Mamun studied

The master manifest is left untouched; the Mamun-overlap / pure / ternary rows
stay there as validation references. This script only writes the filtered copy,
preserving every column so stage1c/stage2 read it unchanged.
"""
import csv
import re
from collections import Counter
from pathlib import Path

from pymatgen.core import Composition

MASTER = Path("data/bulk_manifest.csv")
OUT = Path("data/bulk_manifest_binary.csv")
OVERLAP = Path("scripts/mamun_overlap.py")


def mamun_metals():
    blk = re.search(r"MAMUN_METALS\s*=\s*[\{\[\(](.*?)[\}\]\)]",
                    OVERLAP.read_text(), re.S).group(1)
    return set(re.findall(r"""["']([A-Z][a-z]?)["']""", blk))


def reduced_ratio(formula):
    amts = Composition(formula).reduced_composition.get_el_amt_dict()
    return tuple(sorted(int(round(x)) for x in amts.values()))


def keep(row, mamun):
    if row["status"] != "ok":
        return False
    try:
        els = Composition(row["formula"]).elements
    except Exception:
        return False
    if len(els) != 2:
        return False
    if row["in_mamun"] in ("1",):
        return False
    if not all(e.is_metal for e in els):
        return False
    if not all(e.symbol in mamun for e in els):
        return False
    if reduced_ratio(row["formula"]) in [(1, 1), (1, 3)]:
        return False
    return True


def main():
    mamun = mamun_metals()
    reader = csv.DictReader(open(MASTER))
    rows = list(reader)
    kept = [r for r in rows if keep(r, mamun)]

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=reader.fieldnames)
        w.writeheader()
        w.writerows(kept)

    print("master rows              : {}".format(len(rows)))
    print("kept (novel bimetallics) : {}  -> {}".format(len(kept), OUT))
    print("  tier                   : {}".format(dict(Counter(r["tier"] for r in kept))))
    print("  all binary             : {}".format(
        all(len(Composition(r["formula"]).elements) == 2 for r in kept)))
    print("  any in_mamun           : {}".format(any(r["in_mamun"] in ("1",) for r in kept)))
    print("  any Mamun 1:1 / 3:1    : {}".format(
        any(reduced_ratio(r["formula"]) in [(1, 1), (1, 3)] for r in kept)))
    print("  Mamun metal set size   : {}".format(len(mamun)))


if __name__ == "__main__":
    main()
