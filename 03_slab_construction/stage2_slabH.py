#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_slabH.py -- slab+H generation following the Mamun convention.

Verified against Mamun et al., Sci Data 6, 76 (2019):
  A1(111)  -> 12 atoms, 3 layers, 0.25 ML,  4 sites  (paper reports 4)
  L12(111) -> 12 atoms, 3 layers, 0.25 ML,  9 sites  (paper reports 9)
  L10(111) -> 12 atoms, 3 layers, 0.25 ML, 10 sites  (paper reports 10)
The 12-atom count matches their published Mo3Ru9 entry (3 layers x 4 atoms).

Construction:
  * slabs are cut from the standardized cell, so `miller` identifies a real surface
  * only the KEEP_PER_BULK lowest-surface-energy (facet, termination) pairs from
    the surface screen are used (on a bimetallic, different terminations expose
    different elements)
  * in-plane supercells with >= 4 top-layer atoms give 0.25 ML, matching Mamun
  * metal atoms within FREE_DEPTH of the top surface and the H atom are relaxed,
    the rest of the slab is fixed
  * in_unit_planes=False, so slab thickness and vacuum are in Angstrom
  * MatterSim relaxation with BFGS (fmax = FMAX, at most MAX_STEPS steps)

Usage:
  python 03_slab_construction/stage2_slabH.py [--dry-run] [--limit N]
"""
import argparse, csv, os, sys, time, warnings
from pathlib import Path

import numpy as np
from ase.optimize import BFGS
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

sys.path.insert(0, str(Path(__file__).parent))
from slab_common import (layer_groups, n_top_atoms, slab_thickness,
                         supercell_to_coverage, fix_bottom_layers,
                         fix_bottom_by_depth)

warnings.filterwarnings("ignore")

STD_DIR = Path("data/structures/bulk_std")
MANIFEST = Path("data/bulk_manifest_binary.csv")
OUT_DIR = Path("data/relaxed_slabH_v3")
META_FILE = Path("data/slabH_metadata_v3.csv")
CHKPT_FILE = Path("data/slabH_checkpoint_v3.csv")
LOG_FILE = Path("data/slabH_generation_v3.log")
FAILED_FILE = Path("data/slabH_failed_v3.log")

MILLER_INDICES = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
MIN_SLAB_A = 8.0        # A: minimum slab thickness, leaves a fixed interior below FREE_DEPTH
VACUUM_A = 15.0
MIN_TOP_ATOMS = 4       # -> 0.25 ML with a single H, as in Mamun
MAX_REP = 4
H_HEIGHT = 1.5
FREE_DEPTH = 2.0        # A: relax metal atoms within this depth of the top surface
SITE_TYPES = ["ontop", "bridge", "hollow"]
SYMM_REDUCE = 0.01      # coordinate-match atol; internal symprec is 0.1 (fixed)
NEAR_REDUCE = 0.01      # does not merge hcp and fcc hollows
FMAX = 0.05
MAX_STEPS = 500
MAX_SLAB_ATOMS = 400    # skip pathological cells
SCREEN_FILE = Path("data/surface_screen.csv")   # stage1c surface-energy screen
KEEP_PER_BULK = 3       # keep only the 3 lowest-gamma (facet, termination) per bulk


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _device()

META_HEADER = ["bulk_file", "formula", "sg_symbol", "prototype",
               "tier", "in_mamun", "sg_stable", "miller",
               "n_terminations", "term_index", "supercell", "slab_atoms",
               "n_layers", "top_layer_atoms", "coverage_ML", "surface_area",
               "thickness_A", "surface_composition", "gamma_J_m2",
               "site_type", "site_index",
               "n_atoms", "n_fixed", "relaxed_file", "converged"]

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
_log = open(LOG_FILE, "a", buffering=1)


def log(m):
    line = "[{}] {}".format(time.strftime("%H:%M:%S"), m)
    print(line, flush=True)
    _log.write(line + "\n")


def failed(*a):
    with open(FAILED_FILE, "a") as f:
        f.write("\t".join(str(x) for x in a) + "\n")


_calc = None


def get_calc():
    global _calc
    if _calc is None:
        from mattersim.forcefield import MatterSimCalculator
        _calc = MatterSimCalculator(device=DEVICE)
        log("MatterSim ready on {}".format(DEVICE))
    return _calc


def relax(struct):
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(struct)
    n_fixed, _ = fix_bottom_by_depth(atoms, FREE_DEPTH)
    atoms.calc = get_calc()
    conv = BFGS(atoms, logfile=os.devnull).run(fmax=FMAX, steps=MAX_STEPS)
    return adaptor.get_structure(atoms), bool(conv), n_fixed


def surface_composition(slab):
    top = layer_groups(slab)[0][-1]
    syms = sorted(slab[i].specie.symbol for i in top)
    return "".join("{}{}".format(s, syms.count(s)) for s in sorted(set(syms)))


def load_screen(path, keep_per_bulk):
    """Read the stage1c surface screen and return the set of kept
    (bulk_file, miller, term_index) triples -- the keep_per_bulk lowest-gamma
    surfaces per bulk -- plus a {triple: gamma_J_m2} map for the metadata."""
    by_bulk = {}
    for r in csv.DictReader(open(path)):
        by_bulk.setdefault(r["bulk_file"], []).append(
            (float(r["gamma_J_m2"]), r["miller"], int(r["term_index"])))
    kept, gamma = set(), {}
    for bf, lst in by_bulk.items():
        for g, ms, ti in sorted(lst)[:keep_per_bulk]:
            kept.add((bf, ms, ti))
            gamma[(bf, ms, ti)] = g
    return kept, gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set(open(CHKPT_FILE).read().split()) if CHKPT_FILE.exists() else set()

    new = not META_FILE.exists()
    mfh = open(META_FILE, "a", newline="", buffering=1)
    mw = csv.DictWriter(mfh, fieldnames=META_HEADER, extrasaction="ignore")
    if new:
        mw.writeheader()

    bulks = [r for r in csv.DictReader(open(MANIFEST)) if r["status"] == "ok"]
    if args.limit:
        bulks = bulks[:args.limit]

    if not SCREEN_FILE.exists():
        log("ERROR: {} not found -- run stage1c_surface_screen.py first".format(SCREEN_FILE))
        sys.exit(1)
    kept_pairs, gamma_map = load_screen(SCREEN_FILE, KEEP_PER_BULK)
    log("surface screen: {} (bulk,facet,term) kept (<= {}/bulk) from {}".format(
        len(kept_pairs), KEEP_PER_BULK, SCREEN_FILE))

    log("=" * 66)
    log("SLAB+H (Mamun convention) | {} bulks | {} | dry-run={}".format(
        len(bulks), DEVICE, args.dry_run))
    log("min_slab={}A vacuum={}A target>={} top atoms ({:.2f} ML) BFGS fmax={}".format(
        MIN_SLAB_A, VACUUM_A, MIN_TOP_ATOMS, 1.0 / MIN_TOP_ATOMS, FMAX))
    log("resuming: {} already done".format(len(done)))
    log("=" * 66)

    t0 = time.time()
    n_ok = n_skip = n_fail = 0

    for bi, row in enumerate(bulks, 1):
        stem = Path(row["std_file"]).stem
        try:
            bulk = Structure.from_file(str(STD_DIR / row["std_file"]))
        except Exception as e:
            failed(stem, "-", "load", str(e)[:60])
            continue

        log("[{}/{}] {} {} ({} atoms)".format(
            bi, len(bulks), row["formula"], row["sg_symbol"], row["natoms_std"]))

        facets = ([tuple(int(c) for c in row["mamun_facet"])]
                  if row.get("tier") == "A" and row.get("mamun_facet")
                  else MILLER_INDICES)
        for miller in facets:
            ms = "".join(map(str, miller))
            try:
                slabs = SlabGenerator(
                    bulk, miller, MIN_SLAB_A, VACUUM_A, center_slab=True,
                    in_unit_planes=False, primitive=True).get_slabs(symmetrize=False)
            except Exception as e:
                failed(stem, ms, "slabgen", str(e)[:60])
                continue
            if not slabs:
                failed(stem, ms, "no_slabs", "")
                continue

            for ti, raw in enumerate(slabs):
                if (row["bulk_file"], ms, ti) not in kept_pairs:
                    continue  # not among the KEEP_PER_BULK lowest-gamma surfaces
                slab, rep, ntop = supercell_to_coverage(raw, MIN_TOP_ATOMS, MAX_REP)
                if slab is None:
                    failed(stem, ms, "term{}_supercell".format(ti),
                           "top={}".format(n_top_atoms(raw)))
                    continue
                if len(slab) > MAX_SLAB_ATOMS:
                    failed(stem, ms, "term{}_too_big".format(ti), len(slab))
                    continue

                try:
                    asf = AdsorbateSiteFinder(slab)
                    sd = asf.find_adsorption_sites(
                        distance=H_HEIGHT, symm_reduce=SYMM_REDUCE,
                        near_reduce=NEAR_REDUCE, positions=SITE_TYPES,
                        no_obtuse_hollow=True)
                    sites = [(st, np.array(c)) for st in SITE_TYPES
                             for c in sd.get(st, [])]
                except Exception as e:
                    failed(stem, ms, "term{}_sites".format(ti), str(e)[:60])
                    continue
                if not sites:
                    failed(stem, ms, "term{}_no_sites".format(ti), "")
                    continue

                nlay = len(layer_groups(slab)[0])
                cov = 1.0 / ntop
                surfcomp = surface_composition(slab)
                log("  ({}) term{}/{} {} | {} at | {} lay | top={} "
                    "{:.3f} ML | {} | {} sites".format(
                        ms, ti, len(slabs), rep, len(slab), nlay, ntop,
                        cov, surfcomp, len(sites)))

                base = dict(
                    bulk_file=row["bulk_file"], formula=row["formula"],
                    sg_symbol=row["sg_symbol"], prototype=row["prototype"],
                    tier=row.get("tier", ""), in_mamun=row.get("in_mamun", ""),
                    sg_stable=row.get("sg_stable", ""),
                    miller=ms, n_terminations=len(slabs), term_index=ti,
                    supercell=rep, slab_atoms=len(slab), n_layers=nlay,
                    top_layer_atoms=ntop, coverage_ML=round(cov, 4),
                    surface_area=round(slab.surface_area, 3),
                    thickness_A=round(slab_thickness(slab), 2),
                    surface_composition=surfcomp,
                    gamma_J_m2=gamma_map.get((row["bulk_file"], ms, ti), ""))

                for si, (stype, coord) in enumerate(sites):
                    sid = "{}__{}__t{}__{}__{}".format(stem, ms, ti, stype, si)
                    if sid in done:
                        n_skip += 1
                        continue
                    try:
                        sh = slab.copy()
                        sh.append("H", coord, coords_are_cartesian=True)
                    except Exception as e:
                        failed(stem, ms, "place_" + sid, str(e)[:60])
                        n_fail += 1
                        continue

                    if args.dry_run:
                        n_ok += 1
                        continue

                    try:
                        rel, conv, nfix = relax(sh)
                    except Exception as e:
                        failed(stem, ms, "relax_" + sid,
                               "{}: {}".format(type(e).__name__, str(e)[:60]))
                        n_fail += 1
                        continue

                    name = sid + ".cif"
                    try:
                        CifWriter(rel).write_file(str(OUT_DIR / name))
                    except Exception as e:
                        failed(stem, ms, "write_" + sid, str(e)[:60])
                        n_fail += 1
                        continue

                    mw.writerow(dict(base, site_type=stype, site_index=si,
                                     n_atoms=len(rel), n_fixed=nfix,
                                     relaxed_file=name,
                                     converged="OK" if conv else "NOT_CONVERGED"))
                    mfh.flush()
                    with open(CHKPT_FILE, "a") as f:
                        f.write(sid + "\n")
                    n_ok += 1

        if bi % 10 == 0:
            el = (time.time() - t0) / 3600
            log("  ... {} done, {:.0f}/hr".format(n_ok, n_ok / el if el else 0))

    el = (time.time() - t0) / 3600
    log("=" * 66)
    log("FINISHED ok={} skip={} fail={} time={:.2f}hr".format(
        n_ok, n_skip, n_fail, el))
    mfh.close()


if __name__ == "__main__":
    main()
