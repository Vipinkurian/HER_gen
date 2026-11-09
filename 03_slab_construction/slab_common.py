#!/usr/bin/env python3
"""Shared geometry helpers for the Mamun-convention slab pipeline."""
import numpy as np

def surface_heights(struct, skip=("H",)):
    """Heights of atoms projected on the TRUE surface normal (c may be tilted)."""
    m = struct.lattice.matrix
    n = np.cross(m[0], m[1]); n /= np.linalg.norm(n)
    idx = [i for i, s in enumerate(struct) if s.specie.symbol not in skip]
    return idx, np.array([np.dot(struct[i].coords, n) for i in idx]), n

def layer_groups(struct, skip=("H",)):
    """Cluster atoms into layers with a tolerance derived from the data itself.

    A fixed tolerance fails badly: fcc(111) layers sit ~2.1 A apart while a
    low-symmetry intermetallic can have real layers only ~0.4 A apart.
    """
    idx, h, _ = surface_heights(struct, skip)
    if len(idx) < 2:
        return [list(idx)], h
    order = np.argsort(h); hs = h[order]
    gaps = np.diff(hs); big = gaps[gaps > 0.05]
    tol = 0.5 * float(np.median(big)) if len(big) else 0.5
    tol = max(tol, 0.05)
    groups, cur = [], [order[0]]
    for k in range(1, len(order)):
        if hs[k] - hs[k-1] > tol: groups.append(cur); cur = [order[k]]
        else: cur.append(order[k])
    groups.append(cur)
    return [[idx[j] for j in g] for g in groups], h

def n_top_atoms(struct):
    return len(layer_groups(struct)[0][-1])

def slab_thickness(struct):
    _, h, _ = surface_heights(struct)
    return float(h.max() - h.min())

def supercell_to_coverage(slab, min_top=4, max_rep=4):
    """Smallest in-plane supercell giving >= min_top atoms in the top layer.
    min_top=4 reproduces Mamun's 0.25 ML (one adsorbate per 4 surface atoms)."""
    base = n_top_atoms(slab)
    if base == 0: return None, None, 0
    if base >= min_top: return slab, "1x1", base
    # prefer smallest area, then the most SQUARE cell: a 1x4 breaks the
    # hexagonal symmetry of an fcc(111) surface that a 2x2 preserves.
    cands = [(nx*ny, abs(nx-ny), nx, ny)
             for nx in range(1, max_rep + 1) for ny in range(1, max_rep + 1)
             if base * nx * ny >= min_top]
    if not cands: return None, None, base
    _, _, nx, ny = min(cands)
    sc = slab.copy(); sc.make_supercell([[nx,0,0],[0,ny,0],[0,0,1]])
    return sc, f"{nx}x{ny}", n_top_atoms(sc)

def fix_bottom_layers(atoms, n_free_layers=1):
    """Mamun: relax top layer + adsorbate, fix everything below.
    Uses the same adaptive layer clustering, on the ASE object."""
    from ase.constraints import FixAtoms
    from pymatgen.io.ase import AseAtomsAdaptor
    st = AseAtomsAdaptor().get_structure(atoms)
    groups, _ = layer_groups(st)
    free = set()
    for g in groups[-n_free_layers:]: free.update(g)
    free.update(i for i, a in enumerate(atoms) if a.symbol == "H")
    fixed = [i for i in range(len(atoms)) if i not in free]
    atoms.set_constraint(FixAtoms(indices=fixed))
    return len(fixed), len(groups)


def fix_bottom_by_depth(atoms, free_depth=2.0):
    """Relax every metal atom within free_depth (A) of the top surface, plus any
    adsorbate H; fix the rest. Depth transfers where layer COUNT does not: one
    fcc(111) layer spans ~2.1 A but one low-symmetry layer can span ~0.4 A, so
    freeing 'one layer' relaxes ~33% of an fcc slab yet only ~1-5% of an
    intermetallic. free_depth=2.0 frees exactly the top fcc layer (2nd layer
    ~2.08 A down stays fixed) -- matching Mamun -- and a ~2 A physical surface
    region on low-symmetry slabs. Uses the true surface-normal projection."""
    from ase.constraints import FixAtoms
    from pymatgen.io.ase import AseAtomsAdaptor
    st = AseAtomsAdaptor().get_structure(atoms)
    idx, h, _ = surface_heights(st)
    top = h.max()
    free = {i for i, hi in zip(idx, h) if hi >= top - free_depth}
    free.update(i for i, a in enumerate(atoms) if a.symbol == "H")
    fixed = [i for i in range(len(atoms)) if i not in free]
    atoms.set_constraint(FixAtoms(indices=fixed))
    return len(fixed), len(atoms) - len(fixed)
