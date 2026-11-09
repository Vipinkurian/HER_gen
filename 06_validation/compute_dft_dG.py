#!/usr/bin/env python3
"""DFT hydrogen adsorption free energies from the Quantum ESPRESSO outputs in data/dft/.

    dG_H* = E(slab+H) - E(slab) - 1/2 E(H2) + 0.24 eV

E is the final total energy of each converged relaxation (last '!' line of the pw.x output).
The 0.24 eV term is the zero-point and entropy correction for adsorbed H at 298 K.
Writes data/dft/dft_energies_dG.csv.
"""
import csv
import re
from pathlib import Path

RY_TO_EV = 13.605693122994
CHE = 0.24
ROOT = Path(__file__).resolve().parents[1]
DFT = ROOT / "data" / "dft"

# Site group of each candidate (see the paper, Section 4.6).
GROUPS = {
    "Pd5Au": "dE_approx_0_sites",
    "Pd13Ru3": "dE_approx_0_sites",
    "TiPd5": "dE_approx_0_sites",
    "Cu3Pd13": "dE_approx_0_sites",
    "Pt5Rh": "dG_best_sites",
    "Cu5Pt3": "dG_best_sites",
    "Cu2Pt3": "dG_best_sites",
    "Pt7Pb": "dG_best_sites",
}


def final_energy(out_file):
    """Return (final total energy in eV, number of atoms) of a converged pw.x relaxation."""
    text = out_file.read_text()
    energies = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", text, re.M)
    if not energies:
        raise RuntimeError(f"no total energy found in {out_file}")
    if "bfgs converged" not in text:
        raise RuntimeError(f"relaxation not converged in {out_file}")
    nat = int(re.search(r"number of atoms/cell\s+=\s+(\d+)", text).group(1))
    return float(energies[-1]) * RY_TO_EV, nat


def main():
    e_h2, _ = final_energy(next((DFT / "H2").glob("*.out")))
    rows = []
    for name, group in GROUPS.items():
        e_slab_h, nat = final_energy(next((DFT / name / "slab_H").glob("*.out")))
        e_slab, _ = final_energy(next((DFT / name / "clean_slab").glob("*.out")))
        de = e_slab_h - e_slab - 0.5 * e_h2
        rows.append({
            "compound": name,
            "site_group": group,
            "n_atoms_slab_H": nat,
            "E_slab_H_eV": f"{e_slab_h:.5f}",
            "E_slab_eV": f"{e_slab:.5f}",
            "E_H2_eV": f"{e_h2:.5f}",
            "dE_H_eV": f"{de:.3f}",
            "dG_H_eV": f"{de + CHE:.3f}",
        })
        print(f"{name:8s} dG_H* = {de + CHE:+.3f} eV")
    out = DFT / "dft_energies_dG.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
