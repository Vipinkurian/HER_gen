#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, warnings, tempfile, time
warnings.filterwarnings("ignore")
import torch, numpy as np, pandas as pd, dgl
from pathlib import Path
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.graphs import Graph
from jarvis.core.atoms import Atoms as JAtoms

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = Path("alignn_her_out/best_model.pt")
RESULTS_PATH= Path("alignn_her_out/results.json")
CIF_DIR     = Path("data/relaxed_slabH_v3")
META_FILE   = Path("data/slabH_metadata_v3.csv")
OUT_DIR     = Path("predictions_alignn_v3")
BATCH_SIZE  = 64

import json
with open(RESULTS_PATH) as f:
    saved = json.load(f)
CUTOFF       = saved["cutoff"]
MAX_NEIGHBORS= saved["max_neighbors"]

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device={DEVICE}  cutoff={CUTOFF}  max_neighbors={MAX_NEIGHBORS}")

cfg = ALIGNNConfig(name="alignn",
    alignn_layers=saved["alignn_layers"], gcn_layers=saved["gcn_layers"],
    atom_input_features=92, edge_input_features=80,
    triplet_input_features=40, embedding_features=64,
    hidden_features=saved["hidden_features"],
    output_features=1, classification=False)
model = ALIGNN(cfg).to(DEVICE)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
model.eval()
print("ALIGNN model loaded")

def cif_to_graph(cif_path):
    try:
        parser  = CifParser(str(cif_path))
        structs = parser.get_structures(primitive=False)
        if not structs:
            return None, None
        fd, tmp = tempfile.mkstemp(suffix=".vasp")
        os.close(fd)
        try:
            Poscar(structs[0]).write_file(tmp)
            atoms = JAtoms.from_poscar(tmp)
            g, lg = Graph.atom_dgl_multigraph(
                atoms, cutoff=CUTOFF, max_neighbors=MAX_NEIGHBORS)
            return g, lg
        finally:
            os.unlink(tmp)
    except Exception:
        return None, None

@torch.no_grad()
def predict_batch(graphs, line_graphs):
    bg  = dgl.batch(graphs).to(DEVICE)
    blg = dgl.batch(line_graphs).to(DEVICE)
    preds = model((bg, blg, None))
    if isinstance(preds, dict):
        preds = preds["out"]
    return preds.squeeze(-1).cpu().numpy()

meta = pd.read_csv(META_FILE)
print(f"Structures to predict: {len(meta)}")

results, bg_list, blg_list, bmeta = [], [], [], []
n_ok = n_fail = 0
t0 = time.time()

def flush():
    global n_ok, n_fail
    if not bg_list:
        return
    try:
        preds = predict_batch(bg_list, blg_list)
        for i, rec in enumerate(bmeta):
            rec["predicted_dG"] = float(preds[i])
            results.append(rec)
        n_ok += len(bg_list)
    except Exception as e:
        print(f"  Batch error: {e}")
        n_fail += len(bg_list)
    bg_list.clear(); blg_list.clear(); bmeta.clear()

for i, row in meta.iterrows():
    g, lg = cif_to_graph(CIF_DIR / row["relaxed_file"])
    if g is None:
        n_fail += 1
        continue
    bg_list.append(g); blg_list.append(lg)
    bmeta.append(row.to_dict())
    if len(bg_list) >= BATCH_SIZE:
        flush()
    if (i + 1) % 500 == 0:
        rate = (n_ok + n_fail) / (time.time() - t0) * 3600
        print(f"  {i+1}/{len(meta)}  ok={n_ok}  fail={n_fail}  rate={rate:.0f}/hr")

flush()

df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "predictions_all.csv", index=False)
print(f"\nSaved {len(df)} predictions  ok={n_ok}  fail={n_fail}")

df["abs_dG"] = df["predicted_dG"].abs()
best = df.loc[df.groupby("bulk_file")["abs_dG"].idxmin()].sort_values("abs_dG").reset_index(drop=True)
best.to_csv(OUT_DIR / "best_per_bulk.csv", index=False)
print(f"Best-site per bulk: {len(best)}")
for thr in [0.1, 0.2, 0.3]:
    n = (best["abs_dG"] <= thr).sum()
    print(f"  |dG| <= {thr} eV : {n} ({100*n/len(best):.1f}%)")

print(f"\nTop 20 candidates:")
print(f"  {'Rank':<5} {'Formula':<20} {'Miller':<8} {'Site':<10} {'dG (eV)'}")
print("  " + "-"*55)
for rank, (_, row) in enumerate(best.head(20).iterrows(), 1):
    print(f"  {rank:<5} {str(row['formula'])[:20]:<20} {row['miller']:<8} "
          f"{row['site_type']+'_'+str(row['site_index']):<10} {row['predicted_dG']:+.4f}")
