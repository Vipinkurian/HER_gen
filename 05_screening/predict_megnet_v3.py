#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, dgl, matgl
from pathlib import Path
from pymatgen.io.cif import CifParser
from matgl.ext.pymatgen import Structure2Graph
from matgl.models._megnet import MEGNet
from matminer.featurizers import composition as cf
import torch.nn as nn

MODEL_DIR   = Path("megnet_her_v2_final/best_model")
CIF_DIR     = Path("data/relaxed_slabH_v3")
META_FILE   = Path("data/slabH_metadata_v3.csv")
OUT_DIR     = Path("predictions_megnet_v3")
BATCH       = 64
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(MODEL_DIR / "config.json") as f: cfg = json.load(f)
with open(MODEL_DIR / "element_list.json") as f: ELEM_LIST = json.load(f)
CUTOFF = cfg["cutoff"]
NORM_MEAN = np.load(MODEL_DIR / "state_normalizer_mean.npy")
NORM_STD  = np.load(MODEL_DIR / "state_normalizer_std.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  cutoff: {CUTOFF}")

# Magpie state features (identical to training)
_magpie = cf.ElementProperty.from_preset("magpie")
_labels = _magpie.feature_labels()
_IDX1   = next(i for i,l in enumerate(_labels) if l == cfg["state_feat_1"])
_IDX2   = next(i for i,l in enumerate(_labels) if l == cfg["state_feat_2"])

def state_attr(composition):
    try:
        f = _magpie.featurize(composition)
        v = [float(f[_IDX1]), float(f[_IDX2])]
        if any(np.isnan(v)): return [0.0, 0.0]
        return v
    except: return [0.0, 0.0]

def norm_state(raw):
    return ((np.array(raw, np.float32) - NORM_MEAN) / NORM_STD).astype(np.float32)

# Load model: pretrained arch + fine-tuned weights
print("Loading MEGNet ...")
from matgl.layers import BondExpansion
megnet = MEGNet(
    dim_node_embedding=16, dim_edge_embedding=100, dim_state_embedding=2,
    nblocks=3, hidden_layer_sizes_input=(64, 32),
    hidden_layer_sizes_conv=(64, 64, 32), hidden_layer_sizes_output=(32, 16),
    nlayers_set2set=1, niters_set2set=2, activation_type="softplus2",
    is_classification=False, include_state=True, dropout=0.0,
    bond_expansion=BondExpansion(rbf_type="Gaussian", initial=0.0,
                                 final=CUTOFF + 1.0, num_centers=100, width=0.5),
    cutoff=CUTOFF)

# Re-insert dropout (same as training)
old = list(megnet.output_proj.layers)
new = []
for layer in old:
    new.append(layer)
    if "softplus" in type(layer).__name__.lower():
        new.append(nn.Dropout(p=cfg["dropout"]))
megnet.output_proj.layers = nn.ModuleList(new)

state_dict = torch.load(MODEL_DIR / "model_state.pt", map_location=device)
megnet.load_state_dict(state_dict)
megnet.to(device); megnet.eval()
print(f"MEGNet loaded  params={sum(p.numel() for p in megnet.parameters())}")

# Graph converter
converter = Structure2Graph(element_types=ELEM_LIST, cutoff=CUTOFF)

def add_cart(g, lat):
    L = lat[0]
    g.ndata["pos"]          = torch.mm(g.ndata["frac_coords"], L)
    g.edata["pbc_offshift"] = torch.mm(g.edata["pbc_offset"].float(), L)
    return g

@torch.no_grad()
def predict_batch(graphs, sa_list):
    bg = dgl.batch(graphs).to(device)
    sa = torch.tensor(sa_list, dtype=torch.float32).to(device)
    return megnet(bg, sa).squeeze(-1).cpu().numpy()

# Run predictions
meta = pd.read_csv(META_FILE)
print(f"Structures: {len(meta)}")

results, bg_list, sa_list, bm = [], [], [], []
n_ok = n_fail = 0
t0 = time.time()

def flush():
    global n_ok, n_fail
    if not bg_list: return
    try:
        preds = predict_batch(bg_list, sa_list)
        for i, rec in enumerate(bm):
            rec["predicted_dG"] = float(preds[i])
            results.append(rec)
        n_ok += len(bg_list)
    except Exception as e:
        print(f"  batch error: {e}")
        n_fail += len(bg_list)
    bg_list.clear(); sa_list.clear(); bm.clear()

for i, row in meta.iterrows():
    try:
        parser  = CifParser(str(CIF_DIR / row["relaxed_file"]))
        structs = parser.get_structures(primitive=False)
        if not structs: raise ValueError("empty")
        struct = structs[0]
        g, lat, _ = converter.get_graph(struct)
        g = add_cart(g, lat)
        bg_list.append(g)
        sa_list.append(norm_state(state_attr(struct.composition)))
        bm.append(row.to_dict())
        if len(bg_list) >= BATCH: flush()
    except Exception: n_fail += 1
    if (i+1) % 500 == 0:
        rate = (n_ok+n_fail)/(time.time()-t0)*3600
        print(f"  {i+1}/{len(meta)}  ok={n_ok}  fail={n_fail}  {rate:.0f}/hr")

flush()
print(f"\nDone  ok={n_ok}  fail={n_fail}")

df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "predictions_all.csv", index=False)
df["abs_dG"] = df["predicted_dG"].abs()
best = df.loc[df.groupby("bulk_file")["abs_dG"].idxmin()].sort_values("abs_dG").reset_index(drop=True)
best.to_csv(OUT_DIR / "best_per_bulk.csv", index=False)
print(f"Best-site per bulk: {len(best)}")
for thr in [0.1, 0.2, 0.3]:
    n = (best["abs_dG"] <= thr).sum()
    print(f"  |dG| <= {thr} eV : {n} ({100*n/len(best):.1f}%)")

print(f"\nTop 20:")
print(f"  {'Rank':<5} {'Formula':<20} {'Miller':<8} {'Site':<12} {'dG (eV)'}")
print("  "+"-"*55)
for rank, (_, row) in enumerate(best.head(20).iterrows(), 1):
    print(f"  {rank:<5} {str(row['formula'])[:20]:<20} {row['miller']:<8} "
          f"{row['site_type']+'_'+str(row['site_index']):<12} {row['predicted_dG']:+.4f}")
