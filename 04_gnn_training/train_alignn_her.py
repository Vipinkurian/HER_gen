#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALIGNN training for HER reaction energy prediction, using the ALIGNN model API
directly instead of the train_alignn.py command-line tool.

Usage:
    python 04_gnn_training/train_alignn_her.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import time
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.graphs import Graph
from jarvis.core.atoms import Atoms

# CONFIGURATION

DATASET_DIR  = Path("data/alignn_dataset")
OUT_DIR      = Path("alignn_her_out")

RANDOM_SEED  = 42
N_TRAIN      = 6203
N_VAL        = 1330
N_TEST       = 1330
ENERGY_MIN   = -5.0
ENERGY_MAX   =  5.0

BATCH_SIZE   = 32
MAX_EPOCHS   = 300
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 30          # early stopping

# Graph construction
CUTOFF         = 8.0
MAX_NEIGHBORS  = 12

# Model
ALIGNN_LAYERS  = 4
GCN_LAYERS     = 4
HIDDEN_FEATURES = 256
EMBEDDING_FEATURES = 64

PLOT_DPI = 300

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REPRODUCIBILITY

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# DATASET

class SlabHGraphDataset(torch.utils.data.Dataset):
    """
    Reads POSCAR files and builds ALIGNN (bond graph + line graph) pairs.
    """

    def __init__(self, filenames, targets, dataset_dir,
                 cutoff=8.0, max_neighbors=12):
        self.filenames    = filenames
        self.targets      = targets
        self.dataset_dir  = Path(dataset_dir)
        self.cutoff       = cutoff
        self.max_neighbors = max_neighbors

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fpath  = self.dataset_dir / self.filenames[idx]
        target = self.targets[idx]

        atoms  = Atoms.from_poscar(str(fpath))
        g, lg = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )
        return g, lg, torch.tensor(target, dtype=torch.float32)


def collate_fn(batch):
    import dgl
    graphs, line_graphs, targets = zip(*batch)
    bg  = dgl.batch(graphs)
    blg = dgl.batch(line_graphs)
    t   = torch.stack(targets)
    return bg, blg, t


# METRICS

def compute_metrics(preds, targets):
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    r2   = float(r2_score(targets, preds))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# TRAINING HELPERS

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_preds, all_tgts = [], []

    for bg, blg, targets in loader:
        bg      = bg.to(DEVICE)
        blg     = blg.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        preds = model((bg, blg, None))
        if isinstance(preds, dict):
            preds = preds["out"]
        preds = preds.squeeze(-1)

        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        all_preds.extend(preds.detach().cpu().numpy())
        all_tgts.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    m = compute_metrics(np.array(all_preds), np.array(all_tgts))
    return avg_loss, m


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_tgts = [], []

    for bg, blg, targets in loader:
        bg      = bg.to(DEVICE)
        blg     = blg.to(DEVICE)
        targets = targets.to(DEVICE)

        preds = model((bg, blg, None))
        if isinstance(preds, dict):
            preds = preds["out"]
        preds = preds.squeeze(-1)

        loss = criterion(preds, targets)
        total_loss += loss.item() * len(targets)
        all_preds.extend(preds.cpu().numpy())
        all_tgts.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds_arr = np.array(all_preds)
    tgts_arr  = np.array(all_tgts)
    m = compute_metrics(preds_arr, tgts_arr)
    return avg_loss, m, preds_arr, tgts_arr


# PLOTS

def make_plots(test_preds, test_tgts, history, out_dir):
    C1 = "#2166AC"
    C2 = "#D6604D"
    C3 = "#4DAC26"
    C4 = "#F4A582"
    CG = "#666666"

    errors = test_preds - test_tgts
    m      = compute_metrics(test_preds, test_tgts)

    plt.rcParams.update({
        "font.family"      : "sans-serif",
        "font.size"        : 11,
        "axes.linewidth"   : 1.2,
        "xtick.direction"  : "in",
        "ytick.direction"  : "in",
        "legend.framealpha": 0.9,
    })

    # Fig 1: Parity plot
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(test_tgts, test_preds, s=14, alpha=0.45,
               color=C1, edgecolors="none",
               label="Test set (n=%d)" % len(test_tgts))
    lo = min(test_tgts.min(), test_preds.min()) - 0.3
    hi = max(test_tgts.max(), test_preds.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.4, label="y = x")
    ax.fill_between([lo, hi], [lo-0.2, hi-0.2], [lo+0.2, hi+0.2],
                    color=CG, alpha=0.12, label="$\pm$0.2 eV")
    slope, intercept, _, _, _ = stats.linregress(test_tgts, test_preds)
    x_l = np.linspace(lo, hi, 300)
    ax.plot(x_l, slope*x_l+intercept, color=C2, lw=1.4,
            linestyle="-.", label="Linear fit (slope=%.2f)" % slope)
    textstr = ("MAE  = %.3f eV\nRMSE = %.3f eV\nR$^{2}$ = %.3f") % (
        m["mae"], m["rmse"], m["r2"])
    ax.text(0.04, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            va="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=CG, alpha=0.95))
    ax.set_xlabel("DFT Reaction Energy (eV)", fontsize=12)
    ax.set_ylabel("ALIGNN Predicted Energy (eV)", fontsize=12)
    ax.set_title("ALIGNN: HER Reaction Energy Prediction", fontsize=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    fig.savefig(str(out_dir / "fig1_parity.png"), dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(str(out_dir / "fig1_parity.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_parity")

    # Fig 2: Loss + MAE curves
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], color=C1, lw=1.8, label="Train loss")
    ax1.plot(epochs, history["val_loss"],   color=C2, lw=1.8, label="Val loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE Loss", fontsize=12)
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["train_mae"], color=C1, lw=1.4,
             linestyle="--", label="Train MAE")
    ax2.plot(epochs, history["val_mae"],   color=C2, lw=1.4,
             linestyle="--", label="Val MAE")
    ax2.set_ylabel("MAE (eV)", fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=9, loc="upper right")
    ax1.set_title("Training and Validation Convergence (ALIGNN)", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_dir / "fig2_loss_curves.png"), dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(str(out_dir / "fig2_loss_curves.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_loss_curves")

    # Fig 3: Error distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(errors, bins=55, color=C1, alpha=0.70,
            edgecolor="white", linewidth=0.4, density=True)
    mu, sigma = stats.norm.fit(errors)
    x_fit = np.linspace(errors.min()-0.1, errors.max()+0.1, 400)
    ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sigma), color=C2, lw=2.2,
            label="Gaussian fit\n($\mu$=%.3f, $\sigma$=%.3f eV)" % (mu, sigma))
    ax.axvline(0, color="black", lw=1.2, linestyle="--")
    ax.axvline( m["mae"], color=C3, lw=1.5, linestyle=":",
                label="$\pm$MAE=%.3f eV" % m["mae"])
    ax.axvline(-m["mae"], color=C3, lw=1.5, linestyle=":")
    p90 = np.percentile(np.abs(errors), 90)
    ax.text(0.97, 0.96, "90%% of |errors| < %.3f eV" % p90,
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=CG, alpha=0.9))
    ax.set_xlabel("Prediction Error (eV)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Error Distribution on Test Set (ALIGNN)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(str(out_dir / "fig3_error_dist.png"), dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(str(out_dir / "fig3_error_dist.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_error_dist")

    # Fig 4: Cumulative error
    fig, ax = plt.subplots(figsize=(6, 4.5))
    abs_err_sorted = np.sort(np.abs(errors))
    cum_pct = np.arange(1, len(abs_err_sorted)+1) / len(abs_err_sorted) * 100
    ax.plot(abs_err_sorted, cum_pct, color=C1, lw=2.2, label="ALIGNN")
    for thresh, color in [(0.1, C3), (0.2, C4), (0.3, C2)]:
        pct = float(np.mean(np.abs(errors) <= thresh) * 100)
        ax.axvline(thresh, color=color, lw=1.2, linestyle="--",
                   label="%.1f eV: %.1f%%" % (thresh, pct))
        ax.axhline(pct, color=color, lw=0.8, linestyle=":")
    ax.axvspan(0.0, 0.2, alpha=0.06, color=C3,
               label="BEEF-vdW uncertainty")
    ax.set_xlabel("|Prediction Error| (eV)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction (%)", fontsize=12)
    ax.set_title("Cumulative Error: Screening Threshold Analysis", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 101)
    plt.tight_layout()
    fig.savefig(str(out_dir / "fig4_cumulative_error.png"),
                dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(str(out_dir / "fig4_cumulative_error.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_cumulative_error")

    print("\nAll plots saved to %s" % str(out_dir))


# MAIN

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("ALIGNN Training -- HER Reaction Energy")
    print("Device  : %s" % DEVICE)
    print("Dataset : %s" % DATASET_DIR)
    print("=" * 60)

    # 1. Load id_prop.csv
    print("\nLoading dataset ...")
    df = pd.read_csv(str(DATASET_DIR / "id_prop.csv"),
                     header=None, names=["filename", "target"])
    df = df[(df["target"] >= ENERGY_MIN) & (df["target"] <= ENERGY_MAX)]
    df = df.reset_index(drop=True)
    print("  Total samples: %d" % len(df))

    # 2. Split indices  (same sizes as MEGNet for fair comparison)
    idx = list(range(len(df)))
    idx_tv, idx_test = train_test_split(
        idx, test_size=N_TEST, random_state=RANDOM_SEED)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=N_VAL, random_state=RANDOM_SEED)

    print("  Split -- train: %d  val: %d  test: %d" % (
        len(idx_train), len(idx_val), len(idx_test)))

    filenames = df["filename"].tolist()
    targets   = df["target"].tolist()

    # 3. Build datasets
    print("\nBuilding graph datasets (first epoch will be slow) ...")

    def make_subset(indices):
        fnames  = [filenames[i] for i in indices]
        tgts    = [targets[i]   for i in indices]
        return SlabHGraphDataset(fnames, tgts, DATASET_DIR,
                                 cutoff=CUTOFF,
                                 max_neighbors=MAX_NEIGHBORS)

    train_ds = make_subset(idx_train)
    val_ds   = make_subset(idx_val)
    test_ds  = make_subset(idx_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0,
                              collate_fn=collate_fn)

    # 4. Build ALIGNN model
    print("\nBuilding ALIGNN model ...")
    cfg = ALIGNNConfig(
        name               = "alignn",
        alignn_layers      = ALIGNN_LAYERS,
        gcn_layers         = GCN_LAYERS,
        atom_input_features = 92,
        edge_input_features = 80,
        triplet_input_features = 40,
        embedding_features = EMBEDDING_FEATURES,
        hidden_features    = HIDDEN_FEATURES,
        output_features    = 1,
        classification     = False,
    )
    model = ALIGNN(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print("  Parameters: %d" % n_params)

    # 5. Optimiser + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss()

    # 6. Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("  Max epochs : %d  |  Patience : %d" % (MAX_EPOCHS, PATIENCE))
    print("=" * 60)
    print("%6s %10s %10s %10s %10s %10s %10s" % (
        "Epoch", "TrLoss", "ValLoss", "TrMAE", "ValMAE", "ValRMSE", "ValR2"))
    print("-" * 70)

    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_mae",  "val_mae", "val_rmse", "val_r2"]}

    best_val_mae   = float("inf")
    patience_count = 0
    best_ckpt_path = str(OUT_DIR / "best_model.pt")

    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_m = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_m, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_mae"].append(tr_m["mae"])
        history["val_mae"].append(vl_m["mae"])
        history["val_rmse"].append(vl_m["rmse"])
        history["val_r2"].append(vl_m["r2"])

        if epoch % 10 == 0 or epoch == 1:
            print("%6d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f" % (
                epoch, tr_loss, vl_loss,
                tr_m["mae"], vl_m["mae"], vl_m["rmse"], vl_m["r2"]))

        if vl_m["mae"] < best_val_mae:
            best_val_mae   = vl_m["mae"]
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("\nEarly stopping at epoch %d" % epoch)
                break

    elapsed = time.time() - t_start
    print("\nTraining finished in %.1f min" % (elapsed / 60))

    # 7. Test evaluation
    print("\nEvaluating best model on test set ...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    _, test_m, test_preds, test_tgts = evaluate(model, test_loader, criterion)

    print("\n" + "=" * 42)
    print("TEST SET RESULTS  (ALIGNN)")
    print("=" * 42)
    print("  MAE  : %.4f eV" % test_m["mae"])
    print("  RMSE : %.4f eV" % test_m["rmse"])
    print("  R2   : %.4f"    % test_m["r2"])
    print("  Time : %.1f min" % (elapsed / 60))
    print("=" * 42)

    # 8. Plots
    print("\nGenerating publication plots ...")
    make_plots(test_preds, test_tgts, history, plots_dir)

    # 9. Save everything
    pd.DataFrame({
        "actual_eV"   : test_tgts,
        "predicted_eV": test_preds,
        "error_eV"    : test_preds - test_tgts,
    }).to_csv(str(OUT_DIR / "test_preds.csv"), index=False)

    with open(str(OUT_DIR / "history.pkl"), "wb") as f:
        pickle.dump(history, f)

    with open(str(OUT_DIR / "results.json"), "w") as f:
        json.dump({
            "model"          : "ALIGNN",
            "alignn_layers"  : ALIGNN_LAYERS,
            "gcn_layers"     : GCN_LAYERS,
            "hidden_features": HIDDEN_FEATURES,
            "cutoff"         : CUTOFF,
            "max_neighbors"  : MAX_NEIGHBORS,
            "n_params"       : n_params,
            "dataset_split"  : {"train": len(idx_train),
                                 "val":   len(idx_val),
                                 "test":  len(idx_test)},
            "test_metrics"   : test_m,
            "training_time_min": elapsed / 60,
        }, f, indent=2)

    print("\nSaved:")
    print("  %s" % best_ckpt_path)
    print("  %s" % str(OUT_DIR / "test_preds.csv"))
    print("  %s" % str(OUT_DIR / "results.json"))
    print("  %s" % str(plots_dir))
    print("\nDone.")


if __name__ == "__main__":
    main()
