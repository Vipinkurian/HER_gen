#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune pretrained MEGNet (MatGL v1.2.1, DGL backend)
for HER reaction energy prediction.

Configuration:
  - state_attr: [mean_NdUnfilled, mean_MendeleevNumber]
  - weight_decay: 1e-4
  - Dropout 0.15 in output MLP
  - Huber delta 0.5
  - Single-stage full fine-tuning, LR=5e-4
  - 200 epochs, patience=25

Plots:
  Fig 1: Parity plot with metrics box
  Fig 2: Training / validation loss curves
  Fig 3: Error distribution with Gaussian fit
  Fig 4: Cumulative error curve
"""

import json
import pickle
import time
import warnings
from io import StringIO
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

from pymatgen.io.cif import CifParser
from matminer.featurizers import composition as cf

# CONFIGURATION

DATA_CSV         = "data/training/MamunHighT2019_SlabH_only_cleaned.csv"   # run from the repository root
OUT_DIR          = Path("megnet_her_v2_final")
PRETRAINED_MODEL = "MEGNet-MP-2018.6.1-Eform"

RANDOM_SEED  = 42
VAL_FRAC     = 0.15
TEST_FRAC    = 0.15
ENERGY_MIN   = -5.0
ENERGY_MAX   =  5.0

BATCH_SIZE   = 32
MAX_EPOCHS   = 200
LR           = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 25
CUTOFF       = 5.0
DROPOUT      = 0.15
HUBER_DELTA  = 0.5

STATE_FEAT_1 = "MagpieData mean NdUnfilled"
STATE_FEAT_2 = "MagpieData mean MendeleevNumber"

PLOT_DPI = 300

# STATE FEATURE COMPUTATION

_magpie        = cf.ElementProperty.from_preset("magpie")
_magpie_labels = _magpie.feature_labels()


def _get_idx(label):
    for i, l in enumerate(_magpie_labels):
        if l == label:
            return i
    raise ValueError("Feature not found: %s" % label)


_IDX1 = _get_idx(STATE_FEAT_1)
_IDX2 = _get_idx(STATE_FEAT_2)

print("state_attr indices -- %s: %d | %s: %d" % (
    STATE_FEAT_1, _IDX1, STATE_FEAT_2, _IDX2))


def compute_state_attr(composition):
    try:
        feats = _magpie.featurize(composition)
        v1 = float(feats[_IDX1])
        v2 = float(feats[_IDX2])
        if np.isnan(v1) or np.isnan(v2):
            return [0.0, 0.0]
        return [v1, v2]
    except Exception:
        return [0.0, 0.0]


class StateNormalizer:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, state_attrs):
        arr = np.array(state_attrs)
        self.mean = arr.mean(axis=0)
        self.std  = arr.std(axis=0)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, state_attrs):
        arr = np.array(state_attrs, dtype=np.float32)
        return ((arr - self.mean) / self.std).astype(np.float32)

    def save(self, path):
        np.save(str(path) + "_mean.npy", self.mean)
        np.save(str(path) + "_std.npy",  self.std)


# DROPOUT PATCH
# output_proj.layers is a ModuleList:
# [Linear, SoftPlus2, Linear, SoftPlus2, Linear]
# Insert Dropout after each SoftPlus2

def patch_output_proj_with_dropout(megnet, dropout_p):
    old_layers = list(megnet.output_proj.layers)
    new_layers = []
    for layer in old_layers:
        new_layers.append(layer)
        if "softplus" in type(layer).__name__.lower():
            new_layers.append(nn.Dropout(p=dropout_p))
    megnet.output_proj.layers = nn.ModuleList(new_layers)
    print("  Output MLP patched with Dropout(%.2f)" % dropout_p)
    return megnet


# HELPERS

def parse_cif_string(cif_str):
    try:
        parser  = CifParser(StringIO(str(cif_str)))
        structs = parser.get_structures(primitive=False)
        return structs[0] if structs else None
    except Exception:
        return None


def compute_metrics(preds, targets):
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    r2   = float(r2_score(targets, preds))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def add_cartesian_coords(g, lattice):
    L = lattice[0]
    g.ndata["pos"]          = torch.mm(g.ndata["frac_coords"], L)
    g.edata["pbc_offshift"] = torch.mm(g.edata["pbc_offset"].float(), L)
    return g


# DATASET

class SlabHDataset(torch.utils.data.Dataset):

    def __init__(self, graphs, state_attrs, energies):
        self.graphs      = graphs
        self.state_attrs = state_attrs
        self.energies    = energies

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph      = self.graphs[idx]
        state_attr = torch.tensor(self.state_attrs[idx], dtype=torch.float32)
        label      = torch.tensor([self.energies[idx]], dtype=torch.float32)
        return graph, state_attr, label


def collate_fn(batch):
    graphs      = [b[0] for b in batch]
    state_attrs = torch.stack([b[1] for b in batch], dim=0)
    labels      = torch.stack([b[2] for b in batch], dim=0)
    bg          = dgl.batch(graphs)
    return bg, state_attrs, labels


def build_graphs(structures, compositions, converter):
    graphs     = []
    state_raws = []
    indices    = []
    failed     = 0
    for i, (struct, comp) in enumerate(zip(structures, compositions)):
        try:
            g, lattice, _ = converter.get_graph(struct)
            g = add_cartesian_coords(g, lattice)
            graphs.append(g)
            state_raws.append(compute_state_attr(comp))
            indices.append(i)
        except Exception:
            failed += 1
    print("    built %d graphs, %d failed" % (len(graphs), failed))
    return graphs, state_raws, indices


# LIGHTNING MODULE

class MEGNetFinetuner(pl.LightningModule):

    def __init__(self, megnet_module, lr, weight_decay, huber_delta):
        super().__init__()
        self.model        = megnet_module
        self.lr           = lr
        self.weight_decay = weight_decay
        self.criterion    = nn.HuberLoss(delta=huber_delta)
        self._train_preds   = []
        self._train_targets = []
        self._val_preds     = []
        self._val_targets   = []

    def forward(self, graph, state_attr):
        return self.model(graph, state_attr)

    def _step(self, batch):
        graph, state_attr, labels = batch
        preds  = self(graph, state_attr).squeeze(-1)
        labels = labels.squeeze(-1).to(preds.device)
        loss   = self.criterion(preds, labels)
        return loss, preds.detach().cpu(), labels.detach().cpu()

    def training_step(self, batch, _):
        loss, p, t = self._step(batch)
        self._train_preds.append(p)
        self._train_targets.append(t)
        self.log("train_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=len(t))
        return loss

    def on_train_epoch_end(self):
        p = torch.cat(self._train_preds).numpy()
        t = torch.cat(self._train_targets).numpy()
        m = compute_metrics(p, t)
        self.log_dict({"train_mae": m["mae"], "train_r2": m["r2"]})
        self._train_preds.clear()
        self._train_targets.clear()

    def validation_step(self, batch, _):
        loss, p, t = self._step(batch)
        self._val_preds.append(p)
        self._val_targets.append(t)
        self.log("val_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=len(t))
        return loss

    def on_validation_epoch_end(self):
        p = torch.cat(self._val_preds).numpy()
        t = torch.cat(self._val_targets).numpy()
        m = compute_metrics(p, t)
        self.log_dict({"val_mae":  m["mae"],
                       "val_rmse": m["rmse"],
                       "val_r2":   m["r2"]},
                      prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=MAX_EPOCHS, eta_min=1e-6
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


# INFERENCE

@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        graph, state_attr, labels = batch
        graph      = graph.to(device)
        state_attr = state_attr.to(device)
        preds      = model(graph, state_attr).squeeze(-1).cpu()
        all_preds.append(preds)
        all_targets.append(labels.squeeze(-1))
    return (torch.cat(all_preds).numpy(),
            torch.cat(all_targets).numpy())


# PLOTS

def make_publication_plots(test_preds, test_tgts, history, out_dir):
    """
    Fig 1 -- Parity plot              (predicted vs DFT)
    Fig 2 -- Loss curves              (train vs val, both MAE and loss)
    Fig 3 -- Error distribution       (histogram + Gaussian fit)
    Fig 4 -- Cumulative error curve   (screening threshold analysis)
    """

    # colourblind-safe palette
    C1 = "#2166AC"   # blue
    C2 = "#D6604D"   # red
    C3 = "#4DAC26"   # green
    C4 = "#F4A582"   # orange
    CG = "#666666"   # grey

    errors = test_preds - test_tgts
    m      = compute_metrics(test_preds, test_tgts)

    plt.rcParams.update({
        "font.family"      : "sans-serif",
        "font.size"        : 11,
        "axes.linewidth"   : 1.2,
        "xtick.direction"  : "in",
        "ytick.direction"  : "in",
        "xtick.major.size" : 4,
        "ytick.major.size" : 4,
        "legend.framealpha": 0.9,
        "legend.edgecolor" : CG,
    })

    # FIG 1: Parity plot
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.scatter(test_tgts, test_preds,
               s=14, alpha=0.45, color=C1,
               edgecolors="none", zorder=3,
               label="Test set (n=%d)" % len(test_tgts))

    lo = min(test_tgts.min(), test_preds.min()) - 0.3
    hi = max(test_tgts.max(), test_preds.max()) + 0.3

    ax.plot([lo, hi], [lo, hi], "k--", lw=1.4, zorder=2, label="y = x")

    # +/- 0.2 eV band
    ax.fill_between([lo, hi],
                    [lo - 0.2, hi - 0.2],
                    [lo + 0.2, hi + 0.2],
                    color=CG, alpha=0.12, zorder=1,
                    label="$\pm$0.2 eV band")

    slope, intercept, _, _, _ = stats.linregress(test_tgts, test_preds)
    x_line = np.linspace(lo, hi, 300)
    ax.plot(x_line, slope * x_line + intercept,
            color=C2, lw=1.4, linestyle="-.", zorder=2,
            label="Linear fit (slope=%.2f)" % slope)

    textstr = ("MAE  = %.3f eV\n"
               "RMSE = %.3f eV\n"
               "R$^{2}$    = %.3f") % (m["mae"], m["rmse"], m["r2"])
    ax.text(0.04, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor="white", edgecolor=CG, alpha=0.95))

    ax.set_xlabel("DFT Reaction Energy (eV)", fontsize=12)
    ax.set_ylabel("Predicted Reaction Energy (eV)", fontsize=12)
    ax.set_title("MEGNet: HER Reaction Energy Prediction", fontsize=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    fig.savefig(str(out_dir / "fig1_parity.png"), dpi=PLOT_DPI,
                bbox_inches="tight")
    fig.savefig(str(out_dir / "fig1_parity.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_parity")

    # FIG 2: Training curves  (loss + MAE on same figure, two y-axes)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    if history is not None:
        df_h   = pd.DataFrame(history)
        t_loss = df_h.dropna(subset=["train_loss"])
        v_loss = df_h.dropna(subset=["val_loss"])
        t_mae  = df_h.dropna(subset=["train_mae"])
        v_mae  = df_h.dropna(subset=["val_mae"])

        if len(t_loss) > 0:
            ax1.plot(t_loss["epoch"], t_loss["train_loss"],
                     color=C1, lw=1.8, label="Train loss")
        if len(v_loss) > 0:
            ax1.plot(v_loss["epoch"], v_loss["val_loss"],
                     color=C2, lw=1.8, label="Val loss")

        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Huber Loss", fontsize=12, color="black")
        ax1.tick_params(axis="y")

        ax2 = ax1.twinx()
        if len(t_mae) > 0:
            ax2.plot(t_mae["epoch"], t_mae["train_mae"],
                     color=C1, lw=1.4, linestyle="--",
                     label="Train MAE")
        if len(v_mae) > 0:
            ax2.plot(v_mae["epoch"], v_mae["val_mae"],
                     color=C2, lw=1.4, linestyle="--",
                     label="Val MAE")
        ax2.set_ylabel("MAE (eV)", fontsize=12, color="black")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=9, loc="upper right")

    ax1.set_title("Training and Validation Convergence", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_dir / "fig2_loss_curves.png"), dpi=PLOT_DPI,
                bbox_inches="tight")
    fig.savefig(str(out_dir / "fig2_loss_curves.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_loss_curves")

    # FIG 3: Error distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))

    counts, bins, _ = ax.hist(
        errors, bins=55, color=C1, alpha=0.70,
        edgecolor="white", linewidth=0.4,
        density=True, label="Prediction errors")

    mu, sigma = stats.norm.fit(errors)
    x_fit = np.linspace(errors.min() - 0.1, errors.max() + 0.1, 400)
    ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sigma),
            color=C2, lw=2.2,
            label="Gaussian fit\n($\mu$=%.3f, $\sigma$=%.3f eV)" % (mu, sigma))

    ax.axvline(0, color="black", lw=1.2, linestyle="--", zorder=3)
    ax.axvline( m["mae"], color=C3, lw=1.5, linestyle=":",
                label="$\pm$MAE = %.3f eV" % m["mae"])
    ax.axvline(-m["mae"], color=C3, lw=1.5, linestyle=":")

    p90 = np.percentile(np.abs(errors), 90)
    ax.text(0.97, 0.96,
            "90%% of |errors| < %.3f eV" % p90,
            transform=ax.transAxes, fontsize=9,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white", edgecolor=CG, alpha=0.9))

    ax.set_xlabel("Prediction Error (eV)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Error Distribution on Test Set", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(str(out_dir / "fig3_error_dist.png"), dpi=PLOT_DPI,
                bbox_inches="tight")
    fig.savefig(str(out_dir / "fig3_error_dist.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_error_dist")

    # FIG 4: Cumulative error curve
    # fraction of structures predicted within a given error
    fig, ax = plt.subplots(figsize=(6, 4.5))

    abs_err_sorted  = np.sort(np.abs(errors))
    cumulative_pct  = np.arange(1, len(abs_err_sorted) + 1) / len(abs_err_sorted) * 100

    ax.plot(abs_err_sorted, cumulative_pct,
            color=C1, lw=2.2, label="MEGNet (fine-tuned)")

    # BEEF-vdW intrinsic uncertainty reference line
    ax.axvline(0.1, color=CG, lw=1.0, linestyle=":")
    ax.axvline(0.2, color=CG, lw=1.0, linestyle=":")
    ax.axvline(0.3, color=CG, lw=1.0, linestyle=":")

    for thresh, color in [(0.1, C3), (0.2, C4), (0.3, C2)]:
        pct = float(np.mean(np.abs(errors) <= thresh) * 100)
        ax.scatter([thresh], [pct], color=color, s=55, zorder=5)
        ax.annotate("%.0f%%" % pct,
                    xy=(thresh, pct),
                    xytext=(thresh + 0.03, pct - 4),
                    fontsize=9, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

    ax.axvspan(0.0, 0.2, alpha=0.06, color=C3,
               label="BEEF-vdW uncertainty (~0.2 eV)")

    ax.set_xlabel("|Prediction Error| (eV)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction (%)", fontsize=12)
    ax.set_title("Cumulative Error: Screening Threshold Analysis", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 101)
    plt.tight_layout()
    fig.savefig(str(out_dir / "fig4_cumulative_error.png"), dpi=PLOT_DPI,
                bbox_inches="tight")
    fig.savefig(str(out_dir / "fig4_cumulative_error.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_cumulative_error")

    print("\nAll plots saved to %s" % str(out_dir))
    print("Summary metrics -- MAE: %.4f eV | RMSE: %.4f eV | R2: %.4f" % (
        m["mae"], m["rmse"], m["r2"]))


# MAIN

def main():
    pl.seed_everything(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("MEGNet Fine-tuning v2 FINAL -- HER Reaction Energy")
    print("MatGL v%s  |  Device: %s" % (matgl.__version__, device))
    print("=" * 60)

    # 1. Load dataset
    print("\nLoading dataset ...")
    try:
        df = pd.read_csv(DATA_CSV, sep="\t")
        if df.shape[1] < 3:
            raise ValueError
    except Exception:
        df = pd.read_csv(DATA_CSV)

    print("  Raw rows    : %d" % len(df))
    df = df.dropna(subset=["structure", "reactionEnergy"])
    df = df[(df["reactionEnergy"] >= ENERGY_MIN) &
            (df["reactionEnergy"] <= ENERGY_MAX)]
    print("  After filter: %d rows" % len(df))

    # 2. Parse CIF strings
    print("\nParsing CIF structures ...")
    structures, compositions, energies = [], [], []
    n_failed = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 1000 == 0:
            print("  %d / %d ..." % (i + 1, len(df)))
        s = parse_cif_string(row["structure"])
        if s is not None:
            structures.append(s)
            compositions.append(s.composition)
            energies.append(float(row["reactionEnergy"]))
        else:
            n_failed += 1
    print("  Parsed : %d  Failed: %d" % (len(structures), n_failed))

    # 3. Split
    idx = list(range(len(structures)))
    idx_tv, idx_test = train_test_split(
        idx, test_size=TEST_FRAC, random_state=RANDOM_SEED)
    idx_train, idx_val = train_test_split(
        idx_tv,
        test_size=VAL_FRAC / (1.0 - TEST_FRAC),
        random_state=RANDOM_SEED,
    )

    def subset(indices):
        return ([structures[i]   for i in indices],
                [compositions[i] for i in indices],
                [energies[i]     for i in indices])

    train_s, train_c, train_e = subset(idx_train)
    val_s,   val_c,   val_e   = subset(idx_val)
    test_s,  test_c,  test_e  = subset(idx_test)

    print("\nSplit -- train: %d  val: %d  test: %d" % (
        len(train_s), len(val_s), len(test_s)))

    # 4. Element list and converter
    all_structs = train_s + val_s + test_s
    elem_list   = get_element_list(all_structs)
    print("Element vocab : %d elements" % len(elem_list))
    converter   = Structure2Graph(element_types=elem_list, cutoff=CUTOFF)

    # 5. Build graphs
    print("\nBuilding DGL graphs + computing state features ...")
    t0 = time.time()

    print("  Training set ...")
    train_graphs, train_sa_raw, train_ok = build_graphs(
        train_s, train_c, converter)
    train_energies = [train_e[i] for i in train_ok]

    print("  Validation set ...")
    val_graphs, val_sa_raw, val_ok = build_graphs(
        val_s, val_c, converter)
    val_energies = [val_e[i] for i in val_ok]

    print("  Test set ...")
    test_graphs, test_sa_raw, test_ok = build_graphs(
        test_s, test_c, converter)
    test_energies = [test_e[i] for i in test_ok]

    print("  Done in %.1fs" % (time.time() - t0))
    print("  Final -- train: %d  val: %d  test: %d" % (
        len(train_graphs), len(val_graphs), len(test_graphs)))

    # 6. Normalise state_attr (fit on train only -- no leakage)
    print("\nNormalising state features ...")
    normalizer = StateNormalizer()
    normalizer.fit(train_sa_raw)
    print("  %s -- mean: %.4f  std: %.4f" % (
        STATE_FEAT_1, normalizer.mean[0], normalizer.std[0]))
    print("  %s -- mean: %.4f  std: %.4f" % (
        STATE_FEAT_2, normalizer.mean[1], normalizer.std[1]))

    train_sa = normalizer.transform(train_sa_raw)
    val_sa   = normalizer.transform(val_sa_raw)
    test_sa  = normalizer.transform(test_sa_raw)

    # 7. DataLoaders
    train_ds = SlabHDataset(train_graphs, train_sa, train_energies)
    val_ds   = SlabHDataset(val_graphs,   val_sa,   val_energies)
    test_ds  = SlabHDataset(test_graphs,  test_sa,  test_energies)

    def make_loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=0, collate_fn=collate_fn)

    train_loader = make_loader(train_ds, True)
    val_loader   = make_loader(val_ds,   False)
    test_loader  = make_loader(test_ds,  False)

    # 8. Load pretrained MEGNet + patch dropout
    print("\nLoading pretrained model: %s ..." % PRETRAINED_MODEL)
    pretrained = matgl.load_model(PRETRAINED_MODEL)
    megnet     = pretrained.model
    print("  Parameters  : %d" % sum(p.numel() for p in megnet.parameters()))
    megnet = patch_output_proj_with_dropout(megnet, DROPOUT)

    # 9. Train
    ckpt_cb = ModelCheckpoint(
        dirpath  = str(OUT_DIR / "checkpoints"),
        filename = "best-{epoch:03d}-{val_mae:.4f}",
        monitor="val_mae", mode="min", save_top_k=1)
    es_cb  = EarlyStopping(monitor="val_mae", patience=PATIENCE, mode="min")
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    logger = CSVLogger(save_dir=str(OUT_DIR), name="logs")

    lit = MEGNetFinetuner(
        megnet_module=megnet,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        huber_delta=HUBER_DELTA,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        callbacks=[ckpt_cb, es_cb, lr_cb],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=5.0,
    )

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    t_train = time.time()
    trainer.fit(lit, train_loader, val_loader)
    elapsed = time.time() - t_train
    print("\nTraining finished in %.1f min" % (elapsed / 60))

    # 10. Test evaluation
    print("\nEvaluating best checkpoint on test set ...")
    best_ckpt = ckpt_cb.best_model_path
    print("  Best checkpoint: %s" % best_ckpt)

    best_lit = MEGNetFinetuner.load_from_checkpoint(
        best_ckpt,
        megnet_module=megnet,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        huber_delta=HUBER_DELTA,
    )
    best_net = best_lit.model.to(device)

    test_preds, test_tgts = evaluate_loader(best_net, test_loader, device)
    test_m = compute_metrics(test_preds, test_tgts)

    print("\n" + "=" * 42)
    print("TEST SET RESULTS")
    print("=" * 42)
    print("  MAE  : %.4f eV" % test_m["mae"])
    print("  RMSE : %.4f eV" % test_m["rmse"])
    print("  R2   : %.4f"    % test_m["r2"])
    print("  Time : %.1f min" % (elapsed / 60))
    print("=" * 42)

    # 11. Publication plots
    print("\nGenerating publication plots ...")
    log_files = list((OUT_DIR / "logs").rglob("metrics.csv"))
    history   = None
    if log_files:
        history = pd.read_csv(str(log_files[0])).to_dict("list")

    make_publication_plots(test_preds, test_tgts, history, plots_dir)

    # 12. Save model and results
    model_save = OUT_DIR / "best_model"
    model_save.mkdir(exist_ok=True)

    torch.save(best_net.state_dict(),
               str(model_save / "model_state.pt"))
    with open(str(model_save / "element_list.json"), "w") as f:
        json.dump(elem_list, f)
    with open(str(model_save / "config.json"), "w") as f:
        json.dump({
            "cutoff"        : CUTOFF,
            "pretrained"    : PRETRAINED_MODEL,
            "state_feat_1"  : STATE_FEAT_1,
            "state_feat_2"  : STATE_FEAT_2,
            "state_attr_dim": 2,
            "dropout"       : DROPOUT,
            "huber_delta"   : HUBER_DELTA,
            "lr"            : LR,
            "weight_decay"  : WEIGHT_DECAY,
        }, f, indent=2)

    normalizer.save(str(model_save / "state_normalizer"))
    print("Model saved     : %s" % str(model_save))

    pd.DataFrame({
        "actual_eV"   : test_tgts,
        "predicted_eV": test_preds,
        "error_eV"    : test_preds - test_tgts,
    }).to_csv(str(OUT_DIR / "test_preds.csv"), index=False)
    print("Predictions     : %s" % str(OUT_DIR / "test_preds.csv"))

    with open(str(OUT_DIR / "results.json"), "w") as f:
        json.dump({
            "version"         : "v2_final",
            "pretrained_model": PRETRAINED_MODEL,
            "state_features"  : [STATE_FEAT_1, STATE_FEAT_2],
            "dropout"         : DROPOUT,
            "huber_delta"     : HUBER_DELTA,
            "lr"              : LR,
            "weight_decay"    : WEIGHT_DECAY,
            "dataset_split"   : {"train": len(train_graphs),
                                  "val":   len(val_graphs),
                                  "test":  len(test_graphs)},
            "test_metrics"     : test_m,
            "training_time_min": elapsed / 60,
            "best_checkpoint"  : str(best_ckpt),
        }, f, indent=2)
    print("Summary         : %s" % str(OUT_DIR / "results.json"))
    print("\nDone.")


if __name__ == "__main__":
    main()
