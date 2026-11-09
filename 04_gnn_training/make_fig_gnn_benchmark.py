#!/usr/bin/env python3
"""Surrogate benchmarking figure: (a) ALIGNN and (b) MEGNet test-set parity on the
1,330 held-out Mamun slab+H structures (fig_gnn_benchmark, main text), plus a
three-panel version adding (c) training/validation MAE per epoch
(fig_gnn_benchmark_with_curves, repository / SI).
Labels are the Mamun electronic reaction energies Delta E_H* (BEEF-vdW).
Data: data/gnn_test/ (alignn_test_preds.csv, megnet_test_preds.csv, alignn_history.pkl,
      megnet_metrics.csv). The figures are written next to this script."""
import pickle
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "data/gnn_test"
OUT = Path(__file__).resolve().parent
mpl.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 9, "axes.labelsize": 9.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "xtick.direction": "in", "ytick.direction": "in"})
CA, CM = "#2166AC", "#D6604D"
DE = r"$\Delta E_{\mathrm{H}^*}$"


def parity(ax, csv, name, col):
    t = pd.read_csv(csv); a, p = t.iloc[:, 0].values, t.iloc[:, 1].values
    e = p - a; mae = np.abs(e).mean(); rmse = np.sqrt((e ** 2).mean())
    r2 = 1 - (e ** 2).sum() / ((a - a.mean()) ** 2).sum(); k, b = np.polyfit(a, p, 1)
    L = (-2.9, 2.6)
    ax.fill_between(L, [L[0] - 0.2, L[1] - 0.2], [L[0] + 0.2, L[1] + 0.2], color="0.88", lw=0, zorder=0)
    ax.plot(L, L, "k--", lw=0.9, zorder=1)
    ax.plot(L, [k * L[0] + b, k * L[1] + b], color="0.35", lw=0.9, ls="-.", zorder=1)
    ax.scatter(a, p, s=6, color=col, alpha=0.45, lw=0, zorder=2, rasterized=True)
    ax.set(xlim=L, ylim=L, xlabel=f"DFT {DE} (eV)", ylabel=f"{name} {DE} (eV)")
    ax.set_aspect("equal")
    ax.text(0.97, 0.04, f"MAE {mae:.3f} eV\n$R^2$ {r2:.3f}\nslope {k:.2f}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9))
    ax.text(0.05, 0.95, name, transform=ax.transAxes, va="top", fontsize=8.5, fontweight="bold", color=col)


# ---- main-text figure: parity plots only
fig2, ax2 = plt.subplots(1, 2, figsize=(5.4, 2.75))
parity(ax2[0], G / "alignn_test_preds.csv", "ALIGNN", CA)
parity(ax2[1], G / "megnet_test_preds.csv", "MEGNet", CM)
for ax, lab in zip(ax2, "ab"):
    ax.text(-0.02, 1.04, f"({lab})", transform=ax.transAxes, fontsize=9.5, fontweight="bold", ha="right")
fig2.tight_layout(w_pad=1.5)
for e in ("pdf", "png"):
    fig2.savefig(OUT / f"fig_gnn_benchmark.{e}")
plt.close(fig2)

# ---- three-panel version with learning curves (kept for the repository / SI)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1, 1, 1.15]})
parity(axs[0], G / "alignn_test_preds.csv", "ALIGNN", CA)
parity(axs[1], G / "megnet_test_preds.csv", "MEGNet", CM)
for ax, lab in zip(axs, "abc"):
    ax.text(-0.02, 1.04, f"({lab})", transform=ax.transAxes, fontsize=9.5, fontweight="bold", ha="right")

ax = axs[2]
h = pickle.load(open(G / "alignn_history.pkl", "rb"))
ep = np.arange(1, len(h["val_mae"]) + 1)
ax.plot(ep, h["train_mae"], color=CA, lw=1.0, ls="--", label="ALIGNN train")
ax.plot(ep, h["val_mae"], color=CA, lw=1.3, label="ALIGNN validation")
m = pd.read_csv(G / "megnet_metrics.csv")
tr = m.dropna(subset=["train_mae"]).groupby("epoch").train_mae.mean()
va = m.dropna(subset=["val_mae"]).groupby("epoch").val_mae.mean()
ax.plot(tr.index + 1, tr.values, color=CM, lw=1.0, ls="--", label="MEGNet train")
ax.plot(va.index + 1, va.values, color=CM, lw=1.3, label="MEGNet validation")
for x, y, c in [(ep[np.argmin(h["val_mae"])], min(h["val_mae"]), CA), (va.idxmin() + 1, va.min(), CM)]:
    ax.scatter([x], [y], marker="o", s=22, facecolor="white", edgecolor=c, lw=1.2, zorder=5)
ax.set(xlabel="Epoch", ylabel="MAE (eV)", xlim=(0, 202), ylim=(0.05, 0.45))
ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(w_pad=1.2)
for e in ("pdf", "png"):
    fig.savefig(OUT / f"fig_gnn_benchmark_with_curves.{e}")
print("wrote fig_gnn_benchmark", "| ALIGNN best epoch", ep[np.argmin(h["val_mae"])], "| MEGNet best epoch", va.idxmin() + 1)
