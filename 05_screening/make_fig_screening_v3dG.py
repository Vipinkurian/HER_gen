#!/usr/bin/env python3
"""Paper Figs 3-5 (screening distribution, facet/site, consensus) on the
free-energy scale: Delta G_H = GNN Delta E_H (Mamun, electronic) + 0.24 eV.
Reads data/predictions/alignn_predictions_all.csv and consensus_all.csv and writes
the figures next to this script."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/predictions"
OUT = Path(__file__).resolve().parent
CHE = 0.24
mpl.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 11, "savefig.dpi": 300,
    "savefig.bbox": "tight", "xtick.direction": "in", "ytick.direction": "in"})
CA, CM, CG = "#2166AC", "#D6604D", "#4DAC26"
DG = r"$\Delta G_{\mathrm{H}^*}$"
DFT4 = ["Pd5Au", "Pd13Ru3", "TiPd5", "Cu3Pd13"]

al = pd.read_csv(P / "alignn_predictions_all.csv")
al["dG"] = al["predicted_dG"] + CHE
best = al.loc[(al["dG"].abs()).groupby(al["bulk_file"]).idxmin()]
cons = pd.read_csv(P / "consensus_all.csv")


def save(fig, name):
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{e}")
    plt.close(fig)
    print("wrote", name)


def window(ax, vertical=True):
    (ax.axvspan if vertical else ax.axhspan)(-0.2, 0.2, color=CG, alpha=0.13, zorder=0)
    (ax.axvline if vertical else ax.axhline)(0, ls="--", color="0.3", lw=1.1, zorder=1)


# ---- Fig 3: distributions
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
bins = np.linspace(-1.0, 1.6, 105)
for ax, v, t in [(axs[0], al["dG"], f"(a) All {len(al):,} slab+H"),
                 (axs[1], best["dG"], f"(b) Best site per bulk ({len(best)})")]:
    window(ax)
    ax.hist(v, bins=bins, color=CA, alpha=0.85, zorder=2)
    ax.set(xlabel=f"ALIGNN {DG} (eV)", ylabel="Count", title=t, xlim=(-1.0, 1.6))
save(fig, "fig_screening_dist")

# ---- Fig 4: facet violins (best site) + site-type counts (all)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ax = axs[0]; window(ax, vertical=False)
fac = [100, 110, 111]
data = [best.loc[best["miller"] == f, "dG"].values for f in fac]
vp = ax.violinplot(data, showmedians=True)
for b in vp["bodies"]:
    b.set_facecolor(CA); b.set_alpha(0.55)
for k in ("cbars", "cmins", "cmaxes", "cmedians"):
    vp[k].set_color(CA)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([f"({f})" for f in fac])
ax.set(ylabel=f"best-site {DG} (eV)", title="(a) By facet", ylim=(-1.0, 1.4))
ax = axs[1]
cnt = al["site_type"].value_counts().reindex(["bridge", "hollow", "ontop"])
bars = ax.bar(["bridge", "hollow", "on-top"], cnt.values, color=[CA, CM, CG], alpha=0.9)
for b, n in zip(bars, cnt.values):
    ax.text(b.get_x() + b.get_width() / 2, n, f"{100*n/len(al):.1f}%", ha="center", va="bottom", fontsize=10)
ax.set(ylabel="Count", title=f"(b) Site types ({len(al):,})")
fig.subplots_adjust(wspace=0.32)
save(fig, "fig_facet")

# ---- Fig 5: parity + hit rates
fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]; L = (-0.8, 1.2)
ax.axvspan(-0.2, 0.2, color=CG, alpha=0.08); ax.axhspan(-0.2, 0.2, color=CG, alpha=0.08)
ax.axhline(0, ls="--", lw=0.8, color="0.5"); ax.axvline(0, ls="--", lw=0.8, color="0.5")
ax.plot(L, L, color="0.3", lw=1.2)
sc = ax.scatter(cons["predicted_dG_al"], cons["predicted_dG_mg"], c=cons["mean_abs_dG"].clip(0, 0.5),
                cmap="YlOrRd", vmin=0, vmax=0.5, s=16, alpha=0.75, lw=0, zorder=3)
d4 = cons[cons["formula"].isin(DFT4)]
ax.scatter(d4["predicted_dG_al"], d4["predicted_dG_mg"], marker="*", s=260, color="#FF8C00",
           edgecolor="k", lw=0.8, zorder=5)
r = np.corrcoef(cons["predicted_dG_al"], cons["predicted_dG_mg"])[0, 1]
ax.text(0.04, 0.96, f"$r = {r:.3f}$\n$n = {len(cons)}$", transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax.legend(handles=[Line2D([], [], marker="*", ls="", ms=14, mfc="#FF8C00", mec="k",
                          label="DFT candidates (best site)")], loc="lower right", fontsize=9)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02); cb.set_label(r"mean$|\Delta G|$ (eV)")
ax.set(xlim=L, ylim=L, xlabel=f"ALIGNN {DG} (eV)", ylabel=f"MEGNet {DG} (eV)",
       title="(a) Inter-model agreement"); ax.set_aspect("equal")
ax = axs[1]; thr = [0.1, 0.2, 0.3]; x = np.arange(3); w = 0.26
rates = [(CA, "ALIGNN", [100 * (cons["abs_dG_al"] <= t).mean() for t in thr]),
         (CM, "MEGNet", [100 * (cons["abs_dG_mg"] <= t).mean() for t in thr]),
         (CG, "Consensus", [100 * (cons["max_abs_dG"] <= t).mean() for t in thr])]
for k, (c, lab, v) in enumerate(rates):
    bs = ax.bar(x + (k - 1) * w, v, w, color=c, alpha=0.9, label=lab, edgecolor="white")
    for b in bs:
        ax.text(b.get_x() + w / 2, b.get_height() + 0.4, f"{b.get_height():.1f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([r"$\leq0.1$", r"$\leq0.2$", r"$\leq0.3$"])
ax.set(xlabel=r"$|\Delta G_{\mathrm{H}^*}|$ threshold (eV)", ylabel=f"% of {len(cons)} bulks",
       title="(b) Best-site hit rates", ylim=(0, max(rates[1][2]) * 1.25))
ax.legend(loc="upper left"); ax.yaxis.grid(True, alpha=0.35); ax.set_axisbelow(True)
fig.subplots_adjust(wspace=0.55)
save(fig, "fig_consensus")
