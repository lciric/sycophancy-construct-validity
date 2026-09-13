"""
fig5_gap_schematic -- reading sycophancy vs. controlling it.

Numbers drawn (all recomputed from results/ at run time):
  read-out rho        results/sae_residual_analysis.json   proj_probe.rho_original
  3-head ablation     results/n200_judge_results.json      judge_results.baseline / .ablate_3heads (N=200)
  rank-3 ablation     results/n200_rank3_judge_results.json baseline_judgments / rank3_judgments (N=200)
The single-direction (rank-1) box carries no number: those runs' raw outputs
were not preserved, so they are reported qualitatively.
Usage: python figures/make_fig5.py   (idempotent)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import (AXIS, BLUE, FILL_BLUE, FILL_GRAY, FILL_ORANGE, INK, INK2,
                    MUTED, ORANGE, WHITE, fmt_p, heads_n200, load, main_rank3,
                    pct, save)

rho = load("sae_residual_analysis.json")["proj_probe"]["rho_original"]
heads = heads_n200()
rank3 = main_rank3()

fig, ax = plt.subplots(figsize=(9.0, 5.5))
ax.set_xlim(0, 120)
ax.set_ylim(0, 74)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                                mutation_aspect=1.0, zorder=2))


def arrow(p1, p2, color, lw=1.4, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", color=color, linewidth=lw,
                                 linestyle=ls, shrinkA=2, shrinkB=2, mutation_scale=11,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def badge(x, y, ok, color):
    """ok: True = works, False = null, None = provisional (not replicated)."""
    glyph = "?" if ok is None else ("✓" if ok else "✗")
    ax.text(x, y, glyph, ha="center", va="center", fontsize=11,
            color=color, fontweight="bold", zorder=4)


ax.text(2, 73.5, "Reading sycophancy is not controlling it", fontsize=11, color=INK,
        fontweight="bold", va="top")
ax.text(2, 69.4, "Llama 3.1 8B Instruct · opinion sycophancy under critique", fontsize=8,
        color=MUTED, va="top")

# source
box(3, 24, 27, 16, FILL_BLUE, BLUE, lw=1.3)
ax.text(16.5, 34.5, "Residual stream", ha="center", va="center", fontsize=9.5,
        color=INK, fontweight="bold")
ax.text(16.5, 29.5, "(same model, same\nopinion items)", ha="center", va="center",
        fontsize=7.4, color=INK2, linespacing=1.3)

rows = [
    # y, title, body, face, edge, ok, dashed, bold
    (51.5, "READ  ·  linear read-out",
     f"projection on a probe direction tracks judged\nsycophancy:  Spearman ρ = {rho:.2f} (training-fit)",
     WHITE, BLUE, True, False, False),
    (37.0, "CONTROL  ·  single direction (rank 1)",
     "CAA steering, directional steering / ablation, SAE clamping,\nfine-tuning: no clean reduction (raw outputs not preserved)",
     FILL_GRAY, AXIS, False, True, False),
    (22.5, f"CONTROL  ·  units: 3 attention heads  (N = {heads.n})",
     f"{pct(heads.rate_base)}% → {pct(heads.rate_treat)}%   {fmt_p(heads.p)}   — null",
     FILL_GRAY, AXIS, False, False, False),
    (8.0, f"ABLATE  ·  rank-3 subspace  (N = {rank3.n})",
     f"judged rate on stored generations:  {pct(rank3.rate_base)}% → {pct(rank3.rate_treat)}%\n"
     f"{fmt_p(rank3.p)}   ({rank3.b} leave / {rank3.c} enter CHANGED_TO_AGREE)",
     FILL_ORANGE, ORANGE, None, False, True),
]
X0, W, H = 52, 64, 11.5
for y, title, body, fc, ec, ok, dashed, bold in rows:
    box(X0, y, W, H, fc, ec, lw=1.6 if bold else 1.1, ls=(0, (4, 3)) if dashed else "-")
    ax.text(X0 + 2.2, y + H - 3.0, title, fontsize=8.4, color=INK, fontweight="bold",
            va="center")
    ax.text(X0 + 2.2, y + 2.2, body, fontsize=7.6, color=INK if bold else INK2,
            va="bottom", linespacing=1.35, fontweight="bold" if bold else "normal")
    badge(X0 + W - 3.2, y + H - 3.0, ok, ORANGE if bold else (BLUE if ok else MUTED))

arrow((30, 36), (X0 - 0.6, 57.2), BLUE, lw=1.5, rad=-0.15)
ax.text(40, 51.8, "read", fontsize=7.8, color=BLUE, fontweight="bold", ha="center")
arrow((30, 33), (X0 - 0.6, 42.7), MUTED, lw=1.2, ls=(0, (4, 3)))
arrow((30, 30), (X0 - 0.6, 28.2), MUTED, lw=1.2)
arrow((30, 27), (X0 - 0.6, 13.7), ORANGE, lw=1.8, rad=0.15)
ax.text(38.5, 13.5, "ablate", fontsize=7.8, color=ORANGE, fontweight="bold", ha="center")

ax.text(2, 1.2,
        "Rank-3 effect measured on stored generations; a preregistered replication with fresh, "
        "seed-matched generations did not reproduce it (see README › Limitations).",
        fontsize=7, color=MUTED, va="bottom")

save(fig, "fig5_gap_schematic")
