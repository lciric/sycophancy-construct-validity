"""
fig2_interventions -- forest plot of every judged intervention stored in results/.

Effect = sycophancy rate (intervention) - rate (the same run's baseline), items
paired by index; 95% Newcombe interval for a paired difference; exact McNemar p.

Sources (per-item judgments):
  results/n200_rank3_judge_results.json      baseline_judgments / rank3_judgments           (N=200)
  results/multirank_judge_{baseline,rank-3,rank-5,rank-10}.json   judgments                 (N=50)
  results/n200_judge_results.json            judge_results.baseline / .ablate_3heads          (N=200)
  results/definitive_judge_results.json      {opus,sonnet}.ablation_baseline / .ablation_3heads / .ablation_L0H29   (N=50, critique)
  results/expert_positive_judge_results.json {opus,sonnet}.baseline / .ablate_3heads / .ablate_L0H29              (N=50, praise)

Not plotted: das_judge_results.json (DAS; baseline pairing not established, file
outside the published scope), rank-1/2/4 pilot files (invalid pairing, not in results/).
Usage: python figures/make_fig2.py   (idempotent)
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import (AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, WHITE,
                    dual_judge_head_ablations, fmt_p, heads_n200, main_rank3,
                    pilot_sweep, save, signed_pp)

pilot = pilot_sweep()
dual = dual_judge_head_ablations()

# (group, label, color, [(judge_tag, Paired)], headline)
rows = [
    ("Subspace ablation (SVD at the decision point)", None),
    ("Rank 3 · critique · N = 200", ORANGE, [("", main_rank3())], True),
    ("Rank 3 · critique · N = 50 (pilot)", BLUE, [("", pilot[3])], False),
    ("Rank 5 · critique · N = 50 (pilot)", BLUE, [("", pilot[5])], False),
    ("Rank 10 · critique · N = 50 (pilot)", BLUE, [("", pilot[10])], False),
    ("Attention-head ablation (units)", None),
    ("3 heads · critique · N = 200", ORANGE, [("", heads_n200())], False),
    ("3 heads · critique · N = 50", BLUE,
     [("Opus", dual[("critique", "3-head")]["opus"]), ("Sonnet", dual[("critique", "3-head")]["sonnet"])], False),
    ("L0.H29 · critique · N = 50", BLUE,
     [("Opus", dual[("critique", "L0.H29")]["opus"]), ("Sonnet", dual[("critique", "L0.H29")]["sonnet"])], False),
    ("3 heads · expert praise · N = 50", BLUE,
     [("Opus", dual[("praise", "3-head")]["opus"]), ("Sonnet", dual[("praise", "3-head")]["sonnet"])], False),
    ("L0.H29 · expert praise · N = 50", BLUE,
     [("Opus", dual[("praise", "L0.H29")]["opus"]), ("Sonnet", dual[("praise", "L0.H29")]["sonnet"])], False),
]

# vertical layout (top = 0, growing downwards)
y = 0.0
layout = []
for row in rows:
    if row[1] is None:
        y += 0.35 if layout else 0.0
        layout.append(("header", y, row))
        y += 0.85
    else:
        n_lines = len(row[2])
        h = 1.0 if n_lines == 1 else 1.45
        layout.append(("row", y + (h - 1.0) / 2 + 0.5, row))
        y += h
y_max = y

fig, ax = plt.subplots(figsize=(8.6, 5.6))
XMIN, XMAX = -45, 32
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(y_max + 0.1, -0.35)
ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
ax.xaxis.grid(True, color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.tick_params(length=0)
ax.set_xticks([-40, -30, -20, -10, 0, 10, 20, 30])
ax.set_xticklabels([f"{v:+d}".replace("-", "−") if v else "0" for v in (-40, -30, -20, -10, 0, 10, 20, 30)])
ax.set_xlabel("Change in sycophancy rate, intervention − baseline (percentage points)", fontsize=8.5)

trans_lab = ax.get_yaxis_transform()  # x in axes coords, y in data coords
for kind, yc, row in layout:
    if kind == "header":
        ax.text(-0.02, yc + 0.35, row[0], transform=trans_lab, ha="right", va="center",
                fontsize=8.3, color=INK, fontweight="bold")
        continue
    label, color, entries, headline = row
    ax.text(-0.02, yc, label + ("  ★" if headline else ""), transform=trans_lab,
            ha="right", va="center", fontsize=8, color=INK if headline else INK2,
            fontweight="bold" if headline else "normal")
    offs = [0.0] if len(entries) == 1 else [-0.2, 0.2]
    for (tag, pr), dy in zip(entries, offs):
        yy = yc + dy
        hollow = tag == "Sonnet"
        ax.plot([100 * pr.lo, 100 * pr.hi], [yy, yy], color=color, linewidth=1.6,
                solid_capstyle="round", zorder=3)
        ax.plot(100 * pr.diff, yy, "o", markersize=7 if headline else 6,
                markerfacecolor=WHITE if hollow else color, markeredgecolor=color,
                markeredgewidth=1.4, zorder=4)
        txt = f"{signed_pp(pr.diff)} [{signed_pp(pr.lo)}, {signed_pp(pr.hi)}]"
        who = f"{tag}  " if tag else ""
        ax.text(1.02, yy, who + txt, transform=trans_lab, ha="left", va="center",
                fontsize=7.6 if tag else 8, color=INK if headline else INK2,
                fontweight="bold" if headline else "normal")
        ax.text(1.36, yy, fmt_p(pr.p), transform=trans_lab, ha="left", va="center",
                fontsize=7.6 if tag else 8, color=INK if headline else INK2,
                fontweight="bold" if headline else "normal")

# column heads
ax.text(1.02, -0.3, "Δ pp [95% CI]", transform=trans_lab, ha="left", va="center",
        fontsize=7.8, color=MUTED)
ax.text(1.36, -0.3, "McNemar", transform=trans_lab, ha="left", va="center",
        fontsize=7.8, color=MUTED)
ax.text(XMIN + 1, y_max + 0.05, "← less sycophantic", ha="left", va="bottom",
        fontsize=7.4, color=MUTED)
ax.text(XMAX - 1, y_max + 0.05, "more sycophantic →", ha="right", va="bottom",
        fontsize=7.4, color=MUTED)

ax.set_title("Every judged intervention in results/", loc="left", fontsize=10.5,
             color=INK, fontweight="bold", pad=24, x=-0.62)
ax.text(-0.62, 1.035, "Paired difference vs. the same run's baseline · Newcombe 95% CI · exact McNemar p",
        transform=ax.transAxes, fontsize=7.8, color=MUTED, va="bottom")

handles = [
    Line2D([], [], marker="o", color=ORANGE, linestyle="-", markersize=6, label="N = 200"),
    Line2D([], [], marker="o", color=BLUE, linestyle="-", markersize=6, label="N = 50"),
    Line2D([], [], marker="o", color=MUTED, markerfacecolor=WHITE, linestyle="none",
           markersize=6, label="hollow = Sonnet re-judging (dual-judge runs)"),
]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.62, -0.1), ncol=3,
          frameon=False, fontsize=7.6, handlelength=1.6, columnspacing=1.4)
fig.text(0.01, -0.035,
         "Exact McNemar (conditional) and the Newcombe score interval can disagree near α = 0.05 (rank-5 pilot: CI excludes 0, p = 0.052).",
         fontsize=7, color=MUTED, ha="left")

save(fig, "fig2_interventions")
