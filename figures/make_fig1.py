"""
fig1_rank_sweep -- sycophancy rate under critique vs. rank of the ablated subspace.

Sources (all counts recomputed from per-item judgments):
  results/multirank_judge_baseline.json   judgments[50]   pilot baseline
  results/multirank_judge_rank-3.json     judgments[50]   pilot rank 3
  results/multirank_judge_rank-5.json     judgments[50]   pilot rank 5
  results/multirank_judge_rank-10.json    judgments[50]   pilot rank 10
  results/n200_rank3_judge_results.json   baseline_judgments[200], rank3_judgments[200]

Ranks 1-2, 4 and 6-9 have no valid result file in results/ and are not plotted.
Error bars: Wilson 95% intervals. p: exact McNemar vs. the run's own baseline.
Usage: python figures/make_fig1.py   (idempotent; overwrites fig1_rank_sweep.pdf/.png)
"""

import matplotlib.pyplot as plt

from common import (AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, WHITE,
                    fmt_p, main_rank3, pct, pilot_sweep, save, wilson)

pilot = pilot_sweep()
main = main_rank3()

fig, ax = plt.subplots(figsize=(6.6, 4.0))
OFF = 0.16


def point(x, k, n, color, marker, z=3):
    lo, hi = wilson(k, n)
    y = 100 * k / n
    ax.errorbar(x, y, yerr=[[y - 100 * lo], [100 * hi - y]], fmt="none",
                ecolor=color, elinewidth=1.4, capsize=0, zorder=z)
    ax.plot(x, y, marker=marker, markersize=7.5, color=color,
            markeredgecolor=WHITE, markeredgewidth=1.0, zorder=z + 1)
    return y, 100 * lo, 100 * hi


# ---- pilot, N = 50 (one baseline shared by ranks 3 / 5 / 10)
kb, nb = pilot["baseline"]
point(-OFF, kb, nb, BLUE, "o")
for r in (3, 5, 10):
    pr = pilot[r]
    y, lo, hi = point(r - OFF, pr.k_treat, pr.n, BLUE, "o")
    ax.text(r - OFF, lo - 2.2, fmt_p(pr.p), ha="center", va="top",
            fontsize=7.8, color=INK2)

# ---- confirmation, N = 200 (its own baseline)
point(OFF, main.k_base, main.n, ORANGE, "D")
y, lo, hi = point(3 + OFF, main.k_treat, main.n, ORANGE, "D")
ax.text(3 + OFF, hi + 2.0, fmt_p(main.p), ha="center", va="bottom",
        fontsize=8, color=INK, fontweight="bold")
ax.text(3 + OFF, hi + 6.6, f"discordant: {main.b} stop · {main.c} start",
        ha="center", va="bottom", fontsize=7.2, color=INK2)

# direct value labels on the two N = 200 points only (the headline)
ax.text(OFF + 0.28, 100 * main.rate_base, f"{pct(main.rate_base)}%", ha="left",
        va="center", fontsize=7.8, color=INK2)
ax.text(3 + OFF - 0.30, 100 * main.rate_treat + 0.2, f"{pct(main.rate_treat)}%",
        ha="right", va="bottom", fontsize=7.8, color=INK2)

# ---- axes
measured = {0, 3, 5, 10}
ax.set_xticks(range(0, 11))
ax.set_xticklabels(["none"] + [str(k) for k in range(1, 11)])
for k, lab in enumerate(ax.get_xticklabels()):
    lab.set_color(INK2 if k in measured else AXIS)
ax.set_xlim(-0.8, 10.8)
ax.set_ylim(0, 80)
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_yticklabels([f"{v}%" for v in (0, 20, 40, 60, 80)])
ax.yaxis.grid(True, color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.set_xlabel("Rank k of the ablated subspace", fontsize=8.5)
ax.set_ylabel("Sycophancy under critique\n(% items judged CHANGED_TO_AGREE)", fontsize=8.5)

ax.set_title("Subspace-ablation rank sweep", loc="left", fontsize=10.5,
             color=INK, fontweight="bold", pad=22)
ax.text(0, 1.035, "95% Wilson intervals · p = exact McNemar vs. the same run's baseline",
        transform=ax.transAxes, fontsize=7.8, color=MUTED, va="bottom")

h1 = ax.plot([], [], "o", color=BLUE, markersize=7, markeredgecolor=WHITE,
             label=f"Pilot, N = {nb} (ranks 3, 5, 10; shared baseline)")[0]
h2 = ax.plot([], [], "D", color=ORANGE, markersize=6.5, markeredgecolor=WHITE,
             label=f"Confirmation, N = {main.n} (rank 3)")[0]
ax.legend(handles=[h2, h1], loc="lower right", frameon=False, fontsize=7.8,
          bbox_to_anchor=(1.0, 0.07))
ax.text(1.0, 0.015, "Grey ranks (1–2, 4, 6–9): no valid result file in results/ — not plotted.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)

save(fig, "fig1_rank_sweep")
