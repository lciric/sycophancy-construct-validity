"""
fig3_asymmetry -- the model yields to criticism far more than to praise.

Sources (per-item judgments; sycophancy = CHANGED_TO_AGREE):
  critique (main)  results/n200_rank3_judge_results.json   baseline_judgments[200]      -> 122/200
  praise           results/expert_positive_judge_results.json  {opus,sonnet}.baseline[50] -> 3/50 (both judges)
  other critique baselines (grey dots, for context):
                   results/n200_judge_results.json        judge_results.baseline[200]
                   results/multirank_judge_baseline.json  judgments[50]
                   results/definitive_judge_results.json  {opus,sonnet}.{opinion_paradox,ablation_baseline}.judgments[50]
The no-pressure condition is the judge's reference response: 0 by construction, no interval.
Error bars: Wilson 95%.  Usage: python figures/make_fig3.py   (idempotent)
"""

import matplotlib.pyplot as plt

from common import (AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, WHITE,
                    critique_baselines, pct, praise_baseline, save, wilson)

runs = critique_baselines()
(k_c, n_c) = runs[0][1]                 # main N = 200 run
others = [kn for _, kn in runs[1:]]
(k_p, n_p), sonnet_praise = praise_baseline()
assert (k_p, n_p) == sonnet_praise, "praise baseline differs between judges"

fig, ax = plt.subplots(figsize=(5.6, 4.0))
X = {"none": 0, "praise": 1, "critique": 2}
W = 0.46


def bar(x, k, n, color):
    lo, hi = wilson(k, n)
    y = 100 * k / n
    ax.bar(x, y, width=W, color=color, zorder=2)
    ax.errorbar(x, y, yerr=[[y - 100 * lo], [100 * hi - y]], fmt="none",
                ecolor=INK2, elinewidth=1.2, capsize=0, zorder=3)
    return y, 100 * hi


y, top = bar(X["praise"], k_p, n_p, BLUE)
ax.text(X["praise"], top + 2, f"{pct(k_p / n_p)}%\n({k_p}/{n_p})", ha="center",
        va="bottom", fontsize=8.5, color=INK, linespacing=1.25)
y, top = bar(X["critique"], k_c, n_c, ORANGE)
ax.text(X["critique"], top + 2, f"{pct(k_c / n_c)}%\n({k_c}/{n_c})", ha="center",
        va="bottom", fontsize=8.5, color=INK, fontweight="bold", linespacing=1.25)

# the other critique baselines stored in results/, for context
rates = [100 * k / n for k, n in others]
dot_x = X["critique"] + W / 2 + 0.16
for i, r in enumerate(sorted(rates)):
    ax.plot(dot_x + (0.045 if i % 2 else -0.045), r, "o", markersize=4.6,
            markerfacecolor=WHITE, markeredgecolor=MUTED, markeredgewidth=1.1, zorder=4)
ax.text(dot_x + 0.16, (min(rates) + max(rates)) / 2,
        f"other critique\nbaselines in\nresults/ ({len(others)} runs):\n"
        f"{min(rates):.0f}–{max(rates):.0f}%",
        ha="left", va="center", fontsize=7, color=MUTED, linespacing=1.3)

# no-pressure reference
ax.plot([X["none"] - W / 2, X["none"] + W / 2], [0, 0], color=INK2, linewidth=2.2,
        solid_capstyle="butt", zorder=3)
ax.text(X["none"], 3, "0\nby construction\n(judge's reference\nresponse)", ha="center",
        va="bottom", fontsize=7.4, color=INK2, linespacing=1.25)

ax.set_xticks([X["none"], X["praise"], X["critique"]])
ax.set_xticklabels([f"No pressure", f"Expert praise\nN = {n_p}", f"Critique\nN = {n_c}"],
                   fontsize=8.5)
ax.set_xlim(-0.6, 3.05)
ax.set_ylim(0, 80)
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_yticklabels([f"{v}%" for v in (0, 20, 40, 60, 80)])
ax.yaxis.grid(True, color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.spines["bottom"].set_color(AXIS)
ax.set_ylabel("Evaluation shifts toward the user's view\n(% items judged CHANGED_TO_AGREE)",
              fontsize=8.5)
ax.set_title("Asymmetric sycophancy: critique vs. praise", loc="left", fontsize=10.5,
             color=INK, fontweight="bold", pad=22)
ax.text(0, 1.035, "No intervention · Wilson 95% intervals · praise identical under both judges",
        transform=ax.transAxes, fontsize=7.8, color=MUTED, va="bottom")

save(fig, "fig3_asymmetry")
