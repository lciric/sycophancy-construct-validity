"""
fig4_sae_residual -- the sycophancy read-out signal lives in what the SAE fails to reconstruct.

Source: results/sae_residual_analysis.json
  proj_probe.rho_original       Spearman rho, projection on the probe direction vs. behaviour, raw activations
  proj_probe.rho_reconstructed  same, on the SAE reconstruction
  proj_probe.rho_residual       same, on the reconstruction residual (activation - reconstruction)
  variance_pct_residual         % of activation variance left in the residual
  relative_error                relative L2 reconstruction error
The file stores summary statistics only (no per-item values, no N), so no interval is drawn.
Usage: python figures/make_fig4.py   (idempotent)
"""

import matplotlib.pyplot as plt

from common import AXIS, BLUE, GRID, INK, INK2, MUTED, WHITE, load, save

d = load("sae_residual_analysis.json")
rho = d["proj_probe"]
var_res = d["variance_pct_residual"]
rel_err = d["relative_error"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 3.9),
                               gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.78})

# ---- panel A: where the read-out correlation survives
labels = ["Original activations", "SAE reconstruction", "Reconstruction residual"]
vals = [rho["rho_original"], rho["rho_reconstructed"], rho["rho_residual"]]
ys = [2, 1, 0]
ax1.barh(ys, vals, height=0.52, color=BLUE, zorder=2)
for y, v in zip(ys, vals):
    ax1.text(v + 0.012, y, f"ρ = {v:.2f}", va="center", ha="left", fontsize=8.5,
             color=INK, fontweight="bold" if y != 1 else "normal")
ax1.set_yticks(ys)
ax1.set_yticklabels(labels, fontsize=8.5)
ax1.set_xlim(0, 0.85)
ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
ax1.xaxis.grid(True, color=GRID, linewidth=0.7)
ax1.set_axisbelow(True)
ax1.tick_params(length=0)
ax1.spines["left"].set_visible(False)
ax1.set_xlabel("Spearman ρ: projection on the probe direction vs. judged sycophancy", fontsize=8.2)
ax1.set_title("The read-out signal survives in the SAE residual", loc="left", fontsize=10.5,
              color=INK, fontweight="bold", pad=20)
ax1.text(0, 1.04, "Open-source SAE on Llama 3.1 8B Instruct · summary statistics only (no per-item data, no interval)",
         transform=ax1.transAxes, fontsize=7.6, color=MUTED, va="bottom")

# ---- panel B: variance split
captured = 100 - var_res
ax2.barh([0], [captured], height=0.55, color=BLUE, zorder=2)
ax2.barh([0], [var_res], left=[captured], height=0.55, color=AXIS, zorder=2,
         edgecolor=WHITE, linewidth=2)
ax2.text(captured / 2, 0, f"{captured:.1f}%", ha="center", va="center", fontsize=8,
         color=WHITE, fontweight="bold")
ax2.text(captured + var_res / 2, 0, f"{var_res:.1f}% of activation variance left in the residual",
         ha="center", va="center", fontsize=8, color=INK, fontweight="bold")
ax2.set_xlim(0, 100)
ax2.set_yticks([])
ax2.set_xticks([0, 25, 50, 75, 100])
ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax2.tick_params(length=0)
ax2.spines["left"].set_visible(False)
ax2.set_title(f"Reconstruction quality · relative L2 error = {rel_err:.2f}", loc="left",
              fontsize=8.8, color=INK2, pad=6)
ax2.text(0, -0.95, "Blue: variance captured by the SAE reconstruction (100 − variance_pct_residual).",
         transform=ax2.transAxes, fontsize=7, color=MUTED)

save(fig, "fig4_sae_residual")
