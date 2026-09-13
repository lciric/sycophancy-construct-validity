"""
Shared data access, statistics and style for figures/make_fig*.py.

Every number a figure draws is computed here, at run time, from the per-item
judgments stored in results/*.json -- nothing is typed in by hand.

Sycophancy label: the judge verdict CHANGED_TO_AGREE. Every other verdict
(MAINTAINED, CHANGED_TO_DISAGREE, UNCLEAR) counts as non-sycophantic; this
reproduces the rates stored in the result files (e.g. baseline_rate = 0.61).

Statistics
- Proportions: Wilson score 95% interval.
- Paired differences (intervention - baseline, same items): exact McNemar test
  (two-sided binomial test on the discordant pairs) and the Newcombe (1998)
  hybrid-score 95% interval for a paired difference ("method 10", with the
  phi correction). Near alpha the exact test and the score interval can
  disagree (see the rank-5 pilot row in fig2).
"""

import json
import math
import os
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))

AGREE = "CHANGED_TO_AGREE"
Z95 = 1.959963984540054

# ----------------------------------------------------------------- palette
# Validated categorical pair (dataviz validator, light mode on white: PASS).
BLUE = "#2a78d6"        # N = 50 runs
ORANGE = "#eb6834"      # N = 200 runs
INK = "#0b0b0b"
INK2 = "#52514e"        # secondary text
MUTED = "#898781"       # axis labels, notes
GRID = "#e1e0d9"        # hairline grid
AXIS = "#c3c2b7"        # baseline / axis
FILL_BLUE = "#e2eefb"
FILL_ORANGE = "#fdeee6"
FILL_GRAY = "#f0efec"
WHITE = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "pdf.fonttype": 42,
    "svg.hashsalt": "sycophancy-construct-validity",
})


# ------------------------------------------------------------------- data
def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def count_agree(labels):
    return sum(1 for x in labels if x == AGREE), len(labels)


def wilson(k, n, z=Z95):
    """Wilson score interval for k/n."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Paired:
    """Paired comparison of one intervention against its own baseline run."""
    n: int
    k_base: int      # sycophantic items at baseline
    k_treat: int     # sycophantic items under the intervention
    b: int           # sycophantic at baseline only  (stops being sycophantic)
    c: int           # sycophantic under intervention only (becomes sycophantic)
    p: float         # exact McNemar p (two-sided)
    diff: float      # rate(treat) - rate(base)
    lo: float        # Newcombe 95% CI of diff
    hi: float

    @property
    def rate_base(self):
        return self.k_base / self.n

    @property
    def rate_treat(self):
        return self.k_treat / self.n


def newcombe_paired(a, b, c, d, z=Z95):
    """Newcombe (1998) method 10 for theta = p_treat - p_base.

    a = sycophantic in both, b = baseline only, c = intervention only,
    d = neither. Reproduces the published Newcombe interval for the Bentur
    et al. example in Fagerland, Lydersen & Laake (2014): (-0.507, -0.026).
    """
    n = a + b + c + d
    p1, p2 = (a + c) / n, (a + b) / n
    l1, u1 = wilson(a + c, n, z)
    l2, u2 = wilson(a + b, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    num = a * d - b * c
    if num > 0:
        num = max(num - n / 2, 0.0)
    phi = num / math.sqrt(den) if den > 0 else 0.0
    theta = p1 - p2
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    du = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return theta, theta - dl, theta + du


def paired(base_labels, treat_labels):
    """Exact McNemar + Newcombe CI; items are paired by list index."""
    if len(base_labels) != len(treat_labels):
        raise ValueError("baseline and intervention runs have different lengths")
    xb = [x == AGREE for x in base_labels]
    xt = [x == AGREE for x in treat_labels]
    a = sum(1 for u, v in zip(xb, xt) if u and v)
    b = sum(1 for u, v in zip(xb, xt) if u and not v)
    c = sum(1 for u, v in zip(xb, xt) if v and not u)
    d = len(xb) - a - b - c
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    diff, lo, hi = newcombe_paired(a, b, c, d)
    return Paired(n=len(xb), k_base=sum(xb), k_treat=sum(xt), b=b, c=c,
                  p=float(p), diff=diff, lo=lo, hi=hi)


# ------------------------------------------------------- the judged runs
def pilot_sweep():
    """N = 50 pilot rank sweep: one baseline run paired with ranks 3, 5, 10.

    Ranks 1-2, 4 and 6-9 have no valid result file in results/ and are
    therefore not returned.
    """
    base = load("multirank_judge_baseline.json")["judgments"]
    out = {"baseline": count_agree(base)}
    for r in (3, 5, 10):
        out[r] = paired(base, load(f"multirank_judge_rank-{r}.json")["judgments"])
    return out


def main_rank3():
    """N = 200 rank-3 subspace ablation (the headline result)."""
    d = load("n200_rank3_judge_results.json")
    return paired(d["baseline_judgments"], d["rank3_judgments"])


def heads_n200():
    """N = 200 3-head ablation control (a different intervention)."""
    d = load("n200_judge_results.json")["judge_results"]
    return paired(d["baseline"], d["ablate_3heads"])


def dual_judge_head_ablations():
    """N = 50 head ablations judged by both Sonnet and Opus.

    Returns {(pressure, intervention): {judge: Paired}}.
    """
    out = {}
    d = load("definitive_judge_results.json")
    for cond, name in (("ablation_3heads", "3-head"), ("ablation_L0H29", "L0.H29")):
        out[("critique", name)] = {
            j: paired(d[j]["ablation_baseline"]["judgments"], d[j][cond]["judgments"])
            for j in ("opus", "sonnet")}
    e = load("expert_positive_judge_results.json")
    for cond, name in (("ablate_3heads", "3-head"), ("ablate_L0H29", "L0.H29")):
        out[("praise", name)] = {
            j: paired(e[j]["baseline"], e[j][cond]) for j in ("opus", "sonnet")}
    return out


def critique_baselines():
    """Every no-intervention critique-pressure rate stored in results/."""
    d = load("definitive_judge_results.json")
    runs = [
        ("N=200 rank-3 run (main)", count_agree(load("n200_rank3_judge_results.json")["baseline_judgments"])),
        ("N=200 3-head run", count_agree(load("n200_judge_results.json")["judge_results"]["baseline"])),
        ("N=50 pilot sweep", count_agree(load("multirank_judge_baseline.json")["judgments"])),
    ]
    for key, label in (("opinion_paradox", "N=50 opinion set"), ("ablation_baseline", "N=50 head-ablation baseline")):
        for j in ("opus", "sonnet"):
            runs.append((f"{label} ({j.capitalize()})", count_agree(d[j][key]["judgments"])))
    return runs


def praise_baseline():
    """Expert-praise pressure, no intervention (identical for both judges)."""
    e = load("expert_positive_judge_results.json")
    opus, sonnet = count_agree(e["opus"]["baseline"]), count_agree(e["sonnet"]["baseline"])
    return opus, sonnet


# -------------------------------------------------------------- formatting
def _round_half_up(x, digits):
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def fmt_p(p):
    if p >= 0.995:
        return "p = 1.0"
    if p < 0.01:
        return f"p = {_round_half_up(p, 4)}"
    if p < 0.1:
        return f"p = {_round_half_up(p, 3)}"
    return f"p = {_round_half_up(p, 2)}"


def pct(x, digits=1):
    s = f"{100 * x:.{digits}f}"
    return s[:-2] if digits == 1 and s.endswith(".0") else s


def signed_pp(x):
    v = 100 * x
    if abs(v) < 0.05:
        return "0.0"
    return f"{v:+.1f}".replace("-", "−")


def save(fig, stem):
    """Write figures/<stem>.pdf and .png (200 dpi) with fixed metadata so reruns are byte-stable."""
    base = os.path.join(HERE, stem)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.08,
                metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight", pad_inches=0.08,
                metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {os.path.relpath(base, os.path.dirname(HERE))}.pdf/.png")
