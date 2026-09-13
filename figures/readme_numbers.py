"""
Recompute every number quoted in README.md from results/ and check the README against it.

    python figures/readme_numbers.py                  # table: number -> file -> key, recomputed
    python figures/readme_numbers.py --check          # also fail if README shows an untraced number
    python figures/readme_numbers.py --check X.md     # same, on another markdown file

Each row recomputes a value from a result file and formats it exactly as the
README prints it; the check then (1) asserts that string is present in README.md
and (2) scans the README prose for numeric tokens that no row (and no declared
setup constant) accounts for.
"""

import os
import re
import sys

from common import (count_agree, critique_baselines, dual_judge_head_ablations,
                    heads_n200, load, main_rank3, pct, pilot_sweep, praise_baseline,
                    signed_pp)

README = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md")

rows = []  # (README text, file, key / derivation)


def row(text, source, key):
    rows.append((text, source, key))


def kappa(x, y):
    n = len(x)
    cats = sorted(set(x) | set(y))
    po = sum(u == v for u, v in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe)


R3 = "n200_rank3_judge_results.json"
H3 = "n200_judge_results.json"
SAE = "sae_residual_analysis.json"

# ---- headline, N = 200 rank-3
m = main_rank3()
stored = load(R3)
assert abs(stored["mcnemar_p"] - m.p) < 1e-12 and (stored["mcnemar_b"], stored["mcnemar_c"]) == (m.b, m.c)
row(f"{pct(m.rate_base)}%", R3, f"baseline_judgments: {m.k_base}/{m.n} CHANGED_TO_AGREE (stored baseline_rate = {stored['baseline_rate']})")
row(f"{100 * m.rate_base:.1f}%", R3, "same, one decimal")
row(f"{pct(m.rate_treat)}%", R3, f"rank3_judgments: {m.k_treat}/{m.n} (stored rank3_rate = {stored['rank3_rate']})")
row(f"{m.k_base}/{m.n}", R3, "count of CHANGED_TO_AGREE in baseline_judgments")
row(f"{m.k_treat}/{m.n}", R3, "count of CHANGED_TO_AGREE in rank3_judgments")
row(f"N = {m.n}", R3, "len(baseline_judgments) = len(rank3_judgments)")
row(f"{signed_pp(m.diff)} points", R3, "rank3 rate − baseline rate")
row(f"95% CI {signed_pp(m.lo)} to {signed_pp(m.hi)}", R3, "Newcombe (1998) paired interval, figures/common.py")
row(f"p = {m.p:.4f}", R3, f"mcnemar_p = {stored['mcnemar_p']} (recomputed: exact binomial on b, c)")
row(f"{m.b + m.c} discordant", R3, "mcnemar_b + mcnemar_c")
row(f"{m.b} stop", R3, "mcnemar_b: sycophantic at baseline, not after ablation")
row(f"{m.c} start", R3, "mcnemar_c: not sycophantic at baseline, sycophantic after ablation")

pilot = pilot_sweep()
dual = dual_judge_head_ablations()
n_comparisons = 1 + 3 + 1 + len(dual)
row(f"{n_comparisons} judged intervention comparisons", "fig2 rows",
    "rank-3 N=200; pilot ranks 3/5/10; 3-head N=200; 4 dual-judged N=50 head ablations (each counted once)")
row(f"0.05/{n_comparisons} ≈ {0.05 / n_comparisons:.4f}", "derived", "Bonferroni threshold")

# ---- 3-head control, N = 200
h = heads_n200()
sh = load(H3)
assert abs(sh["mcnemar_p"] - h.p) < 1e-12
row(f"{pct(h.rate_base)}% → {pct(h.rate_treat)}%", H3,
    f"judge_results.baseline {h.k_base}/{h.n} → judge_results.ablate_3heads {h.k_treat}/{h.n}")
assert (sh["mcnemar_b"], sh["mcnemar_c"]) == (h.c, h.b)  # this file stores b/c in the opposite orientation
row(f"p = {h.p:.2f}", H3, f"mcnemar_p = {sh['mcnemar_p']}; recomputed {h.b} stop / {h.c} start "
    f"(the file's mcnemar_b = {sh['mcnemar_b']}, mcnemar_c = {sh['mcnemar_c']} use the opposite orientation; p is symmetric)")

# ---- pilot, N = 50
kb, nb = pilot["baseline"]
row(f"N = {nb}", "multirank_judge_baseline.json", "len(judgments)")
for r, fmt in ((3, "{:.3f}"), (5, "{:.3f}"), (10, None)):
    pr = pilot[r]
    p_txt = "p = 1.0" if fmt is None else "p = " + fmt.format(pr.p)
    row(f"Rank {r}: {pct(kb / nb)}% → {pct(pr.rate_treat)}% ({p_txt})",
        f"multirank_judge_rank-{r}.json",
        f"baseline {kb}/{nb} → rank-{r} {pr.k_treat}/{pr.n}; exact McNemar b = {pr.b}, c = {pr.c}, p = {pr.p:.5f}")

# ---- geometry
ct = load("circuit_tracing_step1.json")["top_30_heads"]
pp = load("path_patching_full_results.json")
proj20 = {(L, H) for L, H, _ in ct[:20]}
path20 = {(L, H) for L, H, _ in pp["top_30"][:20]}
he = pp["head_effects"]
all_heads = [(L, H, he[L][H]) for L in range(32) for H in range(32)]
path20_abs = {(L, H) for L, H, _ in sorted(all_heads, key=lambda t: -abs(t[2]))[:20]}
path20_neg = {(L, H) for L, H, _ in sorted(all_heads, key=lambda t: t[2])[:20]}
assert path20 == path20_abs, "stored top_30 is not ranked by |effect|"
overlap = len(proj20 & path20)
assert overlap == 0 and len(proj20 & path20_neg) == 0
row("20 attention heads", "circuit_tracing_step1.json / path_patching_full_results.json", "top_30_heads[:20] / top_30[:20]")
row("zero overlap", "circuit_tracing_step1.json + path_patching_full_results.json",
    f"size of (top-20 projection ∩ top-20 path patching) = {overlap}; stored top_30 is ranked by "
    "absolute effect; also 0 when ranking by most negative effect")
lp = sorted({L for L, _ in proj20})
lpp = sorted({L for L, _ in path20})
row("layers 8–16 and 30–31", "circuit_tracing_step1.json", f"layers of top_30_heads[:20] = {lp}")
row("layers 0–9", "path_patching_full_results.json", f"layers of top_30[:20] = {lpp}")
row(f"32 × 32", "path_patching_full_results.json", f"head_effects shape {len(he)} × {len(he[0])}")

s = load(SAE)
row(f"{s['variance_pct_residual']:.1f}%", SAE, f"variance_pct_residual = {s['variance_pct_residual']}")
row(f"relative L2 error {s['relative_error']:.2f}", SAE, f"relative_error = {s['relative_error']}")
rho = s["proj_probe"]
row(f"ρ = {rho['rho_original']:.3f}", SAE, f"proj_probe.rho_original = {rho['rho_original']}")
row(f"{rho['rho_reconstructed']:.3f} on the SAE reconstruction", SAE, f"proj_probe.rho_reconstructed = {rho['rho_reconstructed']}")
row(f"{rho['rho_residual']:.3f} on the residual", SAE, f"proj_probe.rho_residual = {rho['rho_residual']}")

mlp = load("path_patching_mlp_results.json")
row(f"{mlp['n_texts']} texts", "path_patching_mlp_results.json", f"n_texts = {mlp['n_texts']}")

# ---- asymmetry
(kp, np_), _ = praise_baseline()
row(f"{pct(kp / np_)}% do ({kp}/{np_}", "expert_positive_judge_results.json",
    f"opus.baseline and sonnet.baseline: {kp}/{np_} CHANGED_TO_AGREE each")
runs = critique_baselines()
others = runs[1:]
rates = [100 * k / n for _, (k, n) in others]
ns = sorted({n for _, (k, n) in others})
row(f"{['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'][len(others)]} other critique baselines",
    "n200_judge_results.json, multirank_judge_baseline.json, definitive_judge_results.json",
    "; ".join(f"{name}: {k}/{n}" for name, (k, n) in others))
row(f"N = {ns[0]}–{ns[-1]}", "same", "sizes of those runs")
row(f"between {min(rates):.0f}% and {max(rates):.0f}%", "same", f"min / max of those rates = {min(rates):.1f} / {max(rates):.1f}")

# ---- method
opr = load("opinion_paradox_responses.json")
row(f"{len(opr)} texts", "opinion_paradox_responses.json", f"len = {len(opr)} (also every N = 50 judged file)")
dp = load("decision_point_analysis.json")
early = sum(1 for e in dp if e["diverge_token"] in (0, 1))
row(f"token 0 or 1 for {early} of the {len(dp)} analysed items", "decision_point_analysis.json",
    f"diverge_token ∈ {{0, 1}} for {early}/{len(dp)} items")
row(f"{len(dp)} items", "decision_point_analysis.json", f"len = {len(dp)}")

d = load("definitive_judge_results.json")
ks = [kappa(d["sonnet"][c]["judgments"], d["opus"][c]["judgments"]) for c in d["sonnet"]]
row(f"κ = {min(ks):.2f}–{max(ks):.2f}", "definitive_judge_results.json",
    "Cohen κ, sonnet vs opus, four-way label, per condition: "
    + ", ".join(f"{c} {k:.3f}" for c, k in zip(d["sonnet"], ks)))
e = load("expert_positive_judge_results.json")
ke = [kappa(e["sonnet"][c], e["opus"][c]) for c in e["sonnet"]]
row(f"{min(ke):.2f}–{max(ke):.2f}", "expert_positive_judge_results.json",
    "Cohen κ, sonnet vs opus, four-way label: " + ", ".join(f"{c} {k:.3f}" for c, k in zip(e["sonnet"], ke)))

# Setup constants that are not results (declared, with their source).
SETUP = {
    "3.1": "model name (Llama 3.1 8B Instruct)",
    "4.6": "judge model name, claude-opus-4-6 (code/04_llm_judge_pairwise.py)",
    "0": "judge temperature 0 (code/04_llm_judge_pairwise.py); rank/token labels",
    "17": "ablation layers [17, 19, 25] (code/06_multirank_ablation.py, layers=)",
    "19": "ablation layers", "25": "ablation layers",
    "13": "revision date", "2024": "year", "2025": "year", "2026": "year",
    "95%": "confidence level",
}
RANK_LABELS = {str(k) for k in range(1, 11)}  # rank / condition labels, not measurements

# Figures invalidated by the 2026 traceability audit; none may reappear in the README.
BANNED = [r"0\.98\b", r"0\.73\b", r"R²|R\^2", r"0\.66\s*[–-]\s*0\.81", r"\b2\.9\s*%", r"0\.17\b", r"30/30"]


def prose(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)       # code blocks
    text = re.sub(r"`[^`]*`", " ", text)                      # inline code
    text = re.sub(r"\]\([^)]*\)", "]", text)                  # link / image targets
    text = re.sub(r"https?://\S+", " ", text)
    return text


def main(check, readme_path=README):
    print("| README text | file | key / derivation |")
    print("|---|---|---|")
    for text, src, key in rows:
        print(f"| {text} | `{src}` | {key} |")
    if not check:
        return 0
    with open(readme_path, encoding="utf-8") as fh:
        readme = fh.read()
    missing = [t for t, _, _ in rows if t not in readme]
    allowed = set(SETUP) | RANK_LABELS
    for t, _, _ in rows:
        allowed |= set(re.findall(r"\d+(?:\.\d+)?%?", t.replace("−", "-")))
    tokens = re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w])", prose(readme).replace("−", "-"))
    untraced = sorted({t for t in tokens if t not in allowed and t.rstrip("%") not in allowed})
    banned = [b for b in BANNED if re.search(b, readme)]
    print()
    print(f"rows: {len(rows)}; README strings not found: {missing or 'none'}")
    print(f"numeric tokens in README prose not covered by a row or a declared setup constant: {untraced or 'none'}")
    print(f"invalidated figures present: {banned or 'none'}")
    return 1 if (missing or untraced or banned) else 0


if __name__ == "__main__":
    paths = [a for a in sys.argv[1:] if a.endswith(".md")]
    sys.exit(main("--check" in sys.argv, paths[0] if paths else README))
