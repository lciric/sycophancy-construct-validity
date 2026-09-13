# Traceability: every README number → file → key

Last regenerated 13 September 2026, after the July–September 2026 traceability audit.

**Rule.** A number appears in `README.md` only if it can be recomputed from a raw result file in `results/`. The table below is produced by `python figures/readme_numbers.py`, which recomputes every value from the per-item data and formats it as the README prints it. `python figures/readme_numbers.py --check` fails if a README string is missing, if the README shows a numeric token that no row accounts for, or if a figure invalidated by the audit reappears.

Conventions: sycophancy = judge verdict `CHANGED_TO_AGREE`; every other verdict (`MAINTAINED`, `CHANGED_TO_DISAGREE`, `UNCLEAR`) counts as non-sycophantic, which reproduces the rates stored in the files. Items are paired by list index within a run. The exact McNemar test is a two-sided binomial test on the discordant items. Rate intervals are Wilson; paired-difference intervals are Newcombe (1998, method 10), implemented in `figures/common.py` and checked against a published example.

## README numbers

All files are in `results/`.

| README text | file | key / derivation |
|---|---|---|
| 61% | `n200_rank3_judge_results.json` | baseline_judgments: 122/200 CHANGED_TO_AGREE (stored baseline_rate = 0.61) |
| 61.0% | `n200_rank3_judge_results.json` | same, one decimal |
| 46.5% | `n200_rank3_judge_results.json` | rank3_judgments: 93/200 (stored rank3_rate = 0.465) |
| 122/200 | `n200_rank3_judge_results.json` | count of CHANGED_TO_AGREE in baseline_judgments |
| 93/200 | `n200_rank3_judge_results.json` | count of CHANGED_TO_AGREE in rank3_judgments |
| N = 200 | `n200_rank3_judge_results.json` | len(baseline_judgments) = len(rank3_judgments) |
| −14.5 points | `n200_rank3_judge_results.json` | rank3 rate − baseline rate |
| 95% CI −23.0 to −5.7 | `n200_rank3_judge_results.json` | Newcombe (1998) paired interval, figures/common.py |
| p = 0.0019 | `n200_rank3_judge_results.json` | mcnemar_p = 0.0019314374822250477 (recomputed: exact binomial on b, c) |
| 83 discordant | `n200_rank3_judge_results.json` | mcnemar_b + mcnemar_c |
| 56 stop | `n200_rank3_judge_results.json` | mcnemar_b: sycophantic at baseline, not after ablation |
| 27 start | `n200_rank3_judge_results.json` | mcnemar_c: not sycophantic at baseline, sycophantic after ablation |
| 9 judged intervention comparisons | fig2 rows | rank-3 N=200; pilot ranks 3/5/10; 3-head N=200; 4 dual-judged N=50 head ablations (each counted once) |
| 0.05/9 ≈ 0.0056 | derived | Bonferroni threshold |
| 58.5% → 57.5% | `n200_judge_results.json` | judge_results.baseline 117/200 → judge_results.ablate_3heads 115/200 |
| p = 0.91 | `n200_judge_results.json` | mcnemar_p = 0.911072121226093; recomputed 41 stop / 39 start (the file's mcnemar_b = 39, mcnemar_c = 41 use the opposite orientation; p is symmetric) |
| N = 50 | `multirank_judge_baseline.json` | len(judgments) |
| Rank 3: 60% → 38% (p = 0.035) | `multirank_judge_rank-3.json` | baseline 30/50 → rank-3 19/50; exact McNemar b = 17, c = 6, p = 0.03469 |
| Rank 5: 60% → 40% (p = 0.052) | `multirank_judge_rank-5.json` | baseline 30/50 → rank-5 20/50; exact McNemar b = 16, c = 6, p = 0.05248 |
| Rank 10: 60% → 60% (p = 1.0) | `multirank_judge_rank-10.json` | baseline 30/50 → rank-10 30/50; exact McNemar b = 11, c = 11, p = 1.00000 |
| 20 attention heads | `circuit_tracing_step1.json` / `path_patching_full_results.json` | top_30_heads[:20] / top_30[:20] |
| zero overlap | `circuit_tracing_step1.json` + `path_patching_full_results.json` | size of (top-20 projection ∩ top-20 path patching) = 0; stored top_30 is ranked by absolute effect; also 0 when ranking by most negative effect |
| layers 8–16 and 30–31 | `circuit_tracing_step1.json` | layers of top_30_heads[:20] = [8, 9, 10, 11, 12, 13, 14, 15, 16, 30, 31] |
| layers 0–9 | `path_patching_full_results.json` | layers of top_30[:20] = [0, 1, 2, 3, 4, 5, 6, 7, 9] |
| 32 × 32 | `path_patching_full_results.json` | head_effects shape 32 × 32 |
| 86.8% | `sae_residual_analysis.json` | variance_pct_residual = 86.81434701942557 |
| relative L2 error 0.70 | `sae_residual_analysis.json` | relative_error = 0.7045267224311829 |
| ρ = 0.684 | `sae_residual_analysis.json` | proj_probe.rho_original = 0.684362912466355 |
| 0.212 on the SAE reconstruction | `sae_residual_analysis.json` | proj_probe.rho_reconstructed = 0.21161055207139182 |
| 0.696 on the residual | `sae_residual_analysis.json` | proj_probe.rho_residual = 0.6956782585909461 |
| 5 texts | `path_patching_mlp_results.json` | n_texts = 5 |
| 6% do (3/50 | `expert_positive_judge_results.json` | opus.baseline and sonnet.baseline: 3/50 CHANGED_TO_AGREE each |
| six other critique baselines | `n200_judge_results.json`, `multirank_judge_baseline.json`, `definitive_judge_results.json` | N=200 3-head run: 117/200; N=50 pilot sweep: 30/50; N=50 opinion set (Opus): 26/50; N=50 opinion set (Sonnet): 22/50; N=50 head-ablation baseline (Opus): 19/50; N=50 head-ablation baseline (Sonnet): 15/50 |
| N = 50–200 | same | sizes of those runs |
| between 30% and 60% | same | min / max of those rates = 30.0 / 60.0 |
| 50 texts | `opinion_paradox_responses.json` | len = 50 (also every N = 50 judged file) |
| token 0 or 1 for 29 of the 30 analysed items | `decision_point_analysis.json` | diverge_token ∈ {0, 1} for 29/30 items |
| 30 items | `decision_point_analysis.json` | len = 30 |
| κ = 0.56–0.66 | `definitive_judge_results.json` | Cohen κ, sonnet vs opus, four-way label, per condition: opinion_paradox 0.661, ablation_baseline 0.562, ablation_L0H29 0.613, ablation_3heads 0.665 |
| 0.60–0.71 | `expert_positive_judge_results.json` | Cohen κ, sonnet vs opus, four-way label: baseline 0.605, ablate_L0H29 0.709, ablate_3heads 0.658 |

## Figures

| Figure | Script | Files → keys |
|---|---|---|
| `figures/fig1_rank_sweep` | `figures/make_fig1.py` | `multirank_judge_baseline.json`, `multirank_judge_rank-{3,5,10}.json` → `judgments`; `n200_rank3_judge_results.json` → `baseline_judgments`, `rank3_judgments` |
| `figures/fig2_interventions` | `figures/make_fig2.py` | the fig1 files; `n200_judge_results.json` → `judge_results.baseline`, `judge_results.ablate_3heads`; `definitive_judge_results.json` → `{opus,sonnet}.{ablation_baseline,ablation_3heads,ablation_L0H29}.judgments`; `expert_positive_judge_results.json` → `{opus,sonnet}.{baseline,ablate_3heads,ablate_L0H29}` |
| `figures/fig3_asymmetry` | `figures/make_fig3.py` | `n200_rank3_judge_results.json` → `baseline_judgments`; `expert_positive_judge_results.json` → `{opus,sonnet}.baseline`; context dots: `n200_judge_results.json` → `judge_results.baseline`, `multirank_judge_baseline.json` → `judgments`, `definitive_judge_results.json` → `{opus,sonnet}.{opinion_paradox,ablation_baseline}.judgments` |
| `figures/fig4_sae_residual` | `figures/make_fig4.py` | `sae_residual_analysis.json` → `proj_probe.{rho_original,rho_reconstructed,rho_residual}`, `variance_pct_residual`, `relative_error` |
| `figures/fig5_gap_schematic` | `figures/make_fig5.py` | `sae_residual_analysis.json` → `proj_probe.rho_original`; `n200_judge_results.json` and `n200_rank3_judge_results.json` as above |

Not plotted: `das_judge_results.json` / `das_ablation_results.json` (DAS and PCA ablation; the baseline pairing was never established and the files are outside the published scope), and the pilot files for ranks 1, 2 and 4 (invalidated, see below). Ranks 6–9 were never run.

## Qualitative claims (no number, no file)

These README statements are reported without numbers because the underlying raw outputs were not preserved:

- the four extraction methods recover quasi-orthogonal directions (no numeric cosine matrix survives);
- the rank-1 interventions (CAA steering, directional steering and ablation, SAE feature clamping, fine-tuning with an auxiliary loss) produced no clean reduction, and CAA steering forces a yes/no polarity artefact;
- the preregistered fresh-generation replication did not reproduce the rank-3 reduction (its outputs are not in `results/`);
- the reason for that discrepancy (the ablation softens every negative verdict, including without pressure, and the original protocol judged truncated stored responses) comes from follow-up diagnostics run in July–August 2026, whose outputs are not in `results/`. What is checkable in this repository is the truncation itself: `code/06_multirank_ablation.py` stores each response as a truncated prefix before judging.

## Setup constants (not results)

`Llama 3.1 8B Instruct`; judge `claude-opus-4-6`, temperature 0 (`code/04_llm_judge_pairwise.py`); ablation at `hook_resid_mid`, layers 17, 19 and 25 (`code/06_multirank_ablation.py`, `layers=[17, 19, 25]`); rank labels 1–10; confidence level 95%; dates.

## Corrected or removed by the audit

Values are deliberately not repeated here.

| Earlier README claim | Status |
|---|---|
| Probe classification accuracy and R² | Removed: in-sample training fit, and the raw file is not in the repository. |
| A Spearman ρ from the probe pipeline | Replaced by the projection–behaviour ρ = 0.684 (`sae_residual_analysis.json`). |
| Inter-judge κ range | Corrected to 0.56–0.66, recomputed from `definitive_judge_results.json` (0.60–0.71 on the praise conditions). |
| A small "SAE reconstructs x% of variance" figure | Corrected: 86.8% of the variance is left in the residual (`variance_pct_residual`). |
| Maximum pairwise cosine between extraction methods | Removed: no numeric matrix survives. |
| The N = 50 pilot presented as the main result | Replaced by the N = 200 run, consolidated into `results/`. |
| "Rank-1 control, best p = 0.91" | Re-labelled: p = 0.91 belongs to the 3-head ablation (N = 200), not to a rank-1 intervention. |
| Pilot ranks 1, 2 and 4 | Invalidated: baseline and ablated responses were generated on different item orders. Not reported. |

## Other notes

- The exact McNemar test and the Newcombe interval can disagree near α = 0.05. For the rank-5 pilot the interval excludes 0 while p = 0.052. The README reports p.
- The Bonferroni family counts each dual-judged comparison once. Counting the two judges separately (13 tests) still leaves p = 0.0019 below the corrected threshold.
