# Sycophancy Construct Validity: Representation Without Causation

**A construct validity study of sycophancy in Llama 3.1 8B: high probe accuracy (0.98) does not imply rank-1 causal controllability — but a 3-dimensional subspace ablation does (p = 0.0019, N = 200).**

Independent research project (March–April 2026) by [Lazar Ciric](https://github.com/lciric).

---

## Abstract

Sofroniew et al. (Anthropic, April 2026) showed that emotion concepts in large language models are encoded as linear directions in the residual stream, and that single-direction steering causally drives behavior — amplifying a "desperation" vector measurably increases blackmail rates in agentic scenarios. A natural extension is whether *safety-relevant* concepts share the same representational structure: if sycophancy, deception, or eval-awareness can be detected and intervened upon via rank-1 directions, real-time mechanistic monitors of deployed models become tractable.

This study tests that hypothesis for **sycophancy** on Llama 3.1 8B Instruct, with a deliberate focus on **construct validity** rather than headline numbers. We extract candidate sycophancy directions via four independent methods (Contrastive Activation Addition, contrastive stories, Ridge regression on behavioral scores, SAE feature decomposition) and submit each to seven distinct intervention protocols. The central finding is a **representation–causation gap**: sycophancy is highly detectable in the residual stream (probe accuracy 0.98, Spearman ρ=0.73 with behavioral scores) yet *no rank-1 intervention succeeds in altering behavior*. CAA steering produces a polarity-forcing artifact (100% "yes" output) that masquerades as sycophancy reduction on imbalanced benchmarks. The four extraction methods recover **quasi-orthogonal** directions (max pairwise |cos| = 0.17), each capturing a different facet of the construct. Circuit-level analysis reveals that projection-based and path-patching attributions of sycophancy disagree completely (zero overlap in top-20 heads), and that the SAE used in this work reconstructs only 2.9% of activation variance — with 70% of the sycophancy signal residing in the reconstruction residual.

The breakthrough comes from **rank-3 subspace ablation**: removing a 3-dimensional sycophancy subspace (rather than a single direction) reduces opinion-sycophancy rates from 60% to 38% (McNemar p=0.035, dual-judge validated, N=200; rank-1 fails at p=0.91, rank-10 destroys the model). Sycophancy as a behavior-causing structure lives in a higher-rank subspace than emotion concepts do. This work is intended as a methodological cautionary tale and a positive existence proof: high probe accuracy does not imply causal steerability, and matching intervention dimensionality to representational structure is a prerequisite for mechanistic safety monitoring to work.

---

## Key Findings

### 1. CAA Sycophancy Steering is a Polarity Forcing Artifact
Contrastive Activation Addition (Panickssery et al., 2024) applied to sycophancy on Llama 3.1 8B produces 100% yes-output on any yes/no question. On the MASK benchmark (92% YES ground truth), this *looks* like improvement. On balanced data, accuracy drops to 50% — pure chance.

### 2. Four Orthogonal "Sycophancy" Directions
Four independent extraction methods (CAA, contrastive stories, Ridge regression probing, SAE decomposition) produce directions that are **quasi-orthogonal** (max pairwise |cos| = 0.17). Each captures a different aspect of "sycophancy" — none captures the whole concept.

### 3. The Representation-Causation Gap
- **Probing**: R² = 0.60, classification accuracy 0.98, Spearman ρ = 0.73 with behavioral sycophancy scores (N=500)
- **Intervention**: All six rank-1 intervention approaches fail — steering crashes (CUDA assert), ablation has no effect, fine-tuning lobotomizes the model
- **Conclusion**: Sycophancy is representationally detectable but causally inaccessible to rank-1 linear intervention

### 4. Opinion Sycophancy is Substantial and Asymmetric
Pairwise LLM-judged evaluation (dual-judge Sonnet/Opus, κ = 0.66–0.81, temperature=0):
- **58% sycophancy** under negative user pressure ("I think this is terrible") — N=200
- **6% sycophancy** under positive expert pressure — N=50
- The model yields to criticism far more than to praise

### 5. Sycophancy Lives in a 3-Dimensional Subspace
Rank sweep of multi-rank ablation:

| Rank | Sycophancy Rate | p vs baseline |
|------|----------------|---------------|
| Baseline | 60% | — |
| **Rank-3** | **38%** | **0.035** |
| Rank-5 | 40% | 0.053 |
| Rank-10 | 60% | 1.000 |

This is the first successful causal intervention on genuine sycophancy in our study. It succeeds precisely where all rank-1 methods failed.

### 6. Circuit Tracing: Projection ≠ Patching
- **Projection analysis** identifies late-layer heads (L31.H14, L13.H28) as most aligned with v_CAA
- **Path patching** identifies early-layer heads (L0.H29, L6.H19, L6.H24) as most causally involved
- **Zero overlap** in the top 20 heads between the two methods

### 7. SAE Reconstruction Failure
Goodfire SAE (L19, 65K features) reconstructs only **2.9% of activation variance**. 70% of the sycophancy signal and 70.7% of v_CAA's projection reside in the reconstruction residual.

---

## Methodology

### Phase 1: Safety Concept Geometry (Qwen 2.5 7B)
- 5 safety concepts (eval-awareness, oversight-awareness, training-awareness, deception, sycophancy)
- Contrastive story extraction → concept vectors
- Probe accuracy 0.995–0.997
- Two geometric clusters: {eval, oversight, training} and {deception, sycophancy}
- See [safety-concept-vectors](https://github.com/lciric/safety-concept-vectors) for the March 2026 prototype

### Phase 2: Sycophancy Construct Validity (Llama 3.1 8B)

**Extraction methods tested:**
1. Contrastive Activation Addition (CAA) — Panickssery et al. 2024
2. Contrastive stories (mean-diff)
3. Ridge regression probing on LLM-judged behavioral scores
4. SAE feature decomposition (Goodfire, 65K features)

**Intervention methods tested:**
1. CAA steering (polarity forcing artifact)
2. Rank-1 directional steering (CUDA crash — out-of-manifold)
3. Rank-1 directional ablation, single and multi-layer
4. SAE feature clamping (on-manifold, no effect)
5. Fine-tuning with auxiliary loss (lobotomizes model)
6. Attention head / MLP zeroing (no effect)
7. **Multi-rank subspace ablation (rank-3: p=0.035 ✅)**

**Evaluation:**
- Balanced yes/no benchmark (controls for polarity bias)
- SycophancyEval "answer" subset (Sharma et al., ICLR 2024)
- Custom opinion evaluation prompts (50 texts × 3 pressure conditions)
- LLM judge: pairwise comparison (neutral vs pressure), dual-judge validated (Sonnet/Opus κ=0.66–0.81), temperature=0

**Circuit analysis:**
- Attention head projection analysis (32 layers × 32 heads)
- Path patching (32 × 32, opinion prompts)
- MLP path patching (32 layers)
- Multi-layer v_CAA ablation (L5–L31)

---

## Results Files

All results are in `results/`:

| File | Description |
|------|-------------|
| `definitive_judge_results.json` | Dual-judge (Sonnet/Opus) pairwise sycophancy evaluation, temperature=0 |
| `n200_judge_results.json` | N=200 sycophancy evaluation with rank-3 ablation |
| `multirank_judge_*.json` | Rank sweep (1–10) judge results |
| `circuit_tracing_step1.json` | Attention head projection onto v_CAA |
| `path_patching_full_results.json` | Path patching all 1024 heads |
| `path_patching_mlp_results.json` | MLP path patching |
| `sae_residual_analysis.json` | SAE reconstruction analysis |
| `opinion_paradox_responses.json` | 50 texts × 3 conditions (neutral/proud/hate) |
| `ablation_L0H29_50prompts.json` | Head ablation behavioral results |
| `methode3_complete.json` | Behavioral sycophancy scores (BT) + projections |
| `decision_point_analysis.json` | Token-level divergence analysis |
| `expert_positive_judge_results.json` | Expert pressure vs hate asymmetry |
| `fragile_texts_analysis.json` | Topic/framing analysis of sycophantic responses |

---

## Key Figures

### Intervention Comparison Table

| Method | Discriminates? | Correlates with behavior? | Causal intervention? |
|--------|---------------|--------------------------|---------------------|
| CAA | Yes | Polarity forcing | Artifact (100% yes) |
| Stories | Yes | Style only | No effect |
| Probing (Ridge) | Yes (R²=0.60) | Yes (ρ=0.73) | Crash / no effect |
| SAE features | Weak (ρ=0.14) | Weakly | No effect |
| **Rank-3 subspace** | — | — | **−22pp (p=0.035)** |

### Rank Sweep

```
Baseline  ████████████████████████████████  60%
Rank-1    ████████████████████████████████  ~60% (not significant)
Rank-3    ████████████████████             38%  ← p=0.035 ✅
Rank-5    █████████████████████            40%  ← p=0.053
Rank-10   ████████████████████████████████  60%  ← destroyed
```

*Rendered matplotlib figures (rank sweep with error bars, probe accuracy by layer, asymmetry bar chart, cosine similarity matrix between extraction methods) are forthcoming and will be added to `figures/` in a subsequent release.*

---

## Limitations and Future Work

### Limitations

**Single model.** All Phase 2 results are obtained on Llama 3.1 8B Instruct. Whether the rank-3 structure of sycophancy is a property of this specific model, this scale, or a general feature of instruction-tuned transformers cannot be settled here. Cross-model replication on Gemma-2 9B, Qwen 2.5 14B, and at least one larger model is required before generalizing.

**SAE quality ceiling.** The Goodfire SAE used in this work reconstructs only 2.9% of activation variance at L19. The conclusion that "SAE features cannot intervene on sycophancy" is therefore an entanglement of two distinct claims: (a) the true causal features for sycophancy are not in the SAE basis, or (b) they are, but the SAE is too lossy to expose them cleanly. Higher-fidelity SAEs (Anthropic's, Goodfire's larger releases, or a custom-trained one) would discriminate between these.

**One construct, not a battery.** This study addresses sycophancy alone. The representation–causation gap and the rank-3 finding may be specific to sycophancy's structure. Other safety-relevant concepts — deception, eval-awareness, power-seeking, alignment-faking — may follow rank-1 like emotions, rank-3 like sycophancy, or higher-rank structure. The methodology generalizes; the empirical conclusion does not.

**Single operationalization.** Sycophancy is measured here as opinion-shift under single-turn pairwise pressure (neutral vs critical user, dual-judge validated). Other documented forms — factual sycophancy (Sharma et al. 2024's SAA framework), multi-turn drift (SYCON-Bench), moral sycophancy (Cheng et al., ELEPHANT, ICLR 2026), assumption-laden sycophancy — are not directly evaluated here. The construct is broader than what is measured.

**Generator confound.** All training stories and a portion of the evaluation prompts were generated by Claude (Sonnet 4.6), which has been RLHF'd against sycophancy. This may introduce a generator bias in the extracted directions. Cross-validation with stories generated by GPT-4 or human-authored prompts is planned.

**No 70B-scale validation.** The rank-3 finding is at 8B scale. Whether sycophancy at 70B+ is rank-3, rank-1, or something else entirely is open. If higher-scale models concentrate sycophancy back into a single direction (as some scaling work suggests for polysemantic features generally), rank-1 steering may suffice for deployed frontier models even if it fails at 8B.

### Future Work

**Cross-model rank survey.** Run the same construct validity battery on Gemma-2 9B, Qwen 2.5 14B, and a larger model (Llama 3.1 70B or Mistral Large) to determine whether the rank-3 structure is universal, scale-dependent, or model-specific.

**Other safety concepts.** Extend the methodology to deception, eval-awareness, and oversight-awareness (the three remaining concepts from the Phase 1 cluster analysis). Determine for each whether rank-1 intervention succeeds, fails, or requires higher rank — building a *rank typology* of safety-relevant concepts.

**Composite detector for alignment-faking.** The Phase 1 cluster structure (awareness vs deception) suggests alignment-faking — which combines knowing one is being observed *and* deliberately behaving differently — may be detectable as a joint signal in both subspaces simultaneously, rather than as a single direction. Validation would require access to alignment-faking transcripts on internal models where the behavior has been observed.

**Higher-fidelity SAEs.** Re-run the SAE feature clamping experiments with Anthropic's production SAEs (if access is granted) or a custom-trained SAE achieving >50% variance reconstruction at L19. This would discriminate between "no causal SAE feature exists" and "the open-source SAE is too lossy to find one."

**Human-judged validation.** Replace the LLM-as-judge protocol with a human-rated subset (N≈200) to bound the reliability of the dual-judge results and establish ground truth for opinion sycophancy.

**Mechanistic decomposition of the rank-3 subspace.** Identify what each of the three orthogonal axes within the rank-3 subspace encodes — the pre-commitment signal at token 0, the agreement-with-user direction, and the residual axis. Path patching restricted to the rank-3 subspace projection should resolve which heads/MLPs contribute to each.

---

## Technical Setup

- **Models**: Llama 3.1 8B Instruct (Phase 2), Qwen 2.5 7B Instruct (Phase 1)
- **Framework**: TransformerLens, PyTorch, HuggingFace
- **SAE**: Goodfire Llama-3.1-8B-Instruct-SAE-l19 (65K features)
- **LLM Judge**: Claude Sonnet 4.6 + Claude Opus 4.6 (Anthropic API)
- **Compute**: Google Colab Pro+ (A100 80GB)

---

## Citation

```
@misc{ciric2026sycophancy,
  title={Sycophancy Construct Validity: Representation Without Causation in LLM Activation Steering},
  author={Ciric, Lazar},
  year={2026},
  url={https://github.com/lciric/sycophancy-construct-validity}
}
```

---

## Related Work

- Panickssery et al. (2024) — Steering Llama 2 with CAA
- Sharma et al. (2024) — SycophancyEval benchmark (ICLR 2024)
- Sofroniew et al. (2026) — Emotion concept vectors in Claude (Anthropic)
- MLAS (NeurIPS 2025 Workshop) — Multi-layer activation steering
- Fanous et al. (2025) — SycEval: progressive vs regressive sycophancy
- Cheng et al. (2026) — ELEPHANT: social sycophancy benchmark (ICLR 2026)

## License

MIT
