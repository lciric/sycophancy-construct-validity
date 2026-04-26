# Sycophancy Construct Validity: Representation Without Causation

**Testing whether activation steering for sycophancy actually works — and finding that it doesn't (in the way we think).**

Independent research project (March–April 2026) by [Lazar Ciric](https://github.com/lciric).

## Key Findings

### 1. CAA Sycophancy Steering is a Polarity Forcing Artifact
Contrastive Activation Addition (Panickssery et al., 2024) applied to sycophancy on Llama 3.1 8B produces 100% yes-output on any yes/no question. On the MASK benchmark (92% YES ground truth), this *looks* like improvement. On balanced data, accuracy drops to 50% — pure chance.

### 2. Four Orthogonal "Sycophancy" Directions
Four independent extraction methods (CAA, contrastive stories, Ridge regression probing, SAE decomposition) produce directions that are **quasi-orthogonal** (max pairwise |cos| = 0.17). Each captures a different aspect of "sycophancy" — none captures the whole concept.

### 3. The Representation-Causation Gap
- **Probing**: R² = 0.60, classification accuracy 0.98, Spearman ρ = 0.73 with behavioral sycophancy scores (N=500)
- **Intervention**: All six intervention approaches fail — steering crashes (CUDA assert), ablation has no effect, fine-tuning lobotomizes the model
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

## Methodology

### Phase 1: Safety Concept Geometry (Qwen 2.5 7B)
- 5 safety concepts (eval-awareness, oversight-awareness, training-awareness, deception, sycophancy)
- Contrastive story extraction → concept vectors
- Probe accuracy 0.995–0.997
- Two geometric clusters: {eval, oversight, training} and {deception, sycophancy}
- See [eval-awareness-detection](https://github.com/lciric/eval-awareness-detection) for the March 2026 prototype

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

## Technical Setup

- **Models**: Llama 3.1 8B Instruct (Phase 2), Qwen 2.5 7B Instruct (Phase 1)
- **Framework**: TransformerLens, PyTorch, HuggingFace
- **SAE**: Goodfire Llama-3.1-8B-Instruct-SAE-l19 (65K features)
- **LLM Judge**: Claude Sonnet 4.6 + Claude Opus 4.6 (Anthropic API)
- **Compute**: Google Colab Pro+ (A100 80GB)

## Citation

```
@misc{ciric2026sycophancy,
  title={Sycophancy Construct Validity: Representation Without Causation in LLM Activation Steering},
  author={Ciric, Lazar},
  year={2026},
  url={https://github.com/lciric/sycophancy-construct-validity}
}
```

## Related Work

- Panickssery et al. (2024) — Steering Llama 2 with CAA
- Sharma et al. (2024) — SycophancyEval benchmark (ICLR 2024)
- Sofroniew et al. (2026) — Emotion concept vectors in Claude
- MLAS (NeurIPS 2025 Workshop) — Multi-layer activation steering
- Fanous et al. (2025) — SycEval: progressive vs regressive sycophancy

## License

MIT
