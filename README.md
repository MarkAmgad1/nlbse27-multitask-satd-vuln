# Comments First, Weights Matter: Multitask Vulnerability and SATD Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YI0NAOlEyXxoHPA7qPMk_evr-rAmUBOU)

Research project for **Requirements Engineering and Architectural Design (0103613)**, Al Ain University, Fall 2026-2027.
**Track B1:** NLBSE'27 tool competition, multitask code classification for vulnerability and self-admitted technical debt (SATD) detection.

## Problem
One model reads a C/C++ function with its comments and predicts two binary labels: **vulnerability** (from the code) and **SATD** (from the comments). Data: [`NLBSE/nlbse27-code-comment-classification`](https://huggingface.co/datasets/NLBSE/nlbse27-code-comment-classification), derived from MADE-WIC.

**Research questions**
- **RQ1.** How do TF-IDF + logistic regression, the official CodeBERT baseline and our multitask variant compare?
- **RQ2.** Which of our changes (comments-first input, class-weighted loss, per-label thresholds) contributes? (ablation, 3 seeds)
- **RQ3.** What does the model still get wrong? (error analysis)

## Approach
Same architecture as the official baseline (CodeBERT, shared encoder, two sigmoid outputs) with three changes:
1. **Comments first:** comments are extracted and fed before the code (`<s> comments </s></s> code </s>`), so truncation at 512 tokens cuts code, not comments.
2. **Class-weighted loss:** positives weighted by `sqrt(N_neg / N_pos)` (1.43 vulnerability, 7.72 SATD).
3. **Per-label thresholds** tuned on a validation split.

## Results (public test split, 30,655 functions)

| Approach | Vuln. F1 | SATD F1 | Mean F1 |
|---|---|---|---|
| TF-IDF + LR | 0.748 | 0.854 | 0.801 |
| Official baseline (reproduced) | **0.939** | 0.881 | 0.910 |
| Ours, threshold 0.5 | 0.925 | **0.929** | **0.927** |
| Ours, full method (tuned thresholds) | 0.925 | 0.922 | 0.923 |

- Gain over the baseline in mean F1: +0.013, bootstrap 95% CI [0.006, 0.022]. SATD better (McNemar p = 3.1e-6), vulnerability worse (p < 1e-16) after one epoch.
- Ablation: the class-weighted loss is decisive for SATD (0.889 ± 0.002 vs 0.215 ± 0.373 without it, on 10% of the data).
- Most remaining SATD errors are in functions with no comment in the input.

See [`paper/`](paper/) for the full paper.

## Repository layout
```
notebooks/   NLBSE27_multitask_final.ipynb (with outputs)
src/nlbse27/ preprocessing, metrics and statistics helpers used by the notebook
data/        how to get the dataset (not stored here)
results/     tables and figures produced by the notebook
paper/       LaTeX source (IEEEtran), figures, PDF
```

## How to reproduce
1. Open the notebook in Colab (badge above), select **Runtime → Change runtime type → T4 GPU**, then **Run all**.
2. The notebook mounts Google Drive and caches the trained model (`MyDrive/nlbse27_model`) and all results (`MyDrive/nlbse27_results`).
   - With caches present: about 15 minutes.
   - From scratch: main model about 80 minutes, ablation about 12 minutes per run (9 runs). Every finished run is saved, so the notebook can be resumed across sessions.
3. Everything is seeded (split seed 42, ablation seeds 42, 43, 44).

Requirements for a local run are in `requirements.txt`; a CUDA GPU is strongly recommended.

## Attribution
The dataset, task definition and baseline model come from the official NLBSE'27 repository (<https://github.com/nlbse2027/code-comment-classification>). Our additions: comments-first input, class-weighted loss, per-label thresholds, the TF-IDF baseline, the ablation, statistical tests and error analysis.

## Use of generative AI
An AI assistant (Claude, Anthropic) was used to help write and debug the notebook and to draft and edit the paper. All experiments were run by the author, every reported number was checked against the notebook outputs, and the author takes responsibility for the code and claims.

## License
Code: MIT (see `LICENSE`). Data: MADE-WIC, CC BY 4.0, not redistributed here.
