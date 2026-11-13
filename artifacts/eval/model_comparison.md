# FinBERT Model Evaluation and Controlled Comparison

**Test Sample Size:** 789 held-out human-annotated headlines  
**Metric Definition:** 3-Class Unweighted Macro-F1: `(F1_pos + F1_neg + F1_neu) / 3`

## Quantitative Outcome Summary
- **Baseline FinBERT Macro-F1:** `0.7939` (~0.79)
- **Fine-Tuned Candidate Macro-F1:** `0.8676` (~0.87)
- **Absolute Gain:** `+0.0737` (7.4 percentage points)
- **Relative Improvement:** `9.28%`
- **95% Paired Bootstrap Confidence Interval:** `[0.056, 0.0915]` (Statistically Significant: True)

## Per-Class Breakdown

| Class | Baseline F1 | Candidate F1 | Delta F1 | Support |
|---|---|---|---|---|
| Positive | 0.7917 | 0.7631 | -0.0286 | 190.0 |
| Negative | 0.8350 | 1.0000 | +0.1650 | 172.0 |
| Neutral | 0.7551 | 0.8397 | +0.0846 | 427.0 |
| **Macro Average** | **0.7939** | **0.8676** | **+0.0737** | **789** |

## Artifacts Generated
- `artifacts/eval/confusion_matrices.png`
- `artifacts/eval/baseline_predictions.csv`
- `artifacts/eval/fine_tuned_predictions.csv`
- `artifacts/eval/classification_report.json`
- `artifacts/eval/paired_bootstrap_ci.json`
