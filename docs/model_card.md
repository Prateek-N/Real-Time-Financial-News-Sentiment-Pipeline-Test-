# Model Card: Financial News FinBERT Sentiment Classifier

## Model Details
- **Architecture:** BERT-base fine-tuned on corporate and financial sentiment (110M parameters).
- **Baseline Checkpoint:** `ProsusAI/finbert` (Hugging Face Hub).
- **Label Mapping (`id2label`):**
  - `0`: `positive`
  - `1`: `negative`
  - `2`: `neutral`
- **Output:** Three softmax probabilities summing to 1.0; `sentiment_index = p_positive - p_negative` in `[-1.0, +1.0]`.

## Training and Optimization Protocol
- **Optimizer:** AdamW (`lr=2e-5`, weight decay `0.01`).
- **Scheduler:** Linear warmup with cosine decay.
- **Batch Size:** 16 (online micro-batching), 64 (historical archive scoring).
- **Early Stopping:** Monitored on held-out **validation macro-F1** (never the final test set).
- **Reproducibility:** Pinned seed (`torch.manual_seed(42)`).

## Evaluation Metric: Macro-F1
Per PRD Section 13, model performance is evaluated strictly on unweighted 3-class macro-F1:
$$\text{Macro-F1} = \frac{\text{F1}_{\text{positive}} + \text{F1}_{\text{negative}} + \text{F1}_{\text{neutral}}}{3}$$

### Controlled Empirical Comparison Results
- **Held-Out Test Sample Size:** 789 untouched human gold annotations
- **Baseline FinBERT Macro-F1:** `0.7939` (~0.79)
- **Fine-Tuned Candidate Macro-F1:** `0.8676` (~0.87)
- **Absolute Gain:** `+0.0737` (7.4 percentage points)
- **Relative Gain:** `9.28%`
- **95% Paired Bootstrap Confidence Interval:** `[0.0560, 0.0915]` (Statistically Significant: True)

| Sentiment Class | Baseline F1 | Fine-Tuned F1 | Delta F1 | Support |
|---|---|---|---|---|
| Positive | 0.7917 | 0.7631 | -0.0286 | 190 |
| Negative | 0.8350 | 1.0000 | +0.1650 | 172 |
| Neutral | 0.7551 | 0.8397 | +0.0846 | 427 |
| **Macro Average** | **0.7939** | **0.8676** | **+0.0737** | **789** |

- **Artifacts:** Verified in `artifacts/eval/model_comparison.md`, `artifacts/eval/classification_report.json`, and `artifacts/eval/paired_bootstrap_ci.json`.

