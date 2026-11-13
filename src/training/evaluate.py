from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def predict_dataset(model, tokenizer, headlines: List[str], device, batch_size: int = 32) -> Tuple[List[str], np.ndarray]:
    """Runs batch inference and returns predicted string labels and softmax probabilities."""
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    all_preds: List[str] = []
    all_probs: List[np.ndarray] = []

    with torch.inference_mode():
        for i in range(0, len(headlines), batch_size):
            chunk = headlines[i:i + batch_size]
            tokens = tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            logits = model(**tokens).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for p_vec in probs:
                best_idx = int(p_vec.argmax())
                all_preds.append(id2label.get(best_idx, "neutral"))
                all_probs.append(p_vec)

    return all_preds, np.array(all_probs)


def paired_bootstrap_test(
    y_true: List[str],
    y_pred_base: List[str],
    y_pred_cand: List[str],
    n_resamples: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Computes paired bootstrap confidence interval for Delta macro-F1 per PRD Section 13."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_t = np.array(y_true)
    y_b = np.array(y_pred_base)
    y_c = np.array(y_pred_cand)

    deltas = []
    labels = ["positive", "negative", "neutral"]

    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        sample_true = y_t[indices]
        sample_base = y_b[indices]
        sample_cand = y_c[indices]

        f1_base = f1_score(sample_true, sample_base, average="macro", labels=labels, zero_division=0)
        f1_cand = f1_score(sample_true, sample_cand, average="macro", labels=labels, zero_division=0)
        deltas.append(f1_cand - f1_base)

    deltas = np.sort(deltas)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    mean_delta = float(np.mean(deltas))

    return {
        "resamples": n_resamples,
        "mean_delta_macro_f1": round(mean_delta, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "excludes_zero": bool(ci_lower > 0)
    }


def evaluate_models(
    test_csv: str = "data/gold_test.csv",
    baseline_model_id: str = "ProsusAI/finbert",
    candidate_model_dir: str = "artifacts/model",
    eval_dir: str = "artifacts/eval"
) -> Dict[str, Any]:
    """Rigorous evaluation comparing baseline and improved models on identical held-out test data."""
    os.makedirs(eval_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading held-out test set from %s...", test_csv)
    test_df = pd.read_csv(test_csv)
    headlines = test_df["headline"].tolist()
    y_true = test_df["gold_label"].tolist()

    # 1. Evaluate Baseline Model
    logger.info("Evaluating baseline model: %s", baseline_model_id)
    base_tokenizer = AutoTokenizer.from_pretrained(baseline_model_id)
    base_model = AutoModelForSequenceClassification.from_pretrained(baseline_model_id).to(device)
    base_preds, base_probs = predict_dataset(base_model, base_tokenizer, headlines, device)

    # 2. Evaluate Candidate Model
    logger.info("Evaluating candidate fine-tuned model: %s", candidate_model_dir)
    cand_tokenizer = AutoTokenizer.from_pretrained(candidate_model_dir)
    cand_model = AutoModelForSequenceClassification.from_pretrained(candidate_model_dir).to(device)
    cand_preds, cand_probs = predict_dataset(cand_model, cand_tokenizer, headlines, device)

    # 3. Calculate Core Metrics
    labels = ["positive", "negative", "neutral"]
    base_macro_f1 = f1_score(y_true, base_preds, average="macro", labels=labels)
    cand_macro_f1 = f1_score(y_true, cand_preds, average="macro", labels=labels)

    base_report = classification_report(y_true, base_preds, labels=labels, output_dict=True)
    cand_report = classification_report(y_true, cand_preds, labels=labels, output_dict=True)

    # 4. Bootstrap Confidence Interval
    bootstrap_res = paired_bootstrap_test(y_true, base_preds, cand_preds, n_resamples=1000)

    # 5. Save Prediction CSVs
    test_df["baseline_predicted"] = base_preds
    test_df.to_csv(os.path.join(eval_dir, "baseline_predictions.csv"), index=False)

    test_df["fine_tuned_predicted"] = cand_preds
    test_df.to_csv(os.path.join(eval_dir, "fine_tuned_predictions.csv"), index=False)

    # 6. Plot Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm_base = confusion_matrix(y_true, base_preds, labels=labels)
    cm_cand = confusion_matrix(y_true, cand_preds, labels=labels)

    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"Baseline FinBERT (Macro-F1: {base_macro_f1:.4f})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True Gold")

    sns.heatmap(cm_cand, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"Fine-Tuned Model (Macro-F1: {cand_macro_f1:.4f})")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True Gold")

    plt.tight_layout()
    cm_path = os.path.join(eval_dir, "confusion_matrices.png")
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    logger.info("Saved confusion matrices to %s", cm_path)

    # 7. Classification Report Artifact
    full_report = {
        "evaluation_dataset": {
            "test_csv": test_csv,
            "sample_size": len(test_df),
            "class_distribution": test_df["gold_label"].value_counts().to_dict()
        },
        "baseline": {
            "model_id": baseline_model_id,
            "macro_f1": round(base_macro_f1, 4),
            "classification_report": base_report
        },
        "candidate": {
            "model_dir": candidate_model_dir,
            "macro_f1": round(cand_macro_f1, 4),
            "classification_report": cand_report
        },
        "comparison": {
            "absolute_macro_f1_gain": round(cand_macro_f1 - base_macro_f1, 4),
            "relative_improvement_pct": round(((cand_macro_f1 - base_macro_f1) / base_macro_f1) * 100.0, 2),
            "paired_bootstrap_95_ci": bootstrap_res
        }
    }

    with open(os.path.join(eval_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    with open(os.path.join(eval_dir, "paired_bootstrap_ci.json"), "w", encoding="utf-8") as f:
        json.dump(bootstrap_res, f, indent=2)

    # 8. Markdown Comparison Summary
    md_summary = f"""# FinBERT Model Evaluation and Controlled Comparison

**Test Sample Size:** {len(test_df)} held-out human-annotated headlines  
**Metric Definition:** 3-Class Unweighted Macro-F1: `(F1_pos + F1_neg + F1_neu) / 3`

## Quantitative Outcome Summary
- **Baseline FinBERT Macro-F1:** `{base_macro_f1:.4f}` (~{round(base_macro_f1, 2)})
- **Fine-Tuned Candidate Macro-F1:** `{cand_macro_f1:.4f}` (~{round(cand_macro_f1, 2)})
- **Absolute Gain:** `+{cand_macro_f1 - base_macro_f1:.4f}` ({((cand_macro_f1 - base_macro_f1) * 100):.1f} percentage points)
- **Relative Improvement:** `{full_report['comparison']['relative_improvement_pct']}%`
- **95% Paired Bootstrap Confidence Interval:** `[{bootstrap_res['ci_95_lower']}, {bootstrap_res['ci_95_upper']}]` (Statistically Significant: {bootstrap_res['excludes_zero']})

## Per-Class Breakdown

| Class | Baseline F1 | Candidate F1 | Delta F1 | Support |
|---|---|---|---|---|
| Positive | {base_report['positive']['f1-score']:.4f} | {cand_report['positive']['f1-score']:.4f} | {cand_report['positive']['f1-score'] - base_report['positive']['f1-score']:+.4f} | {base_report['positive']['support']} |
| Negative | {base_report['negative']['f1-score']:.4f} | {cand_report['negative']['f1-score']:.4f} | {cand_report['negative']['f1-score'] - base_report['negative']['f1-score']:+.4f} | {base_report['negative']['support']} |
| Neutral | {base_report['neutral']['f1-score']:.4f} | {cand_report['neutral']['f1-score']:.4f} | {cand_report['neutral']['f1-score'] - base_report['neutral']['f1-score']:+.4f} | {base_report['neutral']['support']} |
| **Macro Average** | **{base_macro_f1:.4f}** | **{cand_macro_f1:.4f}** | **{cand_macro_f1 - base_macro_f1:+.4f}** | **{len(test_df)}** |

## Artifacts Generated
- `artifacts/eval/confusion_matrices.png`
- `artifacts/eval/baseline_predictions.csv`
- `artifacts/eval/fine_tuned_predictions.csv`
- `artifacts/eval/classification_report.json`
- `artifacts/eval/paired_bootstrap_ci.json`
"""

    with open(os.path.join(eval_dir, "model_comparison.md"), "w", encoding="utf-8") as f:
        f.write(md_summary)

    logger.info("Evaluation complete! Baseline Macro-F1: %.4f, Candidate Macro-F1: %.4f", base_macro_f1, cand_macro_f1)
    return full_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline vs Fine-Tuned FinBERT on Held-Out Test Set")
    parser.add_argument("--test-set", default="data/gold_test.csv", help="Test set CSV")
    parser.add_argument("--baseline", default="ProsusAI/finbert", help="Baseline model ID")
    parser.add_argument("--candidate", default="artifacts/model", help="Candidate model checkpoint directory")
    args = parser.parse_args()

    evaluate_models(test_csv=args.test_set, baseline_model_id=args.baseline, candidate_model_dir=args.candidate)


if __name__ == "__main__":
    main()

