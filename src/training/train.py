from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HeadlineDataset(Dataset):
    def __init__(self, headlines: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            headlines,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def train_model(
    train_csv: str = "data/gold_train.csv",
    val_csv: str = "data/gold_val.csv",
    base_model_id: str = "ProsusAI/finbert",
    output_dir: str = "artifacts/model",
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading training data from %s and validation from %s", train_csv, val_csv)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_id)
    model.to(device)

    # Official FinBERT id2label: {0: 'positive', 1: 'negative', 2: 'neutral'}
    label2id = {"positive": 0, "negative": 1, "neutral": 2}

    train_labels = [label2id[l] for l in train_df["gold_label"]]
    val_labels = [label2id[l] for l in val_df["gold_label"]]

    train_dataset = HeadlineDataset(train_df["headline"].tolist(), train_labels, tokenizer)
    val_dataset = HeadlineDataset(val_df["headline"].tolist(), val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_macro_f1 = 0.0

    logger.info("Starting fine-tuning for %d epochs on %s (total steps: %d)...", epochs, device, total_steps)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_preds: List[int] = []
        val_targets: List[int] = []

        with torch.inference_mode():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        # PRD Section 13: strictly unweighted macro-F1
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro")
        logger.info("Epoch %d/%d - Avg Train Loss: %.4f - Val Macro-F1: %.4f", epoch, epochs, avg_train_loss, val_macro_f1)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            logger.info("New best validation macro-F1: %.4f! Saving checkpoint to %s", best_val_macro_f1, output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    train_summary = {
        "base_model": base_model_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "best_val_macro_f1": round(float(best_val_macro_f1), 4),
        "device": str(device)
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)

    return best_val_macro_f1


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT with Validation Macro-F1 Selection")
    parser.add_argument("--train", default="data/gold_train.csv", help="Train CSV")
    parser.add_argument("--val", default="data/gold_val.csv", help="Val CSV")
    parser.add_argument("--base-model", default="ProsusAI/finbert", help="Base model identifier")
    parser.add_argument("--output", default="artifacts/model", help="Output model directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_model(
        train_csv=args.train,
        val_csv=args.val,
        base_model_id=args.base_model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

