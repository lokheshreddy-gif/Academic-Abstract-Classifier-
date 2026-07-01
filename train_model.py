import os
import json
import ast
import csv
from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


DATA_DIR = "."
FINAL_MODEL_DIR = "final_model"
TARGET_SUBSET_SIZE = 1200
CPU_MAX_SAMPLES = 800
RANDOM_SEED = 42


def find_first_table_file(base_dir: str) -> str:
    for root, _, files in os.walk(base_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith(".csv") or lower.endswith(".tsv"):
                return os.path.join(root, name)
    raise FileNotFoundError("No CSV/TSV file found in workspace.")


def detect_columns(cols: List[str]) -> Tuple[str, str]:
    lower_cols = [c.lower() for c in cols]

    abstract_candidates = [
        "abstract",
        "summary",
        "summaries",
        "text",
        "content",
        "abstracts",
        "description",
    ]
    label_candidates = [
        "label",
        "labels",
        "category",
        "categories",
        "subject",
        "subjects",
        "term",
        "terms",
        "class",
        "classes",
        "field",
    ]

    abstract_col: Optional[str] = None
    label_col: Optional[str] = None

    for cand in abstract_candidates:
        for orig, low in zip(cols, lower_cols):
            if low == cand or cand in low:
                abstract_col = orig
                break
        if abstract_col:
            break

    for cand in label_candidates:
        for orig, low in zip(cols, lower_cols):
            if low == cand or cand in low:
                label_col = orig
                break
        if label_col:
            break

    # Heuristic fallbacks
    if abstract_col is None:
        for orig, low in zip(cols, lower_cols):
            if "abs" in low:
                abstract_col = orig
                break

    if label_col is None:
        for orig, low in zip(cols, lower_cols):
            if "cat" in low or "subj" in low:
                label_col = orig
                break

    # Final fallback: first two columns
    if abstract_col is None or label_col is None:
        if len(cols) >= 2:
            abstract_col = abstract_col or cols[0]
            label_col = label_col or cols[1]
        else:
            raise ValueError("Could not detect suitable abstract/label columns.")

    return abstract_col, label_col


def normalize_label(raw_label: str) -> str:
    s = "" if raw_label is None else str(raw_label).strip()
    # Try to parse Python list-like labels, e.g. "['cs.CV', 'cs.LG']"
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                return str(parsed[0])
        except Exception:
            pass
    # Otherwise, split on comma or space and take the first token
    if "," in s:
        return s.split(",")[0].strip()
    if " " in s:
        return s.split()[0].strip()
    return s


def prepare_rows(path: str) -> Tuple[List[Dict[str, str]], str, str]:
    sep = "\t" if path.lower().endswith(".tsv") else ","
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        cols = reader.fieldnames or []
        abstract_col, label_col = detect_columns(cols)

        rows: List[Dict[str, str]] = []
        for row in reader:
            raw_abs = row.get(abstract_col, "")
            raw_label = row.get(label_col, "")
            abstract = "" if raw_abs is None else str(raw_abs).strip()
            label = normalize_label(raw_label)
            if not abstract or not label:
                continue
            if len(abstract) < 40:
                continue
            rows.append({"abstract": abstract, "label": label})

    return rows, "abstract", "label"


def build_balanced_subset(rows: List[Dict[str, str]], target_size: int) -> List[Dict[str, str]]:
    if len(rows) <= target_size:
        rng = np.random.default_rng(RANDOM_SEED)
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        return [rows[i] for i in indices]

    labels = sorted({r["label"] for r in rows})
    per_class = max(1, target_size // len(labels))

    rng = np.random.default_rng(RANDOM_SEED)
    balanced: List[Dict[str, str]] = []

    for lbl in labels:
        group = [r for r in rows if r["label"] == lbl]
        if not group:
            continue
        idxs = rng.choice(len(group), size=per_class, replace=len(group) < per_class)
        for i in idxs:
            balanced.append(group[int(i)])

    rng.shuffle(balanced)
    if len(balanced) > target_size:
        balanced = balanced[:target_size]
    return balanced


class AbstractDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    table_path = find_first_table_file(DATA_DIR)
    print(f"Using data file: {table_path}")

    rows, abstract_col, label_col = prepare_rows(table_path)
    print(f"Detected abstract column: {abstract_col}, label column: {label_col}")
    print(f"Dataset size after cleaning: {len(rows)}")

    subset_target = TARGET_SUBSET_SIZE
    if not torch.cuda.is_available():
        subset_target = min(subset_target, CPU_MAX_SAMPLES)
        print(f"No GPU detected. Limiting subset size to {subset_target}")

    balanced_rows = build_balanced_subset(rows, subset_target)
    print(f"Balanced subset size: {len(balanced_rows)}")

    label_list = sorted({r["label"] for r in balanced_rows})
    label_to_id = {lbl: i for i, lbl in enumerate(label_list)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    # Build arrays for labels
    texts = [r["abstract"] for r in balanced_rows]
    label_ids = np.array([label_to_id[r["label"]] for r in balanced_rows], dtype=int)

    # Manual stratified 80/10/10 split
    rng = np.random.default_rng(RANDOM_SEED)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for lbl_id in sorted(set(label_ids.tolist())):
        idx = np.where(label_ids == lbl_id)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        n_test = n - n_train - n_val
        train_indices.extend(idx[:n_train])
        val_indices.extend(idx[n_train : n_train + n_val])
        test_indices.extend(idx[n_train + n_val : n_train + n_val + n_test])

    def build_split(indices: List[int]) -> Tuple[List[str], List[int], List[Dict[str, str]]]:
        split_texts = [texts[i] for i in indices]
        split_labels = [int(label_ids[i]) for i in indices]
        split_rows = [
            {
                "abstract": texts[i],
                "label": balanced_rows[i]["label"],
                "label_id": int(label_ids[i]),
            }
            for i in indices
        ]
        return split_texts, split_labels, split_rows

    train_texts, train_label_ids, train_rows = build_split(train_indices)
    val_texts, val_label_ids, val_rows = build_split(val_indices)
    test_texts, test_label_ids, test_rows = build_split(test_indices)

    # Save splits as CSV files
    for filename, split_rows in [
        ("train.csv", train_rows),
        ("val.csv", val_rows),
        ("test.csv", test_rows),
    ]:
        with open(filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["abstract", "label", "label_id"])
            writer.writeheader()
            for row in split_rows:
                writer.writerow(row)

    print(
        f"Splits sizes: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}, "
        f"num_labels={len(label_list)}"
    )

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    max_length = 128
    train_dataset = AbstractDataset(train_texts, train_label_ids, tokenizer, max_length)
    val_dataset = AbstractDataset(val_texts, val_label_ids, tokenizer, max_length)
    test_dataset = AbstractDataset(test_texts, test_label_ids, tokenizer, max_length)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())

        num_labels = len(label_list)
        # Build confusion matrix for metrics
        cm_local = np.zeros((num_labels, num_labels), dtype=int)
        for t, p in zip(labels, preds):
            cm_local[int(t), int(p)] += 1

        per_class_precisions = []
        per_class_recalls = []
        per_class_f1s = []

        for i in range(num_labels):
            tp = cm_local[i, i]
            fp = cm_local[:, i].sum() - tp
            fn = cm_local[i, :].sum() - tp
            precision_i = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall_i = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1_i = (
                2 * precision_i * recall_i / (precision_i + recall_i)
                if precision_i + recall_i > 0
                else 0.0
            )
            per_class_precisions.append(precision_i)
            per_class_recalls.append(recall_i)
            per_class_f1s.append(f1_i)

        precision_macro = float(np.mean(per_class_precisions))
        recall_macro = float(np.mean(per_class_recalls))
        f1_macro = float(np.mean(per_class_f1s))

        return {
            "accuracy": acc,
            "precision": precision_macro,
            "recall": recall_macro,
            "f1": f1_macro,
        }

    batch_size = 16
    args = TrainingArguments(
        output_dir="model_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("Evaluating on test set...")
    test_output = trainer.predict(test_dataset)
    test_logits = test_output.predictions
    test_labels = np.array(test_output.label_ids)
    test_preds = np.argmax(test_logits, axis=-1)

    num_labels = len(label_list)
    cm = np.zeros((num_labels, num_labels), dtype=int)
    for t, p in zip(test_labels, test_preds):
        cm[int(t), int(p)] += 1

    acc = float((test_labels == test_preds).mean())

    per_class = {}
    per_class_precisions = []
    per_class_recalls = []
    per_class_f1s = []

    for i, label_name in enumerate(label_list):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision_i = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall_i = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1_i = (
            2 * precision_i * recall_i / (precision_i + recall_i)
            if precision_i + recall_i > 0
            else 0.0
        )

        per_class_precisions.append(precision_i)
        per_class_recalls.append(recall_i)
        per_class_f1s.append(f1_i)

        per_class[label_name] = {
            "precision": precision_i,
            "recall": recall_i,
            "f1": f1_i,
            "support": int((test_labels == i).sum()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    precision_macro = float(np.mean(per_class_precisions))
    recall_macro = float(np.mean(per_class_recalls))
    f1_macro = float(np.mean(per_class_f1s))

    metrics = {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_class": per_class,
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix
    with open("confusion_matrix.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + label_list)
        for i, label_name in enumerate(label_list):
            row = [label_name] + [int(x) for x in cm[i]]
            writer.writerow(row)

    # Save misclassified samples
    with open("misclassified_samples.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["abstract", "true_label", "pred_label"])
        for text, true_id, pred_id in zip(test_texts, test_labels, test_preds):
            if int(true_id) != int(pred_id):
                writer.writerow(
                    [text, id_to_label[int(true_id)], id_to_label[int(pred_id)]]
                )

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    trainer.model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    with open(os.path.join(FINAL_MODEL_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(label_list, f, indent=2)

    print("Training and evaluation complete. Artifacts saved.")


if __name__ == "__main__":
    main()


