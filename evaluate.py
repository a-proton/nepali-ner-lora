"""
Evaluation script for Nepali NER LoRA model
Authors: Avinash Gautam, Smarika Ghimire
Pokhara University, Nepal
avinasgautam344@gmail.com , smarikaghimire35@gmail.com
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
import random

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "model_path":  "./outputs/best_model",
    "base_model":  "google-bert/bert-base-multilingual-cased",
    "data_path":   "nepali-ner/data/ebiquity/cleaned/total.bio",
    "max_length":  128,
    "batch_size":  16,
    "seed":        42,
}

# ============================================================
# LOAD DATA
# ============================================================
def load_test_data(filepath, seed=42):
    """Load test split from EBIQUITY dataset."""
    random.seed(seed)
    sentences = []
    current = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    current.append((parts[0], parts[1]))

    if current:
        sentences.append(current)

    random.shuffle(sentences)
    total = len(sentences)
    test = sentences[int(total * 0.9):]
    print(f"Test sentences: {len(test)}")
    return test


def build_label_map(filepath):
    """Build label mapping from full dataset."""
    unique_tags = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                unique_tags.add(parts[1])

    label_list = sorted(list(unique_tags))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label  = {i: l for l, i in label2id.items()}
    return label2id, id2label


# ============================================================
# DATASET
# ============================================================
def tokenize_and_align_labels(sentences, tokenizer, label2id, max_length=128):
    tokenized = []
    for sentence in sentences:
        words = [w for w, _ in sentence]
        tags  = [t for _, t in sentence]

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != prev_word_idx:
                aligned_labels.append(label2id[tags[word_idx]])
            else:
                aligned_labels.append(-100)
            prev_word_idx = word_idx

        encoding["labels"] = torch.tensor([aligned_labels])
        tokenized.append(encoding)

    return tokenized


class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        return {
            "input_ids":      item["input_ids"].squeeze(),
            "attention_mask": item["attention_mask"].squeeze(),
            "labels":         item["labels"].squeeze()
        }


# ============================================================
# EVALUATION
# ============================================================
def get_predictions(model, loader, device, id2label):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=2)

            for pred_seq, label_seq in zip(predictions, labels):
                pred_tags  = []
                label_tags = []
                for pred, label in zip(pred_seq, label_seq):
                    if label.item() != -100:
                        pred_tags.append(id2label[pred.item()])
                        label_tags.append(id2label[label.item()])
                all_preds.append(pred_tags)
                all_labels.append(label_tags)

    return all_preds, all_labels


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Nepali NER LoRA - Evaluation")
    print("=" * 60)

    # Load data
    label2id, id2label = build_label_map(CONFIG["data_path"])
    test_data = load_test_data(CONFIG["data_path"])

    # Load tokenizer and model
    print(f"\nLoading model from {CONFIG['model_path']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
    model = AutoModelForTokenClassification.from_pretrained(
        CONFIG["model_path"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}!")

    # Tokenize test data
    test_enc = tokenize_and_align_labels(
        test_data, tokenizer, label2id, CONFIG["max_length"]
    )
    test_loader = DataLoader(
        NERDataset(test_enc),
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

    # Get predictions
    print("\nRunning evaluation...")
    predictions, true_labels = get_predictions(
        model, test_loader, device, id2label
    )

    # Calculate metrics
    f1        = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall    = recall_score(true_labels, predictions)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print("\nDetailed Report:")
    print(classification_report(true_labels, predictions))

    # Save results
    with open("results.txt", "w") as f:
        f.write("NEPALI NER LORA - EVALUATION RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"F1 Score:  {f1*100:.2f}%\n")
        f.write(f"Precision: {precision*100:.2f}%\n")
        f.write(f"Recall:    {recall*100:.2f}%\n")
        f.write("\nDetailed Report:\n")
        f.write(classification_report(true_labels, predictions))

    print("\nResults saved to results.txt")


if __name__ == "__main__":
    main()