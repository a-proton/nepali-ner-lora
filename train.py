"""
Parameter-Efficient Nepali Named Entity Recognition
with LoRA and Hindi Cross-Lingual Transfer

Authors: Avinash Gautam, Smarika Ghimire
Institution: School of Engineering, Pokhara University, Nepal
Email: avinashgautam@student.pu.edu.np , smarikaghimire@student.pu.edu.np


Paper: Parameter-Efficient Nepali NER with LoRA and Hindi Transfer
GitHub: https://github.com/a-proton/nepali-ner-lora
HuggingFace: https://huggingface.co/a-proton/nepali-ner-lora

Results:
- Config 5 (all attention): 81.87% ± 0.81% F1
- Config 2 (query+value):   79.25% ± 0.71% F1
- Full fine-tuning baseline: 86.2% F1 (Yadav et al., 2024)
"""

import os
import random
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics import classification_report

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Model
    "model_name":      "google-bert/bert-base-multilingual-cased",

    # LoRA - Config 5 (best: all attention modules)
    "lora_r":          16,
    "lora_alpha":      32,
    "lora_dropout":    0.1,
    "target_modules":  ["query", "value", "key", "dense"],

    # Training
    "max_length":      128,
    "batch_size":      16,
    "epochs":          5,
    "learning_rate":   2e-4,
    "warmup_ratio":    0.1,
    "grad_clip":       1.0,
    "seed":            42,

    # Data
    "nepali_data":     "nepali-ner/data/ebiquity_v2/raw/total.bio",
    "hindi_size":      6000,

    # Output
    "output_dir":      "./outputs/best_model",
}

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# DATA LOADING
# ============================================================
def load_nepali_data(filepath, seed=42):
    """Load and split EBIQUITY Nepali NER dataset."""
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

    # Fixed split using numpy for reproducibility
    np.random.seed(seed)
    indices = np.random.permutation(len(sentences))
    total   = len(sentences)
    train   = [sentences[i] for i in indices[:int(total*0.8)]]
    valid   = [sentences[i] for i in indices[int(total*0.8):int(total*0.9)]]
    test    = [sentences[i] for i in indices[int(total*0.9):]]

    print(f"Nepali - Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
    return train, valid, test


def load_hindi_data(size=6000):
    """Load WikiANN Hindi NER for cross-lingual transfer."""
    print(f"Loading Hindi WikiANN ({size} sentences)...")
    hindi = load_dataset("wikiann", "hi")
    tag_names = hindi['train'].features['ner_tags'].feature.names

    sentences = []
    for item in list(hindi['train']) + list(hindi['validation']):
        sentence = [
            (token, tag_names[tag_id])
            for token, tag_id in zip(item['tokens'], item['ner_tags'])
        ]
        if sentence:
            sentences.append(sentence)

    # Use fixed subset
    random.seed(42)
    sentences = random.sample(sentences, min(size, len(sentences)))
    print(f"Hindi data loaded: {len(sentences)} sentences")
    return sentences


def build_label_map(sentences):
    """Build label to ID mapping."""
    unique_tags = set()
    for sentence in sentences:
        for _, tag in sentence:
            unique_tags.add(tag)
    label_list = sorted(list(unique_tags))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label  = {i: l for l, i in label2id.items()}
    return label2id, id2label

# ============================================================
# TOKENIZATION
# ============================================================
def tokenize_and_align_labels(sentences, tokenizer, label2id, max_length=128):
    """Tokenize and align NER labels with subword tokens."""
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

# ============================================================
# DATASET
# ============================================================
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
# TRAINING
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), CONFIG["grad_clip"]
        )
        optimizer.step()
        scheduler.step()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} "
                  f"- Loss: {loss.item():.4f}")

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()

    return total_loss / len(loader)


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
    print("Nepali NER with LoRA and Hindi Transfer")
    print("Config 5: All attention modules (best configuration)")
    print("=" * 60)

    set_seed(CONFIG["seed"])

    # Download Nepali dataset
    if not os.path.exists("nepali-ner"):
        os.system("git clone https://github.com/oya163/nepali-ner.git")

    # Load data
    train_nepali, valid_nepali, test_nepali = load_nepali_data(
        CONFIG["nepali_data"], CONFIG["seed"]
    )
    hindi_sentences = load_hindi_data(CONFIG["hindi_size"])

    # Build label map
    all_sentences = train_nepali + valid_nepali + test_nepali
    label2id, id2label = build_label_map(all_sentences)
    print(f"Labels: {label2id}")

    # Combine Hindi + Nepali
    combined_train = train_nepali + hindi_sentences
    random.shuffle(combined_train)
    print(f"Combined training: {len(combined_train)} sentences")

    # Load tokenizer
    print(f"\nLoading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    # Tokenize
    print("Tokenizing data...")
    combined_enc = tokenize_and_align_labels(
        combined_train, tokenizer, label2id, CONFIG["max_length"]
    )
    valid_enc = tokenize_and_align_labels(
        valid_nepali, tokenizer, label2id, CONFIG["max_length"]
    )
    test_enc = tokenize_and_align_labels(
        test_nepali, tokenizer, label2id, CONFIG["max_length"]
    )
    print("Tokenization complete!")

    # Dataloaders
    combined_loader = DataLoader(
        NERDataset(combined_enc),
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )
    valid_loader = DataLoader(
        NERDataset(valid_enc),
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )
    test_loader = DataLoader(
        NERDataset(test_enc),
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

    # Load model with LoRA
    print(f"\nLoading {CONFIG['model_name']} + LoRA...")
    model = AutoModelForTokenClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.print_trainable_parameters()
    print(f"Device: {device}")

    # Training
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    total_steps = len(combined_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * CONFIG["warmup_ratio"]),
        num_training_steps=total_steps
    )

    print("\nStarting training...")
    print("=" * 60)

    best_valid_loss = float("inf")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    for epoch in range(CONFIG["epochs"]):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 40)

        train_loss = train_epoch(
            model, combined_loader, optimizer, scheduler, device
        )
        valid_loss = evaluate(model, valid_loader, device)
        epoch_time = time.time() - start_time

        print(f"\n  Train Loss:      {train_loss:.4f}")
        print(f"  Validation Loss: {valid_loss:.4f}")
        print(f"  Time:            {epoch_time:.1f}s")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.save_pretrained(CONFIG["output_dir"])
            tokenizer.save_pretrained(CONFIG["output_dir"])
            print(f"  Best model saved to {CONFIG['output_dir']}")

    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_valid_loss:.4f}")

    # Evaluate
    print("\nEvaluating on test set...")
    predictions, true_labels = get_predictions(
        model, test_loader, device, id2label
    )

    f1        = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall    = recall_score(true_labels, predictions)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print("\nDetailed Report:")
    print(classification_report(true_labels, predictions))


if __name__ == "__main__":
    main()