# Nepali NER with LoRA and Hindi Transfer

First systematic study of parameter-efficient Named Entity
Recognition for Nepali using Low-Rank Adaptation (LoRA)
with Hindi cross-lingual transfer.

**Paper:** Parameter-Efficient Nepali NER with LoRA and
Hindi Cross-Lingual Transfer  
**Authors:** Avinash Gautam, Smarika Ghimire  
**Institution:** Pokhara University, Nepal  
**Code:** https://github.com/a-proton/nepali-ner-lora  
**Model:** https://huggingface.co/a-proton/nepali-ner-lora

---

## Main Results (EBIQUITY Test Set)

| Model                              | F1                 | Params   | Time      |
| ---------------------------------- | ------------------ | -------- | --------- |
| mBERT Full FT (Yadav 2024)         | 86.2%              | 177M     | 12hr      |
| **Config 5: LoRA all attn (ours)** | **81.87% ± 0.81%** | **2.7M** | **12min** |
| Config 2: LoRA q+v                 | 79.25% ± 0.71%     | 0.6M     | 12min     |
| Config 3: Two-phase LoRA           | 75.31%\*           | 0.6M     | 24min     |
| Config 4: LoRA r=32                | 72.06%\*           | 0.6M     | 12min     |
| Config 1: LoRA q+v Nepali only     | 77.92%\*           | 0.6M     | 12min     |

\*Single-seed result (seed=42)  
Results for Config 2 and Config 5 reported as
mean ± std over 5 seeds (42, 123, 456, 789, 2024)

**98.50% fewer parameters. 60× faster training.
Zero inference overhead (weights merged at inference).**

---

## PEFT Comparison (Nepali-only training)

All methods below use Nepali-only training to isolate
the effect of PEFT method choice from Hindi transfer.

| Method          | F1         | Trainable Params |
| --------------- | ---------- | ---------------- |
| Full FT (mBERT) | 79.97%     | 177M             |
| **LoRA (ours)** | **77.74%** | **595K**         |
| Prefix Tuning   | 63.96%     | 374K             |
| BitFit          | 60.97%     | 102K             |

LoRA outperforms all PEFT baselines by +13-17% F1.

---

## Hindi Data Scaling

| Hindi Sentences | F1         | Improvement          |
| --------------- | ---------- | -------------------- |
| 0 (Nepali only) | 75.06%     | baseline             |
| 1,000           | 77.21%     | +2.15%               |
| 3,000           | 75.76%     | +0.70%               |
| **6,000**       | **77.90%** | **+2.85%** ← optimal |

Note: Single-seed results. Non-monotonic trend at
3,000 sentences likely due to subset composition.

---

## Module Ablation

| Target Modules      | F1         | Params   |
| ------------------- | ---------- | -------- |
| query only          | 73.47%     | 300K     |
| value only          | 76.02%     | 300K     |
| query + value       | 77.74%     | 595K     |
| query + value + key | 78.04%     | 890K     |
| **all attention**   | **81.43%** | **2.7M** |

Single-seed results. Multi-seed result for all
attention: 81.87% ± 0.81% (reported in paper).

---

## Key Findings

1. LoRA all attention modules outperforms q+v by +2.62% F1
2. Hindi transfer improves Nepali NER by +2.85% F1
3. Combined training beats sequential by +4.2% F1
4. LoRA outperforms BitFit by +16.77% F1
5. r=16 prevents overfitting in low-resource settings
6. Zero inference overhead via weight merging

---

## Setup

```bash
pip install -r requirements.txt
```

## Data

```bash
git clone https://github.com/oya163/nepali-ner.git
```

Hindi data downloads automatically via HuggingFace
datasets library (WikiANN Hindi, 6,000 sentences).

## Training

```bash
python train.py
```

Default config uses all attention modules (best result).
Edit CONFIG dictionary in train.py to try other configurations:

```python
# Config 5 (best - default)
"target_modules": ["query", "value", "key", "dense"]

# Config 2 (lighter)
"target_modules": ["query", "value"]
```

## Evaluation

```bash
python evaluate.py
```

## Reproducibility

- Exact train/validation/test splits in data_splits.json
- Results reported as mean ± std over 5 random seeds
- Seed values: 42, 123, 456, 789, 2024
- Best model weights available on HuggingFace

## Model

HuggingFace: https://huggingface.co/a-proton/nepali-ner-lora

Uploaded model uses seed=456 (best individual run).
Paper reports mean ± std over all 5 seeds.

## Citation

Paper under review. Citation will be added upon publication.

## License

Apache 2.0

## Acknowledgments

AI writing assistance was used for drafting and editing
the paper. All experiments and research decisions were
made by the authors.
