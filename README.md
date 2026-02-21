# AI Cyber Lab 2: Phishing URL Detection

## 1. Project Description
This repository implements a baseline machine learning pipeline for phishing URL detection (Track 1).  
The goal is binary classification of URLs into:
- `bad` (phishing / malicious)
- `good` (benign)

The project is structured for reproducibility and quick experimentation with separate data loading, training, and evaluation modules.

## 2. Dataset Source and Features
Dataset file: `data/raw/phishing_site_urls.csv` (provided in the lab materials).  
Columns:
- `URL`: raw URL string
- `Label`: class label (`bad` or `good`)

The baseline uses lexical URL features, including:
- URL length
- Counts of digits, letters, and symbols
- Counts of dots, hyphens, underscores, slashes, `?`, `=`, and `@`
- Presence of HTTPS prefix
- Presence of IP-like URL patterns
- Presence of suspicious keywords (for example: `login`, `verify`, `secure`, `password`, `paypal`)
- Digit/symbol ratios

## 3. Installation Instructions
From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Training and Evaluation Commands
From project root:

```bash
python -m src.train
python -m src.eval
```

Expected output artifacts:
- `results/model.json`
- `results/metrics.json`
- `results/confusion_matrix.png`

## 5. Baseline Results
Metrics generated from `python -m src.eval`:

- Accuracy: `0.8481`
- Precision: `0.8681`
- Recall: `0.3837`
- F1-score: `0.5322`

Confusion matrix (`[[TN, FP], [FN, TP]]`):
- `[[77247, 1332], [14078, 8764]]`

## 6. Ethics and Safety Considerations
- This project is for defensive cybersecurity education and detection research only.
- Predictions may include false positives and false negatives; outputs should support analysts, not replace them.
- The model uses URL string heuristics and can inherit dataset bias or miss novel attack patterns.
- Never use this model as a sole control for blocking decisions in production without additional monitoring and validation.
