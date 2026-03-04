# Provenance-Aware Trust Framework for Autonomous Vehicles: Leveraging Generative AI for Decentralized Information Validation



This repository implements a two-stage AI+Crypto pipeline for detecting malicious
Over-the-Air (OTA) software updates and V2X messages in Autonomous Vehicle networks.
It includes synthetic dataset generation, model fine-tuning, two validation systems
(traditional cryptographic and AI-enhanced), a reputation/Sybil-detection layer,
and comprehensive comparison analysis.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Repository Structure](#repository-structure)
3. [Pipeline at a Glance](#pipeline-at-a-glance)
4. [Environment Setup](#environment-setup)
5. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   - [Step 1 — Generate the Dataset](#step-1--generate-the-dataset)
   - [Step 2 — Fine-Tune the Classifier](#step-2--fine-tune-the-classifier)
   - [Step 3 — Run the Comparison Analysis](#step-3--run-the-comparison-analysis)
6. [Key Components](#key-components)
7. [Output Files](#output-files)
8. [Experimental Configuration](#experimental-configuration)
9. [Known Limitations](#known-limitations)


---

## System Overview

The pipeline defends against three classes of attacks that traditional
cryptography alone cannot reliably detect:

| Attack Class | Example | Why Crypto Misses It |
|---|---|---|
| **Sophisticated/unknown** | Complex multi-field anomaly | Signature is valid |
| **Context inconsistency** | Weather node distributing firmware | Cert is legitimate |
| **Infrastructure compromise** | Charging station with high-ver update | Chain is intact |

---

## Repository Structure

```
XYZ/
├── generate_realistic_dataset.py   # Step 1 — Synthetic dataset generation
├── Fixed_Fine_Tuning.py            # Step 2 — DistilBERT fine-tuning (anti-leakage)
├── Traditional_Cryptographic_System.py  # Baseline: crypto-only validator
├── Enhanced_AI_Crypto_System.py    # Main: two-stage AI+Crypto validator
├── reputation_system.py            # Reputation, Sybil & collusion detection
├── Comparison_Analysis.py          # Step 3 — Full comparative evaluation
│
├── data/                           # Generated CSV datasets (created in Step 1)
│   ├── training_set.csv
│   ├── validation_set.csv
│   ├── test_set.csv
│   └── structured_data.csv
│
├── results_no_leakage/             # Saved model checkpoints (created in Step 2)
│   └── checkpoint-*/
│
├── fixed_training_results.json     
├── extended_comparison_metrics.json 
├── comprehensive_comparison_results.csv
│
└── plots/                         
    ├── plot_performance_metrics.pdf
    ├── plot_error_rates.pdf
    ├── plot_confusion_matrix_traditional.pdf
    ├── plot_confusion_matrix_ai_crypto.pdf
    ├── plot_detection_by_attack_type.pdf
    ├── plot_f1_across_datasets.pdf
    ├── roc_curves_comparison.pdf
    └── precision_recall_curves.pdf
```

> **Note:** Other `.py` files in the directory are earlier experimental versions
> kept for reference. The canonical pipeline uses the three files listed above.

---

## Pipeline at a Glance

```
generate_realistic_dataset.py
        │  produces data/
        ▼
Fixed_Fine_Tuning.py
        │  produces results_no_leakage/  +  fixed_training_results.json
        ▼
Comparison_Analysis.py
        │  imports TraditionalCryptographicValidator
        │  imports EnhancedAICryptoValidator  ←── reputation_system.py
        │
        ▼  produces CSVs, PDFs, JSON
```

---

## Environment Setup

### Google Colab (recommended — used for all reported experiments)

```python
# In a Colab cell:
!pip install transformers datasets scikit-learn pandas numpy \
             matplotlib seaborn cryptography torch
```

GPU runtime: **Tesla T4** (FP16 inference, 16 GB VRAM).  
All latency figures reported in the paper are measured on this platform.

### Local / conda

```bash
conda create -n av_validation python=3.10
conda activate av_validation
pip install transformers datasets scikit-learn pandas numpy \
            matplotlib seaborn cryptography torch
```

---

## Step-by-Step Execution Guide

### Step 1 — Generate the Dataset

```bash
python generate_realistic_dataset.py
```

**What it does:**
- Synthesizes **10,000 records** from nine communication channel types
  (`OTA`, `P2P`, `charging_station`, `nearby_car`, `parking_garage`,
   `police_station`, `traffic_sight`, `weather_station`, `flood_alert`)
- Labels records using **7 attack-scenario predicates** (see table below)


**Attack scenarios encoded in the labels:**

| Scenario | Bernoulli p | Predicate |
|---|---|---|
| Timing attack | 0.05 | `t > 450` AND `ver < 5` |
| Version rollback | 0.03 | `ver > 25` AND `t < 50` |
| Path manipulation | 0.04 | `path_len > 6` AND `channel == P2P` |
| Context inconsistency | 0.03 | `PoliceDept` with `ver > 30` OR `WeatherSvc` via P2P |
| Infrastructure compromise | 0.02 | charging/parking channel AND `ver > 28` |
| Sophisticated/unknown | 0.08 | `(t%7 + ver%5 + path_len%3) > 8` |
| Random catch-all | 0.15 | Independent Bernoulli on remaining benign samples |

**Anti-leakage check** — the script prints per-role and per-creator malicious
rates; no field should exceed 0.8 correlation.

**Expected output:**
```
./data/training_set.csv      7,000 rows
./data/validation_set.csv    1,000 rows
./data/test_set.csv          2,000 rows
./data/structured_data.csv   10,000 rows (with attack_type column)
```

---

### Step 2 — Fine-Tune the Classifier

```bash
python Fixed_Fine_Tuning.py
```

**What it does:**
- Loads the three CSV splits from `./data/`
- Applies a secondary cleaning pass on training data only
  (regex-based timestamp/version jitter; test data is kept pristine)
- Fine-tunes `distilbert-base-uncased` (66 M parameters) for binary
  sequence classification using a `WeightedTrainer` with class-balanced
  cross-entropy loss
- Writes `./fixed_training_results.json` containing the `best_checkpoint`
  path used automatically by Step 3

**Key hyperparameters (editable at top of file):**

| Parameter | Default |
|---|---|
| Base model | `distilbert-base-uncased` |
| Max token length | 512 |
| Batch size (train) | 16 |
| Learning rate | 2e-5 |
| Epochs | 10 (early stopping) |
| Output dir | `./results_no_leakage/` |

**Expected output:**
```
./results_no_leakage/checkpoint-<best_step>/
./fixed_training_results.json
```

---

### Step 3 — Run the Comparison Analysis

```bash
python Comparison_Analysis.py
```

**What it does:**
1. Auto-locates the best checkpoint from `fixed_training_results.json`
   (falls back to scanning `results_no_leakage/` if needed)
2. Evaluates **Traditional Cryptographic System** on train / val / test sets
3. Evaluates **Enhanced AI+Crypto System** on the same splits
4. Prints side-by-side metric tables (accuracy, precision, recall, F1,
   specificity, FPR, FNR)
5. Generates **8 publication-quality PDF plots**
6. Prints the **R3 / R4 / R5 extended analysis**:
   - **R3** — Per-source reputation scores, Sybil suspects, collusion pairs,
     temporal decay breakdown
   - **R4** — Stage 1 / Stage 2 / combined latency (mean, P50, P95, max),
     IEEE 1609.2 / ETSI ITS compliance verdict
   - **R5** — Quarantine-by-default statistics (rate, threshold, category)
7. Saves `comprehensive_comparison_results.csv` and
   `extended_comparison_metrics.json`



### `reputation_system.py`
Per-source trustworthiness tracker:

```
R_s(t) = (1/Z) · Σ_i  w_i · o_i · exp(−λ · (t − t_i))
```

**Sybil resistance** flags a source if it has:
- Fewer than 5 total interactions, **or**
- Interactions over fewer than 1 distinct channel, **or**
- A perfectly uniform outcome history (≥ 10 interactions all +1 or all −1)

**Collusion detection** flags source pairs whose verdict agreement rate
exceeds **85 %** over shared interaction timestamps.

## Experimental Configuration

All experiments reported in the paper were run on:

| Resource | Spec |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 (16 GB VRAM) |
| Inference precision | FP16 |
| Stage 1 base model | `distilbert-base-uncased` (66 M params) |
| Stage 2 | Deterministic rule engine (no additional model) |
| Dataset size | 10,000 samples (70/10/20 split) |
| Random seed | 42 |


> **Edge-deployment projection:** INT8-quantized DistilBERT on an NVIDIA
> Xavier NX (21 TOPS) is estimated at 14–18 ms per inference. Stage 2
> remains rule-based (< 0.05 ms) with no additional hardware requirement.

---

## Known Limitations

- **Synthetic data only.** The dataset is generated from a structured grammar
  and does not replicate real-world AV telemetry distributions.
- **Uniform channel/creator frequencies.** Real V2X deployments are heavily
  skewed toward OTA and V2V channels.
- **No temporal correlation.** Successive updates from the same node are
  generated independently.
- **Stage 2 is rule-based.** The current implementation does not deploy a
  generative LLM in Stage 2; the 80–120 ms figure in the paper is a
  projected overhead for a 350 M-parameter model on Xavier NX hardware.

The generation grammar is fully parameterizable; researchers can adjust
channel frequencies, attack probabilities, and Bernoulli thresholds to
match their target deployment context.


*For questions or issues, please open a GitHub issue or contact the
corresponding author.*
