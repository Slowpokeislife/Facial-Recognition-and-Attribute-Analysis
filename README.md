# Facial Recognition System

**A custom convolutional neural network trained on CelebA, then adapted to personalized binary face recognition through transfer learning and fine-tuning.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-CNN-25324A)
![Status](https://img.shields.io/badge/Status-Research_project-64748B)

## Overview

This project investigates whether a convolutional neural network (CNN) trained **from scratch** for facial attribute prediction can transfer its learned visual representations to a smaller, personalized face-recognition task.

The workflow has three stages:

1. **Facial attribute learning:** train a CNN on the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to predict 40 facial attributes.
2. **Transfer Learning v1:** reuse its convolutional layers and train a binary classifier to distinguish a target individual (`yo`) from other faces (`otros`).
3. **Transfer Learning v2:** address class imbalance through class weighting, stronger data augmentation, and threshold analysis.

> **Scope:** The recognition task is binary (target person vs. others), not identification among multiple named individuals. This is a research prototype, **not** a production-ready authentication or anti-spoofing system.

## Methodology

### 1. CNN trained from scratch on CelebA

- Dataset: CelebA, containing **202,599 face images** annotated with **40 binary attributes**.
- Input: **128 × 128 RGB** images normalized to `[0, 1]`.
- Architecture: four convolutional blocks with **64 → 128 → 256 → 512 filters**, max pooling, global average pooling, dense layers, and a **40-unit sigmoid** output.
- Optimization: Adam, binary cross-entropy, early stopping, model checkpoints, and adaptive learning-rate reduction.
- Recorded best validation **binary accuracy: 91.50%**.

**Important:** Binary accuracy here measures predictions over facial-attribute labels, **not** identity-recognition accuracy.

### 2. Personalized recognition (v1)

The pretrained CNN is used as a **feature extractor**. Its original attribute-classification head is replaced with a new binary classifier:

```text
128 × 128 RGB image
       ↓
Pretrained CNN convolutional base
       ↓
Global Average Pooling
       ↓
Dropout → Dense(128, ReLU) → Dropout
       ↓
Dense(1, Sigmoid)
       ↓
Target person / Other person
```

Training occurs in two phases:

- **Feature extraction:** freeze the convolutional base and train the new classification head.
- **Fine-tuning:** unfreeze the convolutional base while keeping its first 10 layers frozen; train with a lower learning rate.

### 3. Class imbalance and decision thresholds (v2)

The second experiment modifies training to improve sensitivity to the underrepresented target class:

- Stronger geometric and brightness augmentation.
- Class weights calculated from the training-set distribution and applied in both phases.
- Comparison of decision thresholds from **0.10 to 0.95** on the recorded validation predictions.

## Results

### Personalized recognition — reported validation results

| Metric | Transfer Learning v1 | Transfer Learning v2 |
|:--|--:|--:|
| Accuracy | 97.69% | 97.22% |
| Precision | 100.00% | 75.00% |
| Recall | 68.75% | 93.75% |
| F1-score | 81.48% | 83.33% |
| Decision threshold | 0.50 | 0.60 |

The v1 metrics are based on the report's validation confusion matrix, with the F1-score computed from precision and recall. V2 metrics are recorded in `Transfer Learning v2/umbrales.csv` at a **0.60** threshold (0.65 and 0.70 yield the same recorded metrics).

The main trade-off is that **v2 recognizes more positive examples but also introduces more false positives**. Higher recall does not mean universally better recognition or security.

![Threshold comparison](Transfer%20Learning%20v2/curvas_umbrales.png)

### Important evaluation limitations

- Results are from the project's **validation data**; they are not independent external benchmark results.
- The v2 threshold was selected using validation results. Its estimated performance may be optimistic until confirmed on a separate, untouched test set.
- The original personal photographs and complete dataset preparation pipeline are not included, so end-to-end reproduction of the personalized experiment requires rebuilding the data splits.
- No presentation-attack detection, demographic fairness evaluation, or deployment-security audit is documented.

## Repository structure

```text
.
├── Modelo CNN celebA/
│   ├── entrenar_celebA_grande.py      # Initial 40-attribute CNN training
│   ├── historial_grande.json         # Recorded training history
│   └── celebA_grande_mejor.h5        # Best model checkpoint
├── Transfer Learning/
│   ├── transfer_yo.py                # Binary recognition: v1
│   ├── historial_fase1.pkl
│   ├── historial_fase2.pkl
│   ├── mejor_modelo_fase1.h5
│   └── mejor_modelo_fase2.h5
├── Transfer Learning v2/
│   ├── transfer_yo_v2.py             # Class-weighted binary recognition
│   ├── historial_fase1_v2.pkl
│   ├── historial_fase2_v2.pkl
│   ├── mejor_modelo_fase1_v2.h5
│   ├── mejor_modelo_fase2_v2.h5
│   ├── umbrales.csv
│   └── curvas_umbrales.png
├── Reporte.ipynb                     # Original experimental report
└── README.md
```

If large model checkpoints are stored in GitHub Releases instead of the repository, update the file tree and link to the release.

## Running the research code

The scripts were developed for a GPU-equipped computing environment and retain **absolute paths to the original training directories**. They are provided as the original training implementation, **not as a ready-to-run application**.

To adapt the scripts to another environment:

1. Set up a compatible Python environment with TensorFlow/Keras, NumPy, pandas, and scikit-learn. Match versions to the saved `.h5` model format where needed.
2. Obtain CelebA from its official source and comply with its license and usage terms.
3. Update `IMG_FOLDER` and `ATTR_FILE` in `Modelo CNN celebA/entrenar_celebA_grande.py`.
4. For personalized experiments, create separate `train/` and `validation/` directories containing `otros/` and `yo/` class subdirectories, using images that you are authorized to process.
5. Update `MODELO_PRETRAINED`, `TRAIN_DIR`, and `VAL_DIR` in the corresponding transfer-learning script.
6. Run the CelebA training script before either transfer-learning script if a suitable pretrained checkpoint is not available.

Avoid evaluating on images that were used for fitting or threshold selection. For a new experiment, reserve a completely untouched **test set**.

## Privacy and responsible use

Facial images and trained identity-recognition checkpoints can carry **biometric privacy risks**. This repository is intended as an educational/research demonstration. Do not publish private training photographs or identifying datasets without appropriate consent. Consider keeping personalized model checkpoints private and making only the attribute-learning model, code, aggregate metrics, and non-identifying diagrams public.

## Key takeaways

- Building a CNN from scratch for multi-label facial attribute prediction.
- Reusing visual representations for a new task with limited data.
- Comparing frozen-feature extraction with partial fine-tuning.
- Handling class imbalance using weighted training and image augmentation.
- Understanding the **precision–recall trade-off** and the limitations of threshold tuning on validation data.

## Author

**Jorge Arturo Uc Torres**  
GitHub: [@Slowpokeislife](https://github.com/Slowpokeislife)
