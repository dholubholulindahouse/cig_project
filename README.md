# Distorted Visual Sequence Pattern Recognition using Deep Learning

A deep learning framework for robust recognition of heavily distorted six-character alphanumeric sequences from grayscale CAPTCHA-style images.

**Author:** Siddhi Jha
**Enrollment No.:** 25117135

---

## Overview

This repository presents an end-to-end computer vision pipeline for recognizing distorted character sequences embedded within noisy and visually challenging images. The task involves predicting a fixed-length six-character code from grayscale images containing significant visual degradation.

The dataset includes a variety of real-world OCR challenges, such as:

* Background noise and clutter
* Character overlap
* Blur and compression artifacts
* Geometric distortions
* Random occlusions
* Irregular spacing and alignment

Despite these challenges, every image contains exactly six characters drawn from a fixed alphabet of 31 symbols.

The objective is to reconstruct the complete code with minimal Character Error Rate (CER).

---

## Problem Formulation

Although the task resembles traditional Optical Character Recognition (OCR), it differs from standard text-recognition problems in several important ways:

* Every label has a fixed length of six characters.
* The alphabet is finite and known beforehand.
* Sequence decoding methods can struggle with repeated or heavily distorted symbols.
* Character positions are fixed and equally important.

Instead of treating the problem as sequence transcription, the solution reformulates it as:

> Six parallel classification tasks operating on a shared visual representation.

Each character position is predicted independently while sharing features extracted by a common convolutional backbone.

---

## Model Architecture

The final model is built entirely from scratch using PyTorch and consists of:

* Deep Residual Convolutional Backbone
* Squeeze-and-Excitation (SE) Attention Blocks
* Global Average Pooling
* Shared Feature Embedding Layer
* Six Independent Classification Heads

### Architecture Flow

```text
Input Image (64 × 200 × 1)
          │
          ▼
Residual SE-CNN Backbone
          │
          ▼
Global Average Pooling
          │
          ▼
512-D Shared Embedding
          │
          ▼
6 Parallel Classification Heads
          │
          ▼
6-Character Prediction
```

### Design Rationale

The architecture combines the representational power of deep residual networks with channel-wise attention through SE blocks. Global Average Pooling allows each prediction head to access information from the entire image, making the model robust to spatial distortions and character displacement.

---

## Dataset

### Training Set

* 19,998 labeled training images
* Fixed-length six-character targets
* Grayscale image format
* Resolution standardized to 64 × 200 pixels

### Test Set

* 5,000 unlabeled images
* Same visual distribution as the training set

### Character Set

```text
23456789ABCDEFGHJKMNPQRSTUVWXYZ
```

Characters such as `0`, `1`, `I`, `L`, and `O` are intentionally excluded to avoid ambiguity.

---

## Data Augmentation

To improve generalization, lightweight image augmentations are applied during training:

* Random affine transformations
* Translation and scaling
* Brightness adjustments
* Contrast variation
* Mild Gaussian blur

Augmentations are performed on the GPU to minimize training overhead.

---

## Training Strategy

The notebook implements:

* Stratified training and validation workflow
* Character Error Rate (CER) monitoring
* Learning-rate scheduling
* Early stopping
* Model checkpointing
* Test-Time Augmentation (TTA)
* Ensemble inference

Two independently trained SE-ResNet models are combined through probability averaging to improve robustness and reduce prediction variance.

---

## Evaluation Metric

Performance is measured using **Character Error Rate (CER)**:

```text
CER = Edit Distance / Target Length
```

CER computes the minimum number of character insertions, deletions, and substitutions required to transform a prediction into the ground-truth label.

A custom implementation of the Levenshtein distance algorithm is used for evaluation.

---

## Results

The final ensemble achieves near-perfect validation performance.

| Model          | Validation CER |
| -------------- | -------------- |
| SE-ResNet A    | 0.00008        |
| SE-ResNet B    | 0.00000        |
| Ensemble + TTA | 0.00000        |

Additional analysis demonstrates:

* Strong per-position accuracy
* Well-calibrated prediction confidence
* Consistent character distributions between validation and test predictions
* Excellent robustness to blur, occlusion, and geometric distortion

---

## Repository Contents

```text
notebook_Siddhi_Jha_25117135.ipynb
│
├── Data loading and preprocessing
├── Exploratory Data Analysis (EDA)
├── Augmentation pipeline
├── SE-ResNet implementation
├── Training and validation
├── CER evaluation
├── Ensemble inference
├── Error analysis
└── Submission generation
```

---

## Running the Project

1. Download the dataset.
2. Place the files in the expected directory structure.
3. Open the notebook.
4. Execute all cells sequentially.

The notebook will automatically:

* Load and preprocess the data
* Train the models
* Evaluate validation performance
* Generate ensemble predictions
* Create the final submission file

---

## Key Contributions

* Reformulation of OCR as fixed-position multi-head classification
* Custom SE-ResNet architecture built entirely from scratch
* GPU-accelerated augmentation pipeline
* Ensemble-based inference with Test-Time Augmentation
* Near-zero Character Error Rate on validation data

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## Author

**Siddhi Jha**
Enrollment No. 25117135

Deep Learning for Distorted Visual Sequence Recognition.
