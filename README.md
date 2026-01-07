# Pretraining → Fine-tuning (Transfer Learning) in PyTorch

This project is a **single, self-contained Jupyter notebook** that demonstrates a full workflow for:
1) **Pretraining** a CNN backbone on **CIFAR-10**, then  
2) **Fine-tuning (transfer learning)** the pretrained backbone on **EMNIST Letters (A–Z)**.

It also includes a small **architecture/training ablation** (batch size, learning rate, batch norm, dropout, and global pooling) and basic error analysis.

---

## What the model does

### Pretraining task (CIFAR-10)
- **Input:** 32×32 RGB images (3 channels)
- **Output:** 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Goal:** learn general-purpose visual features (edges → textures → shapes)

### Fine-tuning task (EMNIST Letters)
- **Input:** grayscale handwritten letters (1 channel)
- **Output:** 26 classes (A–Z)
- **Goal:** reuse pretrained features and adapt the classifier + later layers to a new dataset

---

## Model architecture (from the notebook)

### Backbone: residual-style CNN blocks
- `ConvBlock`: Conv → BN → ELU → Dropout2d → Conv → BN with a **skip/shortcut**
- Stacked stages (64 → 128 → 256 channels)
- Global pooling + MLP classifier head during pretraining
- During fine-tuning:
  - The first conv layer is adapted from **3-channel → 1-channel** by averaging pretrained RGB weights into grayscale weights.
  - Classifier head is replaced for **26 classes**.
  - Earlier layers are **frozen**, later stage(s) + head remain trainable.

---

## Data & preprocessing

### CIFAR-10 (pretraining)
- Split used in notebook:
  - **Train:** 45,000
  - **Validation / “finetune split”:** 5,000
  - **Test:** 10,000
- Augmentations used:
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip
  - RandAugment
  - Normalize (dataset mean/std computed in notebook)
  - RandomErasing (p=0.10)

### EMNIST Letters (fine-tuning)
- **Train:** 124,800
- **Test:** 20,800
- Normalization: mean=0.5, std=0.5 (as used in notebook)
- Labels are shifted from [1..26] → [0..25] inside training loop.

---

## Training setup (pretraining)

- Epochs: **55**
- Optimizer: **SGD** (momentum=0.9, weight_decay=5e-4)
- Loss: **CrossEntropyLoss** with **label smoothing (0.1)**
- LR schedule:
  - **Linear warmup** (first 5 epochs)
  - then **Cosine annealing**
- Checkpointing:
  - Saves `model_checkpoint_epoch_<N>.pth` each epoch

---

## Results (from notebook outputs)

### CIFAR-10 pretraining
- **Best validation accuracy:** **76.40%**
- **Test accuracy:** **76.43%**
- Misclassifications collected: **2357** examples

### EMNIST fine-tuning (transfer learning)
- Pretrained checkpoint loaded: `model_checkpoint_epoch_30.pth`
- Fine-tuning epochs: **15**
- Optimizer: **Adam** on trainable parameters (lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau
- **Best EMNIST test accuracy:** **93.72%**
- Saved model: `best_emnist_finetuned.pth`

---

## Ablation / experiments included
The notebook runs a small sweep over:
- batch size
- learning rate
- batch norm on/off
- dropout on/off
- global pooling on/off

It prints:
- parameter counts (pooling vs no pooling)
- validation accuracy after short training runs
- top configurations by validation accuracy

---

## How to run

### Option A — Run locally
1. Clone the repo
   ```bash
   git clone https://github.com/maheenrazza/cifar10-pretraining-emnist-finetuning.git
   cd cifar10-pretraining-emnist-finetuning

2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Launch Jupyter
   ```bash
   jupyter notebook

4. Open and run:
   ```bash
    pretraining_and_finetuning.ipynb

### Option B — Run in Google Colab
Upload the notebook to Colab and run cells top-to-bottom. (GPU recommended.)

---

### Repository contents

- pretraining_and_finetuning.ipynb — main notebook (pretraining + fine-tuning + experiments)
- requirements.txt — Python dependencies
- .gitignore — ignores notebook checkpoints, caches, etc.

### Notes / limitations
  
- This is an educational workflow and intentionally keeps everything in one notebook.
- CIFAR-10 performance depends on GPU availability and random seed.
- For stronger comparisons, you could add:
  - a baseline EMNIST model trained from scratch
  - full confusion matrix + per-class accuracy
  - multiple runs with mean/std reporting
