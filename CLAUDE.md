# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CNN-based face mask classifier (academic project for COMP 472). Classifies images into 4 categories: **Cloth**, **N95**, **NoMask**, **Surgical**.

## Environment Setup

Requires a conda environment with:
```
torch, torchvision, matplotlib, pandas, skorch, sklearn, Pillow
```

## Commands

**Compute dataset normalization stats:**
```bash
python MeanAndStd.py
```

**Train (standard PyTorch):**
```bash
python CNN.py
```

**Train (Skorch — preferred, generates confusion matrix + metrics):**
```bash
python CNNSk.py
```

**Run inference on a single image:**
```bash
python SignleImage.py
```

**Evaluate a saved model:**
```bash
python test.py
```

## Architecture

**`FaceMaskCNN`** (`FaceMaskCNN.py`): Custom `nn.Module`
- 2 conv blocks: `Conv2d → BatchNorm2d → LeakyReLU → MaxPool2d` (3→32→64 channels)
- 3 FC layers: 16384 → 1024 → 512 → 6 (output is 6 logits despite 4 active classes)
- Dropout(0.1) before FC layers

**Training config** (shared across `CNN.py` and `CNNSk.py`):
- Input size: 64×64 RGB
- Normalization mean: `(0.5211, 0.4858, 0.4651)`, std: `(0.2889, 0.2824, 0.2880)`
- Split: 1200 train / 400 test, batch size 32, 4 epochs, Adam lr=0.001
- Loss: CrossEntropyLoss

**Model files** saved to `./models/`. Scripts reference different checkpoints:
- `CNN.py` → `model30.pth`
- `CNNSk.py` → `model4sk.pth`
- `SignleImage.py` / `test.py` → `model60.pth`

## Dataset Structure

```
dataset/          # Primary dataset (Cloth/, N95/, NoMask/, Surgical/)
AgeBasedDataset/  # Same 4 classes, split by age group (bias analysis)
bias-gender/      # Same 4 classes, split by gender (bias analysis)
singleImages/     # Sample images for inference testing
```

Class label mapping used in inference: `["Cloth", "N95", "NoMask", "Surgical"]`
