# AI Face Mask Detector

A CNN-based face mask classifier that detects whether a person is wearing a **Cloth**, **N95**, or **Surgical** mask — or no mask at all.

This is a 2026 revamped version of a group assignment originally completed in 2022 for the Introduction to AI course (COMP 472) at Concordia University. The model, dataset, and application have been significantly improved from the original submission.

🔗 **Live Demo:** [huggingface.co/spaces/samuelchuang/face-mask-detector](https://huggingface.co/spaces/samuelchuang/face-mask-detector)

---

## Accuracy Improvements

The original assignment achieved ~51% accuracy on a dataset of ~1,600 images. The following experiments were conducted to improve performance:

| Setup | Accuracy | Notes |
|---|---|---|
| Original dataset (~1,600 imgs, 4 epochs) | 51% | Baseline |
| Original dataset (30 epochs) | 58.5% | Ceiling hit regardless of epochs |
| Original dataset + data augmentation | 60.3% | Augmentation alone not enough |
| Expanded dataset (~10,730 imgs, Mendeley) | 92.5% | Subject overlap caused overfitting |
| Manually curated ~3,273 imgs + augmentation | 76.9% | Better generalization |
| Deduplication (perceptual hashing) | 72.3% | Backfired — too few images remaining |
| **Final: ~6,071 imgs + augmentation (16 epochs)** | **79.6%** | Best balance of accuracy and generalization |

### Key Findings
- **Data quantity** had the largest impact on accuracy — more than epochs or augmentation alone
- **Subject overlap** (same people appearing across images) caused severe overfitting with the Mendeley dataset
- **Data augmentation** (random flip, rotation, color jitter) improved generalization but couldn't substitute for more data
- **Deduplication** via perceptual hashing (dHash, threshold ≤ 5) removed near-duplicate and cross-class images, but hurt performance at the ~3,000 image scale
- **N95 vs Surgical** confusion was the hardest problem — nearly identical appearance at 64×64 resolution

### Final Model
- **Architecture:** Custom CNN — 2 conv blocks (BatchNorm + LeakyReLU + MaxPool) + 3 FC layers
- **Input size:** 64×64 RGB
- **Training:** 16 epochs, Adam optimizer, CrossEntropyLoss, batch size 32
- **Valid loss at convergence:** 0.61

---

## Files

| File | Description |
|---|---|
| `FaceMaskCNN.py` | CNN architecture definition |
| `CNNSk.py` | Training script with Skorch, confusion matrix, and metrics |
| `CNN.py` | Alternative training script (standard PyTorch) |
| `SignleImage.py` | Single image inference |
| `app.py` | Gradio web app |
| `deduplicate.py` | Perceptual hash deduplication tool |
| `MeanAndStd.py` | Compute dataset normalization statistics |

---

## Running Locally

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Train:**
```bash
python CNNSk.py
```

**Web app:**
```bash
python app.py
```

**Single image inference:**
```bash
python SignleImage.py
```

---

## Dataset Sources

- Original dataset: project team (2022)
- Cloth/N95/Surgical: [Mendeley Face Mask Wearing Image Dataset](https://data.mendeley.com/datasets/8pn3hg99t4/2) (CC BY 4.0)
- NoMask: [spandanpatnaik09](https://www.kaggle.com/datasets/spandanpatnaik09/face-mask-detectormask-not-mask-incorrect-mask) + [ashishjangra27](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) (CC0)
