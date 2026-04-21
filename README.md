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

## Dataset

The final training dataset (~6,071 images) was manually curated from multiple public sources. Images were selectively pulled from each — not used wholesale — to balance class sizes and reduce subject overlap.

| Source | Platform |
|---|---|
| [coffee124/facemaskn95](https://www.kaggle.com/datasets/coffee124/facemaskn95) | Kaggle |
| [shiekhburhan/face-mask-dataset](https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset) | Kaggle |
| Google Images | Web |
| [Face Mask Wearing Image Dataset](https://data.mendeley.com/datasets/8pn3hg99t4/2) (v2, 2023) | Mendeley Data — tested, removed due to overfitting |

**Why Mendeley was removed:** The Mendeley dataset contains 24,916 images but many are of the same subjects photographed multiple times. When included, the model achieved 92.5% validation accuracy but failed to generalize — it was memorizing individuals, not mask types. Removing it and curating a smaller, more diverse set brought the valid loss down significantly.

---

## Limitations

- **Low input resolution (64×64):** The model classifies images after downscaling to 64×64 pixels. Fine-grained visual details are lost, which is the primary reason N95 and Surgical masks are frequently confused — they look nearly identical at this resolution.
- **Single-person images only:** The model is not a face detector. It classifies the entire image as a single prediction. Multi-person scenes or images where the face is small, occluded, or off-center will produce unreliable results.
- **Mask type, not fit or coverage:** The model identifies the type of mask present, not whether it is worn correctly. A mask pulled below the nose or chin may still be classified as "masked."
- **Dataset bias:** Training data was sourced from publicly available datasets that skew toward certain demographics, lighting conditions, and camera angles. Performance may degrade on images that differ significantly from the training distribution.
- **Static classifier:** The model does not process video or real-time streams. Each image is classified independently with no temporal context.
- **Limited dataset availability:** High-quality, diverse, labeled face mask datasets are scarce. Most publicly available datasets either suffer from subject overlap (same individuals appearing across many images, causing overfitting), class imbalance, or low variety in backgrounds, lighting, and demographics. This made it difficult to grow the dataset beyond ~6,000 images without reintroducing the overfitting problems seen with the larger Mendeley dataset.

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

