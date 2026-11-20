# Lightweight Swin-Mamba U-Net for Lung Nodule Segmentation

A 3.08 M-parameter hybrid **Swin-Transformer + Mamba** U-Net that segments lung nodules on **full 512×512 chest CT slices** from LIDC-IDRI. U-Net and Swin-UNet are retrained under the same pipeline for comparison.

**Project page:** https://rrdhharun2005-ai.github.io/swin-mamba-lung-nodule/

## Results (test set: 105 unseen patients, 2,558 full slices, 1,284 without a nodule)

| Model | Params | Dice | Precision | Recall | False alarms |
|---|---|---|---|---|---|
| **Swin-Mamba 2.5D + TTA (final)** | **3.08 M** | **0.723** | **0.791** | 0.665 | **6.9%** |
| Swin-Mamba 2.5D (raw) | 3.08 M | 0.708 | 0.744 | 0.676 | 12.1% |
| Swin-Mamba 2D | 3.08 M | 0.669 | 0.727 | 0.620 | 16.0% |
| U-Net 2D | 31.04 M | 0.696 | 0.745 | 0.652 | 14.4% |
| Swin-UNet 2D | 41.39 M | 0.670 | 0.715 | 0.631 | 18.2% |

False alarms: share of nodule-free slices where the model predicts 10 or more nodule pixels. All settings were chosen on the validation set; the test set was evaluated once.

## Model

- **Conv stem:** two stride-2 3×3 convolutions (512 → 128).
- **Swin stages (1/4, 1/8):** 8×8 window self-attention (W-MSA, then SW-MSA).
- **VSS / Mamba stages (1/16, 1/32):** four-direction selective scan (SS2D) for global context at linear cost, written in plain PyTorch.
- **Decoder:** bilinear upsampling, skip connections and convolution blocks.
- **2.5D input:** the slice plus its neighbours about 2 mm above and below, as 3 channels.

## Data

LIDC-IDRI, 700 patients, 50% radiologist-consensus masks, lung window [−1000, 400] HU. Patient-wise split 70/15/15 (490 / 105 / 105 patients, 18,362 slices). The dataset is not included; download it from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/lidc-idri/).

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Main pipeline: `model.py` (Swin-Mamba U-Net), `train.py`, `data.py`, `metrics.py`, `baselines.py`, `evaluate.py`, data download and preprocessing |
| `pseudo final/code/` | Final inference pipeline (`pipeline.py`: TTA, blob removal) and Phase 2 experiments (fine-tuning, ensemble, refiner) |
| `pseudo final/results/` | Result tables (CSV), logs and ablation charts |
| `presentation/figures/` | Architecture, training-curve and result figures |
| `docs/` | Project page (GitHub Pages) |

## Run

```bash
pip install -r code/requirements.txt
python code/download_lidc.py          # download LIDC-IDRI scans
python code/preprocess_lidc.py        # slices, masks, patient-wise split
python code/make_25d.py               # 2.5D (3-slice) inputs
python code/train.py                  # train Swin-Mamba U-Net
python "pseudo final/code/predict.py" # cache predictions
python "pseudo final/code/pipeline.py" # tune on validation, evaluate test once
```

Trained on one NVIDIA RTX 4050 Laptop GPU (6 GB).

## Authors

Balla Sai Mahesh, Madesh K K, Shachin D A, R R Dhharun. Supervisor: Dr. Vivek Patel.
School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Delhi NCR, Faridabad.
