# Precipitation Nowcasting with U-Net

Deep learning model for radar-based precipitation nowcasting at **+30 min lead time** over Switzerland, trained on MeteoSwiss radar composites.

---

## Problem

Predicting where and how much it will rain 30 minutes ahead is a core challenge in operational meteorology. Classical optical-flow methods (e.g. pysteps) extrapolate current patterns but struggle with convective initiation and decay. This project trains a U-Net to learn the mapping directly from recent radar frames to a future frame.

Convolutional architectures are well-suited for this task because radar composites are inherently spatial: precipitation patterns have local structure and translational regularity that convolutions can exploit efficiently. The U-Net in particular combines an encoder branch — which progressively reduces spatial resolution while increasing feature depth to capture large-scale patterns — with a decoder branch that restores spatial resolution using skip connections from the encoder. This allows the model to simultaneously reason about broad precipitation systems and fine-scale local structure, which is critical for accurate spatial placement of rain at short lead times.

---

## Approach

| Component | Choice |
|---|---|
| Architecture | U-Net (encoder–decoder with skip connections) |
| Input | 3 radar frames (log1p-scaled reflectivity), 501×371 px |
| Output | 1 predicted frame at +30 min |
| Loss | Weighted L1 — upweights high-intensity pixels to counter class imbalance |
| Optimizer | Adam, lr=1e-4, weight decay=1e-4 |
| Regularisation | BatchNorm + Dropout2d (0.1) + early stopping (patience=10) |
| Augmentation | Random horizontal flip |

The weighted loss is defined as:

```
L = mean( (1 + R) * |pred - target| )
```

where `R = expm1(target)` is the rain rate in mm/10min, so heavy rain events contribute more to the gradient.

---

## Repository structure

```
├── nowcast_01_preprocessing.ipynb   # raw radar → npz samples
├── nowcast_02_enrich_split.ipynb    # temporal train/val/test split + metadata
├── nowcast_03_train.ipynb           # initial prototype (single-GPU, small data)
├── nowcast_04_train_all.py          # full training script (SLURM)
├── nowcast_05_test.py               # inference + metrics on held-out test set
├── radar_data/
│   ├── phase2_samples_meta.csv           # event metadata
│   └── phase2_samples_meta_enriched.csv  # metadata with train/val/test split
├── scripts/
│   ├── run_train.sh                 # SLURM job for training
│   └── run_test.sh                  # SLURM job for evaluation
└── figures/                         # sample output plots (committed)
```

---

## Results

Evaluation on held-out test set (`RUN_NAME = wl1_linear`):

| Metric | Value |
|---|---|
| MAE (mm/10min) | — |
| RMSE (mm/10min) | — |
| CSI @ 0.1 mm/10min | — |
| CSI @ 0.5 mm/10min | — |
| CSI @ 1.0 mm/10min | — |

> Fill in after running `nowcast_05_test.py` and add a sample figure to `figures/`.

---

## Quickstart

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate nowprecip
```

### 2. Prepare data

Place the following files in `radar_data/`:
- `phase2_samples.npz` — input/output radar stacks
- `phase2_samples_meta_enriched.csv` — metadata with `split` column

See `nowcast_01_preprocessing.ipynb` and `nowcast_02_enrich_split.ipynb` for how these are generated.

### 3. Train

Edit `RUN_NAME` in `nowcast_04_train_all.py`, then:

```bash
# Locally
python nowcast_04_train_all.py

# On a SLURM cluster (submit from project root)
sbatch scripts/run_train.sh
```

Checkpoints are saved to `checkpoints/best_model_<RUN_NAME>.pt`.

### 4. Evaluate

```bash
python nowcast_05_test.py   # RUN_NAME must match training
```

Metrics and plots are saved to `test_output/<RUN_NAME>/`.

---

## Environment

Tested on:
- Python 3.10
- PyTorch 2.x
- CUDA 12.x
- pysteps, numpy, pandas, matplotlib

---

## Data

MeteoSwiss radar composite data. Not publicly available — contact MeteoSwiss for access.
