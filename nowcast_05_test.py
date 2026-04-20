import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATA_DIR  = Path("/store_new/mch/msclim/antoumos/R/develop/NOWPRECIP/new_project/radar_data")
CKPT_DIR  = Path("/store_new/mch/msclim/antoumos/R/develop/NOWPRECIP/new_project/checkpoints")
OUT_DIR   = Path("/store_new/mch/msclim/antoumos/R/develop/NOWPRECIP/new_project/test_output")
OUT_DIR.mkdir(exist_ok=True, parents=True)

BATCH_SIZE = 8
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD        = (6, 7, 5, 6)

# ============================================================
# MODEL (must match training)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.enc1       = EncoderBlock(in_channels, features[0])
        self.enc2       = EncoderBlock(features[0], features[1])
        self.enc3       = EncoderBlock(features[1], features[2])
        self.enc4       = EncoderBlock(features[2], features[3])
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        self.dec4       = DecoderBlock(features[3] * 2, features[3])
        self.dec3       = DecoderBlock(features[3], features[2])
        self.dec2       = DecoderBlock(features[2], features[1])
        self.dec1       = DecoderBlock(features[1], features[0])
        self.final      = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        x = self.final(x)
        x = F.relu(x)
        return x[:, :, 5:506, 6:377]

# ============================================================
# LOAD DATA (test split)
# ============================================================
print("Loading data...")
data  = np.load(DATA_DIR / "phase2_samples.npz")
X_all = np.log1p(data["X"]).astype(np.float32)
Y_all = np.log1p(data["Y"]).astype(np.float32)

meta      = pd.read_csv(DATA_DIR / "phase2_samples_meta_enriched.csv")
test_mask = meta["split"] == "test"
X_test    = X_all[test_mask]
Y_test    = Y_all[test_mask]
print(f"Test samples: {len(X_test)}")

# Pad inputs
X_test_pad = torch.from_numpy(X_test)
X_test_pad = torch.stack([F.pad(x, PAD) for x in X_test_pad])
Y_test_t   = torch.from_numpy(Y_test)

loader = DataLoader(TensorDataset(X_test_pad, Y_test_t),
                    batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# LOAD MODEL
# ============================================================
model = UNet().to(DEVICE)
ckpt  = torch.load(CKPT_DIR / "best_model.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded best model from epoch {ckpt['epoch']} (val loss {ckpt['val_loss']:.6f})")

# ============================================================
# INFERENCE
# ============================================================
preds_log, trues_log = [], []

with torch.no_grad():
    for x_b, y_b in loader:
        pred = model(x_b.to(DEVICE)).cpu()
        preds_log.append(pred)
        trues_log.append(y_b)

preds_log = torch.cat(preds_log).numpy()   # (N, 1, 501, 371)
trues_log = torch.cat(trues_log).numpy()

# Convert to mm/h
preds_mmh = np.expm1(preds_log)
trues_mmh = np.expm1(trues_log)

# ============================================================
# METRICS
# ============================================================
mae  = np.mean(np.abs(preds_mmh - trues_mmh))
rmse = np.sqrt(np.mean((preds_mmh - trues_mmh) ** 2))
print(f"\nMAE  (mm/h): {mae:.4f}")
print(f"RMSE (mm/h): {rmse:.4f}")

pd.DataFrame({"mae": [mae], "rmse": [rmse]}).to_csv(OUT_DIR / "metrics.csv", index=False)

# ============================================================
# PLOT: first 4 test cases
# ============================================================
n_plot = min(4, len(preds_mmh))
vmax   = np.percentile(trues_mmh[:n_plot], 99)

fig, axes = plt.subplots(n_plot, 2, figsize=(8, 3 * n_plot))
for i in range(n_plot):
    axes[i, 0].imshow(trues_mmh[i, 0], vmin=0, vmax=vmax, cmap="Blues")
    axes[i, 0].set_title(f"Observed [{i}]")
    axes[i, 0].axis("off")
    axes[i, 1].imshow(preds_mmh[i, 0], vmin=0, vmax=vmax, cmap="Blues")
    axes[i, 1].set_title(f"Predicted [{i}]")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "test_cases.png", dpi=150)
print(f"\nPlot saved to {OUT_DIR / 'test_cases.png'}")
