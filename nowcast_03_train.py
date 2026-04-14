# nowcast_test.py
# Goal: verify the full pipeline works before committing to full training
 
# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
 
# ============================================================
# 1 -- DEVICE CHECK
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
 
# ============================================================
# 2 -- LOAD A SMALL SUBSET OF DATA
# ============================================================
DATA_DIR = Path("/store_new/mch/msclim/antoumos/R/develop/NOWPRECIP/new_project/radar_data")
 
print("\nLoading data...")
data  = np.load(DATA_DIR / "phase2_samples.npz")
X_all = np.log1p(data["X"]).astype(np.float32)
Y_all = np.log1p(data["Y"]).astype(np.float32)
 
# Take only first 64 samples for testing
X_small = X_all[:64]
Y_small = Y_all[:64]
print(f"X_small: {X_small.shape}")
print(f"Y_small: {Y_small.shape}")
 
# ============================================================
# 3 -- DATASET + DATALOADER
# ============================================================
PAD = (6, 7, 5, 6)  # (left, right, top, bottom) → 371→384, 501→512
 
class RadarDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, i):
        x = F.pad(self.X[i], PAD)
        y = F.pad(self.Y[i], PAD)
        return x, y
 
dataset = RadarDataset(X_small, Y_small)
loader  = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
 
# Check one batch
x_batch, y_batch = next(iter(loader))
print(f"\nX batch: {x_batch.shape}")  # (4, 3, 512, 384)
print(f"Y batch: {y_batch.shape}")   # (4, 1, 512, 384)
 
# ============================================================
# 4 -- MODEL
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
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
        x = F.relu(x)                    # ensure non-negative predictions
        return x[:, :, 5:506, 6:377]    # crop back to (501, 371)
 
model = UNet().to(DEVICE)
print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
 
# ============================================================
# 5 -- FORWARD PASS CHECK
# ============================================================
x_batch = x_batch.to(DEVICE)
y_batch = y_batch.to(DEVICE)
y_batch = y_batch[:, :, 5:506, 6:377]  # crop y back to (501, 371)
 
with torch.no_grad():
    pred = model(x_batch)
 
print(f"\nPred shape: {pred.shape}")    # (4, 1, 501, 371)
print(f"Pred min:   {pred.min():.4f}")
print(f"Pred max:   {pred.max():.4f}")
print(f"Any NaN:    {torch.isnan(pred).any()}")
 
# ============================================================
# 6 -- LOSS CHECK
# ============================================================
criterion = nn.MSELoss()
loss = criterion(pred, y_batch)
print(f"\nInitial MSE loss: {loss.item():.6f}")
 
# ============================================================
# 7 -- MINI TRAINING RUN (3 epochs)
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
 
print("\nMini training run (3 epochs)...")
for epoch in range(3):
    model.train()
    epoch_loss = 0.0
 
    for x_b, y_b in loader:
        x_b = x_b.to(DEVICE)
        y_b = y_b[:, :, 5:506, 6:377].to(DEVICE)  # crop y
 
        optimizer.zero_grad()
        pred = model(x_b)
        loss = criterion(pred, y_b)
        loss.backward()
        optimizer.step()
 
        epoch_loss += loss.item()
 
    print(f"Epoch {epoch+1}/3  Loss: {epoch_loss/len(loader):.6f}")
 
print("\nAll checks passed! Ready for full training.")
 