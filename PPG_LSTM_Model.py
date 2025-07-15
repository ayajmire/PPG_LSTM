# lstm_regression_pipeline.py

import os
import zipfile
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# === CONFIG ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ZIP_PATH = "Data File/0_subject.zip"
EXTRACT_PATH = "Data File/0_subject"
PPG_DIR = os.path.join(EXTRACT_PATH, "0_subject")
METADATA_PATH = "Data File/PPG-BP dataset.xlsx"
DOWNSAMPLE = True
TARGET_LENGTH = 210  # 2.1s at 100Hz
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.01
TRAIN_SPLIT = 0.8

# === EXTRACT ZIP ===
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

# === LOAD METADATA ===
df_meta = pd.read_excel(METADATA_PATH, skiprows=1)
df_meta['subject_ID'] = df_meta['subject_ID'].astype(str)

# === FILTER VALID SUBJECTS ===
subject_files = defaultdict(set)
for fname in os.listdir(PPG_DIR):
    if fname.endswith(".txt") and "_" in fname:
        sid, seg = fname.replace(".txt", "").split("_")
        subject_files[sid].add(seg)
valid_subjects = [sid for sid, segs in subject_files.items() if {'1', '2', '3'}.issubset(segs)]
df_valid = df_meta[df_meta['subject_ID'].isin(valid_subjects)].reset_index(drop=True)

# === DOWNSAMPLE & FILTER ===
def lowpass_filter(signal, fs=1000, cutoff=50):
    b, a = butter(4, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal)

# === LOAD DATA ===
X, y_sbp, y_dbp = [], [], []
for _, row in df_valid.iterrows():
    sid = row['subject_ID']
    for start in range(10):  # Create 10 samples per segment
        segments = []
        for i in range(1, 4):
            path = os.path.join(PPG_DIR, f"{sid}_{i}.txt")
            signal = np.loadtxt(path)
            if DOWNSAMPLE:
                signal = lowpass_filter(signal)
                signal = signal[start::10]  # Staggered sampling
            if len(signal) >= TARGET_LENGTH:
                signal = signal[:TARGET_LENGTH]
            else:
                padded = np.zeros(TARGET_LENGTH)
                padded[:len(signal)] = signal
                signal = padded
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
            segments.append(signal)
        if len(segments) == 3:
            X.append(np.stack(segments, axis=1))  # shape: [210, 3]
            y_sbp.append(row['Systolic Blood Pressure(mmHg)'])
            y_dbp.append(row['Diastolic Blood Pressure(mmHg)'])
X = np.array(X)
y_sbp = np.array(y_sbp)
y_dbp = np.array(y_dbp)

print("SBP Range:", np.min(y_sbp), np.max(y_sbp))
print("DBP Range:", np.min(y_dbp), np.max(y_dbp))

# Normalize labels
sbp_mean, sbp_std = y_sbp.mean(), y_sbp.std()
dbp_mean, dbp_std = y_dbp.mean(), y_dbp.std()
y_sbp = (y_sbp - sbp_mean) / sbp_std
y_dbp = (y_dbp - dbp_mean) / dbp_std
np.savez("bp_stats.npz", sbp_mean=sbp_mean, sbp_std=sbp_std, dbp_mean=dbp_mean, dbp_std=dbp_std)


# === DATASET CLASS ===
class PPGDataset(Dataset):
    def __init__(self, X, y_sbp, y_dbp):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_sbp = torch.tensor(y_sbp, dtype=torch.float32).unsqueeze(-1)
        self.y_dbp = torch.tensor(y_dbp, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_sbp[idx], self.y_dbp[idx]

# === LSTM REGRESSION MODEL ===
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_sbp = nn.Linear(hidden_size, 1)
        self.fc_dbp = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step
        sbp = self.fc_sbp(out)
        dbp = self.fc_dbp(out)
        return sbp, dbp

# === TRAINING LOOP ===
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_sbp_batch, y_dbp_batch in dataloader:
            optimizer.zero_grad()
            pred_sbp, pred_dbp = model(X_batch)
            if epoch == 0:
                print("Pred SBP:", pred_sbp[:5].squeeze().detach().numpy())
                print("True SBP:", y_sbp_batch[:5].squeeze().numpy())
            loss_sbp = criterion(pred_sbp, y_sbp_batch)
            loss_dbp = criterion(pred_dbp, y_dbp_batch)
            loss = loss_sbp + loss_dbp
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# === EVALUATION LOOP ===
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    preds_sbp, preds_dbp = [], []
    targets_sbp, targets_dbp = [], []
    with torch.no_grad():
        for X_batch, y_sbp_batch, y_dbp_batch in dataloader:
            pred_sbp, pred_dbp = model(X_batch)
            loss_sbp = criterion(pred_sbp, y_sbp_batch)
            loss_dbp = criterion(pred_dbp, y_dbp_batch)
            total_loss += (loss_sbp + loss_dbp).item()
            preds_sbp.extend(pred_sbp.squeeze().tolist())
            preds_dbp.extend(pred_dbp.squeeze().tolist())
            targets_sbp.extend(y_sbp_batch.squeeze().tolist())
            targets_dbp.extend(y_dbp_batch.squeeze().tolist())
    avg_loss = total_loss / len(dataloader)
    print(f"\nTest Loss: {avg_loss:.4f}")

    # Denormalize predictions
    preds_sbp = [p * sbp_std + sbp_mean for p in preds_sbp]
    preds_dbp = [p * dbp_std + dbp_mean for p in preds_dbp]
    targets_sbp = [t * sbp_std + sbp_mean for t in targets_sbp]
    targets_dbp = [t * dbp_std + dbp_mean for t in targets_dbp]


    return preds_sbp, preds_dbp, targets_sbp, targets_dbp

# === PLOT RESULTS ===
def plot_predictions(pred, true, label):
    plt.figure(figsize=(6, 4))
    plt.scatter(true, pred, alpha=0.6)
    plt.plot([min(true), max(true)], [min(true), max(true)], 'r--')
    plt.xlabel(f"True {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(f"{label} Prediction")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === RUN ===
dataset = PPGDataset(X, y_sbp, y_dbp)
train_size = int(TRAIN_SPLIT * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = LSTMRegressor()
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

train_model(model, train_loader, optimizer, criterion)



# After training
model.eval()
train_preds_sbp, train_preds_dbp = [], []
train_true_sbp, train_true_dbp = [], []
with torch.no_grad():
    for X_batch, y_sbp_batch, y_dbp_batch in train_loader:
        p_sbp, p_dbp = model(X_batch)
        train_preds_sbp.extend(p_sbp.squeeze().tolist())
        train_preds_dbp.extend(p_dbp.squeeze().tolist())
        train_true_sbp.extend(y_sbp_batch.squeeze().tolist())
        train_true_dbp.extend(y_dbp_batch.squeeze().tolist())

print("Train R² SBP:", r2_score(train_true_sbp, train_preds_sbp))
print("Train R² DBP:", r2_score(train_true_dbp, train_preds_dbp))




preds_sbp, preds_dbp, true_sbp, true_dbp = evaluate_model(model, test_loader, criterion)
plot_predictions(preds_sbp, true_sbp, "SBP")
plot_predictions(preds_dbp, true_dbp, "DBP")



# === METRICS ===
true_sbp = np.array(true_sbp)
preds_sbp = np.array(preds_sbp)
true_dbp = np.array(true_dbp)
preds_dbp = np.array(preds_dbp)

print("\nSBP Metrics:")
print("MAE:", mean_absolute_error(true_sbp, preds_sbp))
print("RMSE:", np.sqrt(mean_squared_error(true_sbp, preds_sbp)))
print("R²:", r2_score(true_sbp, preds_sbp))

print("\nDBP Metrics:")
print("MAE:", mean_absolute_error(true_dbp, preds_dbp))
print("RMSE:", np.sqrt(mean_squared_error(true_dbp, preds_dbp)))
print("R²:", r2_score(true_dbp, preds_dbp))

# === SAVE MODEL ===
torch.save(model.state_dict(), "lstm_bp_model.pth")