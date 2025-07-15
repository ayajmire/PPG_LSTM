import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# --- Define your model architecture ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_sbp = nn.Linear(hidden_size, 1)
        self.fc_dbp = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        sbp = self.fc_sbp(out)
        dbp = self.fc_dbp(out)
        return sbp, dbp

# --- Load your CSV data ---
df = pd.read_csv("Validation5_PPG.csv")
df = df.sort_values(by="time")

time_ms = df['time'].values
r_signal = df['R'].values

# --- Resample to 100 Hz (every 10 ms) ---
target_times = np.arange(time_ms[0], time_ms[-1], 10)
interp_fn = interp1d(time_ms, r_signal, kind='linear', fill_value='extrapolate')
r_resampled = interp_fn(target_times)

# --- Extract 3 segments of 210 samples each ---
if len(r_resampled) < 3 * 210:
    raise ValueError("Not enough samples to extract 3×210-length segments.")

segments = []
for i in range(3):
    start = i * 210
    segment = r_resampled[start:start + 210]
    segment = (segment - np.mean(segment)) / np.std(segment)
    segments.append(segment)

# --- Stack into (1, 210, 3) tensor ---
data = np.stack(segments, axis=-1)  # (210, 3)
data = data[np.newaxis, ...]        # (1, 210, 3)
data_tensor = torch.tensor(data, dtype=torch.float32)

# --- Load model and weights ---
model = LSTMRegressor()
model.load_state_dict(torch.load("lstm_bp_model.pth", map_location='cpu'))
model.eval()

# --- Run inference ---
with torch.no_grad():
    pred_sbp, pred_dbp = model(data_tensor)

# === Denormalize ===
# Load in validation
stats = np.load("bp_stats.npz")
sbp_mean = stats['sbp_mean'].item()
sbp_std = stats['sbp_std'].item()
dbp_mean = stats['dbp_mean'].item()
dbp_std = stats['dbp_std'].item()

sbp_raw = pred_sbp.item() * sbp_std + sbp_mean
dbp_raw = pred_dbp.item() * dbp_std + dbp_mean

print(f"✅ SBP Prediction (mmHg): {sbp_raw:.2f}")
print(f"✅ DBP Prediction (mmHg): {dbp_raw:.2f}")

