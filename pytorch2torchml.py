# pytorch2coreml.py

import torch
import coremltools as ct
from PPG_LSTM_Model import LSTMRegressor  # Make sure the class is accessible

# === Step 1: Recreate the Model Structure ===
model = LSTMRegressor()
model.load_state_dict(torch.load("lstm_bp_model.pth", map_location=torch.device('cpu')))
model.eval()

# === Step 2: Create Dummy Input ===
# Shape: (batch_size=1, sequence_length=210, input_features=3)
example_input = torch.randn(1, 210, 3)

# === Step 3: Trace the Model ===
traced_model = torch.jit.trace(model, example_input)

# === Step 4: Convert to Core ML ===
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="ppg_input", shape=example_input.shape)],
)

# === Step 5: Save Core ML Model ===
coreml_model.save("PPG_BP_Estimator.mlpackage")
print("✅ Core ML model saved as PPG_BP_Estimator.mlmodel")
