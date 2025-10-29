import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_csv("Heat_inverse_1D_FBatch_A_Pytorch_200k/training_loss_log_PyTorch.csv")

# --- Extract columns as numpy arrays ---
epoch  = df["epoch"].values.astype(np.float32)
loss   = np.sqrt(df["loss_k"].values.astype(np.float32))/0.02

# --- Downsample every 250 epochs ---
def downsample(e, l, step=750):
    return e[::step], l[::step]

#epoch, loss = downsample(epoch, loss)
# --- Plot ---
plt.figure(figsize=(6, 4))
plt.plot(epoch, loss, 'r-', lw=2, label="PINN: 5 Data Batches (Adam)")
plt.xlabel("epoch")
plt.ylabel("temperature error")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.xlim([0, 1e6])
plt.ylim([1e-3, 1])
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
