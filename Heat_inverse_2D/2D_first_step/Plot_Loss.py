import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_csv("Mini_Batching_ADAM_Final/training_loss_log.csv")
df2 = pd.read_csv("Full_Batch_ADAM/training_loss_log.csv")
df3 = pd.read_csv("Hybrid_LA_Final/training_loss_log_combined.csv")
df4 = pd.read_csv("Full_Batch_LBFGSB/training_loss_log_lbfgs.csv")

# --- Extract columns as numpy arrays ---
epoch  = df["epoch"].values.astype(np.float32)
loss   = np.sqrt(df["loss_u"].values.astype(np.float32))    
epoch_2 = df2["epoch"].values.astype(np.float32)
loss_2  = np.sqrt(df2["loss_u"].values.astype(np.float32))
epoch_3 = df3["epoch"].values.astype(np.float32)
loss_3  = np.sqrt(df3["loss_u"].values.astype(np.float32))
epoch_4 = df4["epoch"].values.astype(np.float32)
loss_4  = np.sqrt(df4["loss_u"].values.astype(np.float32))

# --- Downsample every 250 epochs ---
def downsample(e, l, step=250):
    return e[::step], l[::step]

epoch, loss = downsample(epoch, loss)
epoch_2, loss_2 = downsample(epoch_2, loss_2)
epoch_3, loss_3 = downsample(epoch_3, loss_3)
epoch_4, loss_4 = downsample(epoch_4, loss_4)

# --- Plot ---
plt.figure(figsize=(6, 4))
plt.plot(epoch, loss, color = 'blue', lw=2, label="PINN: 5 Data Batches (Adam)")
plt.plot(epoch_2, loss_2, color = 'green', lw=2,label="PINN: Full Data Set (Adam)")
plt.plot(epoch_3, loss_3, color = 'red',  lw=2,label="PINN: Full Data Set (ADAM-LBFGSB)")
plt.plot(epoch_4, loss_4, color = 'orange',  lw=2,label="PINN: Full Data Set (LBFGSB)")
plt.xlabel("epoch")
plt.ylabel("temperature error")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.xlim([0, 1e6])
plt.ylim([1e-3, 1])
plt.legend()
plt.grid()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
