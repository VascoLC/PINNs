import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Training_Loss.csv")
epoch = df[["epoch"]].values.astype(np.float32)
loss_u = df[["loss_u"]].values.astype(np.float32)

plt.figure(figsize=(5,7.5))
plt.plot(epoch, loss_u, label="Training Loss", linewidth=2)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Mean Squared Error Loss", fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.title("Loss History", fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()

