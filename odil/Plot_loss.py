import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow import keras

df = pd.read_csv("train.csv")
epochs =  df[["epochs"]].values.astype(np.float32)
total_loss = df[["arror_u"]].values.astype(np.float32)

plt.figure(figsize=(5,4))
plt.scatter(epochs, total_loss, s=10, alpha=0.6)
plt.xlabel(""); plt.ylabel("Loss")
plt.title(f"Total Loss")
plt.grid(True)
plt.tight_layout()

plt.show()