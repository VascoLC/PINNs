import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow import keras

# Conductivity vs Temperature
#df = pd.read_csv("2D_second_step/fields_over_time/field_at_t_0.0.csv")
df = pd.read_csv("2D_first_step/Sim2/fields_over_time/field_at_t_0.0.csv")
df_t1 = df[np.isclose(df["t"], 0)]

u_field = df[["u"]].values.astype(np.float32)
k_field = df[["k"]].values.astype(np.float32)

def k_np(u):
    return 0.02 * np.exp(-20 * (u - 0.5)**2)

u_test = np.linspace(0, 1, 500) 
k_analytic = k_np(u_test)

plt.figure(figsize=(5,4))
#plt.plot(u_test, k_analytic, 'r-', lw=2, label="Analytical $k(u)$")
plt.scatter(u_field, k_field, s=10, alpha=0.6)
plt.xlabel("Temperature")
plt.ylabel("Condutivity")
plt.title(f"K vs U")
plt.grid(True)
plt.tight_layout()
plt.show()

'''
# Training Loss
df = pd.read_csv("training_loss_log.csv")
epochs =  df[["epochs"]].values.astype(np.float32)
total_loss = df[["total_loss"]].values.astype(np.float32)

plt.figure(figsize=(5,4))
plt.scatter(epochs, total_loss, s=10, alpha=0.6)
plt.xlabel(""); plt.ylabel("Loss")
plt.title(f"Total Loss")
plt.grid(True)
plt.tight_layout()

plt.show()
'''
