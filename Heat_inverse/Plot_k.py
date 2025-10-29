import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Heat_inverse_1D_FBatch_A_Pytorch_200k/Heat_Inverse_1D_solution_PyTorch.csv")
u  = df["u"].values.astype(np.float64)
k  = df["k"].values.astype(np.float64)

plt.figure(figsize=(6, 4))
plt.plot(u,k)
plt.xlabel("temperature")
plt.ylabel("conductivity")
plt.show()