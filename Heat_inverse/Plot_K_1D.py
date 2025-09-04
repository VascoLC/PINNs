import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def k_np(u):
    return 0.02 * np.exp(-20 * (u - 0.5)**2)

u_test = np.linspace(0, 1, 500) 
k_analytic = k_np(u_test)

#df = pd.read_csv("Train_data_1D.csv")
df = pd.read_csv("Heat_Inverse_1D_solution.csv")
u_field =  df[["u"]].values.astype(np.float32)
k_field = df[["k"]].values.astype(np.float32)

df2 = pd.read_csv("Heat_Inverse_1D_solution_NoBatching.csv")
u_field2 =  df2[["u"]].values.astype(np.float32)
k_field2 = df2[["k"]].values.astype(np.float32)


plt.figure(figsize=(5,4))
plt.plot(u_test, k_analytic, 'r-', lw=2, label="Analytical $k(u)$")
plt.plot(u_field, k_field, 'b-', lw=2, label="Data set batching")
plt.plot(u_field2,k_field2,'g-', lw=2, label="Full data set")
plt.xlabel("Temperature")
plt.ylabel("Condutivity")
plt.title("Analytical vs. Obtained Conductivity")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

