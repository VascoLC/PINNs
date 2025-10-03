import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow import keras

plot_configs = [
    {
        'path': 'C:\PINNs_Git\PINNs\Heat_inverse\Heat_Inverse_1D_FBatch_Nk/training_loss_log_1D_solution_NoBatching_NoK.csv',
        'color': 'red',
        'label': 'PINN: Full Data Set'
    },
    {
        'path': 'C:\PINNs_Git\PINNs\Heat_inverse\Heat_inverse_1D_5Batch_Nk/training_loss_log_1D_solution_Batching_NoK.csv',
        'color': 'green',
        'label': 'PINN: 5 Data Batches'
    }
]


plt.figure(figsize=(5, 4))
ax = plt.gca()

for config in plot_configs:

    df = pd.read_csv(config['path'])

    # Filter the DataFrame to get every 10th epoch
    df_filtered = df[(df['epoch'] <= 100) | (df['epoch'] % 500 == 0)]

    # Extract epochs and loss_k from the filtered DataFrame
    epochs = df_filtered["epoch"].values.astype(np.float32)
    loss_k = df_filtered["loss_k"].values.astype(np.float32)
    
    plt.plot(epochs, np.sqrt(loss_k)/0.02, linewidth=2, color=config['color'], label=config['label'])

plt.legend()
plt.xlabel("epoch")
plt.ylabel("Conductivity Loss")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.xlim([0, 1e6])
plt.ylim([10e-4, 10e-0])
plt.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
