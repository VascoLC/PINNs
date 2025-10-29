import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow import keras

plot_configs = [
    {
        'path': 'C:\PINNs_Git\PINNs\Poisson\Poisson_Fourier_100k_k2/loss_history.csv',
        'color': 'red',
        'label': 'PINN: MsFFN ADAM'
    },
        {
        'path': 'C:\PINNs_Git\PINNs\Poisson\Poisson_variants/loss_history.csv',
        'color': 'orange',
        'label': 'PINN: Full Batch ADAM '
    }
]


plt.figure(figsize=(5, 4))
ax = plt.gca()

for config in plot_configs:

    df = pd.read_csv(config['path'])
    epochs = df[['step']].values.astype(np.float32)
    #loss_k = df[['error_k']].values.astype(np.float32)
    loss_u = df[['loss_q']].values.astype(np.float32)

    # Filter the DataFrame to get every 10th epoch
    #df_filtered = df[(df['epoch'] <= 100) | (df['epoch'] % 500 == 0)]
    # Extract epochs and loss_k from the filtered DataFrame
    #epochs = df_filtered["epoch"].values.astype(np.float32)
    #loss_u = df_filtered["loss_u"].values.astype(np.float32)

    loss_u = np.sqrt(loss_u)
    
    plt.plot(epochs, loss_u, linewidth=2, color=config['color'], label=config['label'])

plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.xlim([0, 1e6])
plt.ylim([10e-4, 10e2])
plt.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
