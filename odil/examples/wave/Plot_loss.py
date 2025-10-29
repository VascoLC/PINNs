import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow import keras

plot_configs = [
    {
        'path': 'C:\PINNs_Git\PINNs\odil\examples\wave\out_wave_adam/train.csv',
        'color': 'red',
        'label': 'ODIL: ADAM'
    },
    {
        'path': 'C:\PINNs_Git\PINNs\odil\examples\wave\out_wave_adam_Experimenting/train.csv',
        'color': 'orange',
        'label': 'mODIL: ADAM '
    },
    {
        'path': 'C:\PINNs_Git\PINNs\odil\examples\wave\out_wave_lbfgsb/train.csv',
        'color': 'green',
        'label': 'ODIL: LBFGSB '
    },
    {
        'path': 'C:\PINNs_Git\PINNs\odil\examples\wave\out_wave_lbfgsb_Experimenting/train.csv',
        'color': 'blue',
        'label': 'mODIL: LBFGSB '
    },
        {
        'path': 'C:\PINNs_Git\PINNs\Wave\Wave_Results_Fourier_40k/loss_history.csv',
        'color': 'purple',
        'label': 'PINN: STMsFFN ADAM '
    },
    {
        'path': 'C:\PINNs_Git\PINNs\Wave\Wave_Results_NoFourier/loss_history_NoF.csv',
        'color': 'brown',
        'label': 'PINN: ADAM '
    }    
]


'''
    {
        'path': 'C:\PINNs_Git\PINNs\Wave\Wave_Results_Fourier_40k/loss_history.csv',
        'color': 'purple',
        'label': 'PINN: STMsFFN ADAM '
    },
    {
        'path': 'C:\PINNs_Git\PINNs\Wave\Wave_Results_NoFourier/loss_history_NoF.csv',
        'color': 'brown',
        'label': 'PINN: ADAM '
    }    
'''


plt.figure(figsize=(5, 4))
ax = plt.gca()

for config in plot_configs:

    df = pd.read_csv(config['path'])
    epochs = df[['epoch']].values.astype(np.float32)
    #loss_k = df[['error_k']].values.astype(np.float32)
    loss_u = df[['error_u']].values.astype(np.float32)

    # Filter the DataFrame to get every 10th epoch
    #df_filtered = df[(df['epoch'] <= 100) | (df['epoch'] % 500 == 0)]
    # Extract epochs and loss_k from the filtered DataFrame
    #epochs = df_filtered["epoch"].values.astype(np.float32)
    #loss_u = df_filtered["loss_u"].values.astype(np.float32)

    if config['path'] == 'C:\PINNs_Git\PINNs\Wave\Wave_Results_Fourier_40k/loss_history.csv':
        loss_u = np.sqrt(loss_u)

    if config['path'] == 'C:\PINNs_Git\PINNs\Wave\Wave_Results_NoFourier/loss_history_NoF.csv':
        loss_u = np.sqrt(loss_u)
    
    plt.plot(epochs, loss_u, linewidth=2, color=config['color'], label=config['label'])

plt.legend()
plt.xlabel("epoch")
plt.ylabel("Error")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.xlim([0, 1e5])
plt.ylim([1e-3, 1e0])
plt.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
