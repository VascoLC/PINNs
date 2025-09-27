import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow import keras

plot_configs = [
    {
        'path': 'out_heat_inverse_1D_adam_MODIL/train.csv',
        'color': 'red',
        'label': 'mODIL: ADAM'
    },
    {
        'path': 'out_heat_inverse_1D_adam_ODIL/train.csv',
        'color': 'green',
        'label': 'ODIL: ADAM'
    },
    {
        'path': 'out_heat_inverse_1D_lbfgsb_MODIL/train.csv',
        'color': 'blue',
        'label': 'mODIL: L-BFGS-B'
    },
    {
        'path': 'out_heat_inverse_1D_lbfgs_ODIL/train.csv',
        'color': 'orange',
        'label': 'ODIL: L-BFGS-B'
    }
]


plt.figure(figsize=(5, 4))
ax = plt.gca()

for config in plot_configs:

    df = pd.read_csv(config['path'])
    
    loss_k = df["error_u"].values.astype(np.float32)
    epochs = df["epoch"].values.astype(np.float32)
    
    plt.plot(epochs, loss_k, linewidth=2, color=config['color'], label=config['label'])

plt.legend()
plt.xlabel("epoch")
plt.ylabel("Temperature Loss")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.xlim([0, 10e4])
plt.grid()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
