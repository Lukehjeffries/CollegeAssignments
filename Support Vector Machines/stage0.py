
import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(42)

def ensure_plots_dir():
    os.makedirs('plots', exist_ok=True)

if __name__ == '__main__':
    ensure_plots_dir()
    print('Stage 0: Setup complete. Plots directory created.')
