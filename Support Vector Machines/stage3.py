
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from stage0 import ensure_plots_dir

def plot_decision(model, X, y, filename, mesh_limits=None):
    ensure_plots_dir()
    if mesh_limits is None:
        x0_min, x0_max = X[:,0].min()-0.5, X[:,0].max()+0.5
        x1_min, x1_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    else:
        x0_min, x0_max, x1_min, x1_max = mesh_limits

    xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 300),
                         np.linspace(x1_min, x1_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap='bwr')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k')
    plt.title(f'{model.kernel} kernel decision boundary')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f'Saved {filename}')

def run_kernels():
    ensure_plots_dir()
    X, y = make_moons(noise=0.2, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    models = {
        'linear': SVC(kernel='linear', C=1.0),
        'poly3' : SVC(kernel='poly', degree=3, C=1.0),
        'rbf'   : SVC(kernel='rbf', gamma=1.0, C=1.0)
    }

    for name, model in models.items():
        model.fit(Xtr, ytr)
        acc = model.score(Xte, yte)
        print(f'{name} kernel accuracy: {acc:.3f}')
        plot_decision(model, Xte, yte, f'plots/stage3_decision_{name}.png')

if __name__ == '__main__':
    run_kernels()
