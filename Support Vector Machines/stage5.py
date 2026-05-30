
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from stage2 import LinearSVM
from stage0 import ensure_plots_dir

def simulate():
    rng = np.random.default_rng(1)
    n = 200
    temp = np.r_[rng.normal(20, 2, n//2), rng.normal(35, 3, n//2)]
    humid = np.r_[rng.normal(50, 5, n//2), rng.normal(30, 4, n//2)]
    y = np.r_[np.zeros(n//2), np.ones(n//2)]
    X = np.c_[temp, humid]
    return X, y

def plot_with_support(model, X, y, filename, show_sv=True):
    ensure_plots_dir()
    x0_min, x0_max = X[:,0].min()-2, X[:,0].max()+2
    x1_min, x1_max = X[:,1].min()-5, X[:,1].max()+5
    xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 300),
                         np.linspace(x1_min, x1_max, 300))
    if hasattr(model, 'predict'):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    else:
        Z = np.zeros_like(xx)
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k', alpha=0.9)
    if show_sv and hasattr(model, 'support_vectors_'):
        sv = model.support_vectors_
        plt.scatter(sv[:,0], sv[:,1], facecolors='none', edgecolors='k', s=90, linewidths=1.5, label='SV')
        plt.legend()
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Humidity (%)')
    plt.title('Environmental - decision boundary and support vectors')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f'Saved {filename}')

def run_env():
    X, y = simulate()
    ensure_plots_dir()
    y_scratch = 2*(y - 0.5)
    scratch = LinearSVM(C=1.0, lr=0.001, epochs=1500)
    scratch.fit(X, y_scratch)
    preds_s = scratch.predict(X)
    acc_s = (preds_s == y_scratch).mean()
    print(f'Scratch linear SVM accuracy (on train): {acc_s:.4f}')

    rbf = SVC(kernel='rbf', C=1.0, gamma=0.1)
    rbf.fit(X, y)
    preds_r = rbf.predict(X)
    acc_r = (preds_r == y).mean()
    print(f'RBF SVM accuracy (on train): {acc_r:.4f}')
    print('RBF support vectors count per class:', rbf.n_support_)

  
    class ScratchWrapper:
        def __init__(self, scratch):
            self.scratch = scratch
        def predict(self, Xq):
            return (self.scratch.predict(Xq) > 0).astype(int)

    plot_with_support(ScratchWrapper(scratch), X, y, 'plots/stage5_scratch_boundary.png', show_sv=False)
    plot_with_support(rbf, X, y, 'plots/stage5_rbf_boundary.png', show_sv=True)

   
    X_feat = np.c_[X, (X[:,0] * X[:,1])]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xf_s = scaler.fit_transform(X_feat)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(Xf_s, y)
    print('Linear SVM on engineered feature accuracy (train):', clf.score(Xf_s, y))
    print('Stage 5 complete. Plots saved in plots/.')
    return

if __name__ == '__main__':
    run_env()
