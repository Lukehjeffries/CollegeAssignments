
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from stage0 import ensure_plots_dir

class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000, verbose=False):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, X, y):
        # expects y in {-1, +1}
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for ep in range(self.epochs):
            # Simple SGD (loop over data)
            for i in range(n):
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    dw = self.w
                    db = 0.0
                else:
                    dw = self.w - self.C * y[i] * X[i]
                    db = -self.C * y[i]
                self.w -= self.lr * dw
                self.b -= self.lr * db
            if self.verbose and (ep % (self.epochs//5 + 1) == 0):
                # compute hinge loss for monitoring
                margins = y * (X.dot(self.w) + self.b)
                hinge = np.maximum(0, 1 - margins).sum()
                obj = 0.5 * (self.w @ self.w) + self.C * hinge
                print(f'ep {ep}: obj={obj:.4f}')
        return self

    def decision_function(self, X):
        return X.dot(self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

def demo():
    ensure_plots_dir()
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=1.2)
    y = 2*(y - 0.5)  # convert to {-1, +1}
    svm = LinearSVM(C=1.0, lr=0.001, epochs=1000, verbose=True)
    svm.fit(X, y)
    preds = svm.predict(X)
    acc = (preds == y).mean()
    print(f'Stage 2 demo training accuracy: {acc:.4f}')

    # plot decision boundary and margins
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300)
    )
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k', alpha=0.9)
    # draw weight vector
    origin = np.array([X[:,0].mean(), X[:,1].mean()])
    w = svm.w
    plt.arrow(origin[0], origin[1], 2*w[0], 2*w[1], head_width=0.4, head_length=0.4, linewidth=2)
    plt.title('Linear SVM (scratch) decision boundary')
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.tight_layout()
    plt.savefig('plots/stage2_svm_boundary.png', dpi=150)
    print('Stage 2 plot saved to plots/stage2_svm_boundary.png')

if __name__ == '__main__':
    demo()
