
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from stage0 import ensure_plots_dir

def run_digits():
    ensure_plots_dir()
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=0, stratify=y)
    param_grid = {'C':[0.1,1,10], 'gamma':[0.001,0.01,0.1]}
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, n_jobs=-1)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    print('Stage 4 - Digits: best params =', grid.best_params_)
    ypred = best.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f'Test accuracy: {acc:.4f}')

    cm = confusion_matrix(yte, ypred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title('Digits Confusion Matrix')
    plt.tight_layout()
    plt.savefig('plots/stage4_confusion_matrix.png', dpi=150)
    plt.close()
    print('Saved plots/stage4_confusion_matrix.png')

    print('Support vectors per class:', best.n_support_)
    print('Total support vectors:', sum(best.n_support_))

if __name__ == '__main__':
    run_digits()
