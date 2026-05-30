RANDOM_SEED = 42

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import sklearn


np.random.seed(RANDOM_SEED)
print("Scikit-learn version:", sklearn.__version__)


data = load_breast_cancer(as_frame=True)
X_df = data.data
y = data.target.values
print("n samples:", X_df.shape[0], "n features:", X_df.shape[1])
print("Class counts:", np.bincount(y))


X = X_df.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)


scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)


clf = LogisticRegression(max_iter=5000, solver='liblinear', random_state=RANDOM_SEED)
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)
y_prob = clf.predict_proba(X_test_s)[:,1]


print("\nLogisticRegression (sklearn)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))


fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
plt.show()
