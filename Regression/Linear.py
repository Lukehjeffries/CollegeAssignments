
RANDOM_SEED = 42

import sys, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import sklearn

# Reproducibility
np.random.seed(RANDOM_SEED)


print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Platform:", platform.platform())


data = fetch_california_housing(as_frame=True)
X_df = data.data
y = data.target.values
print("n samples:", X_df.shape[0], "n features:", X_df.shape[1])


print(X_df.describe().T)
print("Target mean, std:", y.mean(), y.std())


X = X_df.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

print("\nLinearRegression (sklearn)")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Train R2:", r2_score(y_train, y_pred_train))
print("Test R2:", r2_score(y_test, y_pred_test))


lambdas = [1e-3, 1e-2, 1e-1, 1, 10]
train_rmse_ridge, test_rmse_ridge = [], []
train_rmse_lasso, test_rmse_lasso = [], []

for lam in lambdas:
    ridge = Ridge(alpha=lam)
    ridge.fit(X_train, y_train)
    train_rmse_ridge.append(np.sqrt(mean_squared_error(y_train, ridge.predict(X_train))))
    test_rmse_ridge.append(np.sqrt(mean_squared_error(y_test, ridge.predict(X_test))))

    lasso = Lasso(alpha=lam, max_iter=5000)
    lasso.fit(X_train, y_train)
    train_rmse_lasso.append(np.sqrt(mean_squared_error(y_train, lasso.predict(X_train))))
    test_rmse_lasso.append(np.sqrt(mean_squared_error(y_test, lasso.predict(X_test))))

plt.figure(figsize=(8,4))
plt.plot(lambdas, train_rmse_ridge, 'o-', label='Ridge Train')
plt.plot(lambdas, test_rmse_ridge, 'o-', label='Ridge Test')
plt.xscale('log'); plt.xlabel('lambda'); plt.ylabel('RMSE'); plt.title('Ridge RMSE')
plt.legend(); plt.show()


plt.figure(figsize=(8,4))
plt.plot(lambdas, train_rmse_lasso, 'o-', label='Lasso Train')
plt.plot(lambdas, test_rmse_lasso, 'o-', label='Lasso Test')
plt.xscale('log'); plt.xlabel('lambda'); plt.ylabel('RMSE'); plt.title('Lasso RMSE')
plt.legend(); plt.show()
