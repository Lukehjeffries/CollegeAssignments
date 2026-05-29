"""Test simple text-to-hours models using preprocessing and cross-validation.
The script saves as training/estimator.pkl and outputs a CSV with
 the evaluation results.
"""
import os
import sys
import csv
import pickle
from pathlib import Path

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

def read_data(path):
    texts, hours = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            texts.append(r.get("text", ""))
            try:
                hours.append(float(r.get("label_hours", 0)))
            except Exception:
                hours.append(0.0)
    return texts, hours

def main(data_path="training/data.csv", out_model_path="training/estimator.pkl", results_csv="training/model_results.csv"):
    Path(os.path.dirname(out_model_path)).mkdir(parents=True, exist_ok=True)
    texts, hours = read_data(data_path)
    if not texts:
        raise SystemExit("No data found in " + data_path)

    X_train, X_test, y_train, y_test = train_test_split(texts, hours, test_size=0.2, random_state=42)

    
    pipe_rf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=2000)),
        ("rf", RandomForestRegressor(random_state=42))
    ])

    pipe_ridge = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=2000)),
        ("ridge", Ridge())
    ])

    param_grid = {
        "rf__n_estimators": [50, 100],
        "rf__max_depth": [None, 10]
    }

    gs = GridSearchCV(pipe_rf, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    
    preds = best.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))

    
    ridge_scores = cross_val_score(pipe_ridge, texts, hours, cv=5, scoring="neg_mean_absolute_error")
    from sklearn.model_selection import cross_val_score as cvs
    ridge_mse_scores = cvs(pipe_ridge, texts, hours, cv=5, scoring="neg_mean_squared_error")
    ridge_rmse = [math.sqrt(-s) for s in ridge_mse_scores]

    
    with open(out_model_path, "wb") as f:
        pickle.dump(best, f)

    
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mae_test", "rmse_test", "cv_mean_neg_mae", "cv_std", "cv_rmse_mean", "cv_rmse_std", "best_params"])
        cv_rmse_mean = round(sum(ridge_rmse)/len(ridge_rmse),3) if ridge_rmse else ""
        cv_rmse_std = round((sum((x - sum(ridge_rmse)/len(ridge_rmse))**2 for x in ridge_rmse)/len(ridge_rmse))**0.5,3) if ridge_rmse else ""
        writer.writerow(["best_grid_rf", round(mae,3), round(rmse,3), round(gs.best_score_,3), round(ridge_scores.std(),3), cv_rmse_mean, cv_rmse_std, str(gs.best_params_)])

    print(f"Best model saved to {out_model_path}; test MAE={round(mae,3)}")

if __name__ == '__main__':
    main()
