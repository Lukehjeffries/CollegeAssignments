"""The script loads a saved model from training/estimator.pkl, 
computes SHAP values on a small sample, and prints the TF-IDF features 
that contribute most to the predictions.
"""

import os
import sys
import csv
import pickle
from pathlib import Path
import argparse

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    import shap
    import numpy as np
except Exception:
    shap = None


def read_texts(path, n=200):
    texts = []
    p = Path(path)
    if not p.exists():
        return texts
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= n:
                break
            texts.append(r.get("text", ""))
    return texts


def analyze(model_path="training/estimator.pkl", data_path="training/data.csv", sample_n=200, top_k=20):
    model_file = Path(model_path)
    alt = Path("training/trainingresults.pkl")
    if not model_file.exists():
        if alt.exists():
            print(f"{model_path} not found, but found {alt}. Using that file.")
            model_file = alt
        else:
            print("Model not found:", model_path)
            print("Run `python3 training/evaluate_models.py` or `python3 training/train_text_model.py` first to produce the model.")
            return 1

    with model_file.open("rb") as f:
        model = pickle.load(f)

    texts = read_texts(data_path, n=sample_n)
    if not texts:
        print("No text data found in", data_path)
        return 1

    if shap is None:
        print("`shap` is not installed. Install it with:")
        print("  python3 -m pip install shap")
        return 1

    if not (hasattr(model, "named_steps") and "tfidf" in model.named_steps):
        print("Model does not appear to be a Pipeline with a 'tfidf' step. Trying to explain model.predict directly.")

    try:
        if hasattr(model, "named_steps") and "tfidf" in model.named_steps and "rf" in model.named_steps:
            tfidf = model.named_steps["tfidf"]
            rf = model.named_steps["rf"]

            X_sparse = tfidf.transform(texts)
            X = X_sparse.toarray()

            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_vals_arr = np.array(shap_values[0])
            else:
                shap_vals_arr = np.array(shap_values)

            mean_abs = np.mean(np.abs(shap_vals_arr), axis=0)
            feature_names = np.array(tfidf.get_feature_names_out())
            top_idx = np.argsort(-mean_abs)[:top_k]

            print(f"Top {top_k} TF-IDF features by mean(|SHAP value|):")
            for i in top_idx:
                print(f"{feature_names[i]}\t{mean_abs[i]:.4f}")
            return 0
        else:
            explainer = shap.Explainer(model.predict, texts)
            shap_vals = explainer(texts[:min(50, len(texts))])
            print("Computed SHAP with generic Explainer (text-based); try opening plots in a notebook.")
            return 0
    except Exception as e:
        print("SHAP analysis failed:", e)
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="training/estimator.pkl")
    parser.add_argument("--data", default="training/data.csv")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    return analyze(args.model, args.data, sample_n=args.sample, top_k=args.top)


if __name__ == '__main__':
    raise SystemExit(main())
