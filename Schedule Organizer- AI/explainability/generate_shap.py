"""Generate SHAP plots and a plain-language
 summary saved to explainability/summary.md
"""
from pathlib import Path
import pickle
import csv

try:
    import shap
    import numpy as np
    import matplotlib.pyplot as plt
except Exception:
    shap = None

MODEL = Path('training/estimator.pkl')
DATA = Path('training/data.csv')
OUT = Path('explainability/outputs')
OUT.mkdir(parents=True, exist_ok=True)

if not MODEL.exists():
    print('Model not found:', MODEL)
    raise SystemExit(1)

if not DATA.exists():
    print('Data not found:', DATA)
    raise SystemExit(1)

with MODEL.open('rb') as f:
    model = pickle.load(f)

texts = []
with DATA.open(newline='') as f:
    reader = csv.DictReader(f)
    for i, r in enumerate(reader):
        if i >= 200:
            break
        texts.append(r.get('text',''))

if shap is None:
    print('shap not installed. Install with: python3 -m pip install shap')
    raise SystemExit(1)

if hasattr(model, 'named_steps') and 'tfidf' in model.named_steps and 'rf' in model.named_steps:
    tfidf = model.named_steps['tfidf']
    rf = model.named_steps['rf']
    X = tfidf.transform(texts).toarray()
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        arr = shap_vals[0]
    else:
        arr = shap_vals
    mean_abs = np.mean(np.abs(arr), axis=0)
    feature_names = tfidf.get_feature_names_out()
    top_idx = np.argsort(-mean_abs)[:20]
    with open(OUT / 'summary.md', 'w') as f:
        f.write('# SHAP Summary\n\n')
        f.write('Top TF-IDF features by mean(|SHAP|):\n\n')
        for i in top_idx:
            f.write(f"- {feature_names[i]}: {mean_abs[i]:.4f}\n")
    # also save a beeswarm plot
    try:
        shap.summary_plot(arr, features=X, feature_names=feature_names, show=False)
        plt.savefig(OUT / 'shap_summary.png')
        plt.close()
    except Exception as e:
        print('Could not plot SHAP summary:', e)
    print('SHAP outputs written to', OUT)
else:
    print('Model pipeline not compatible with TF-IDF+RF structure; try training with evaluate_models.py')
