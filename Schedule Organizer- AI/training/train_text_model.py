import sys
import os
import csv
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
	sys.path.insert(0, repo_root)

from model import WorkloadEstimator


def main(data_path="training/data.csv", out_model_path="training/estimator.pkl"):
	texts = []
	hours = []
	with open(data_path, newline="") as f:
		reader = csv.DictReader(f)
		for r in reader:
			texts.append(r.get("text", ""))
			try:
				hours.append(float(r.get("label_hours", 0)))
			except ValueError:
				hours.append(0.0)

	if not texts:
		raise SystemExit("No data found in " + data_path)

	X_train, X_test, y_train, y_test = train_test_split(texts, hours, test_size=0.2, random_state=42)

	est = WorkloadEstimator()
	est.fit(X_train, y_train)

	preds = [est.predict_hours(t) for t in X_test]
	mae = mean_absolute_error(y_test, preds)
	print("Trained on", len(X_train), "examples; test MAE:", round(mae, 3))

	with open(out_model_path, "wb") as f:
		pickle.dump(est, f)
	print("Saved model to", out_model_path)


if __name__ == '__main__':
	main()
