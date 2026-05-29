"""The script copies rows marked as confirmed (with label_hours filled in) 
into the main training file, marks them as ingested in training/data_logged.csv, 
and can optionally retrain the model by running training/train_text_model.py.
"""
import csv
from pathlib import Path
import shutil
import subprocess
import argparse

LOG = Path("training/data_logged.csv")
DATA = Path("training/data.csv")

def load_rows(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows

def save_rows(path, rows, fieldnames):
    shutil.copy(path, str(path) + ".bak")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def append_to_data(path, rows):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["text","label_hours"])
        for r in rows:
            writer.writerow([r.get("text",""), r.get("label_hours","")])

def main(retrain=False):
    if not LOG.exists():
        print(LOG, "not found")
        return
    rows = load_rows(LOG)
    if not rows:
        print("No rows in", LOG)
        return

    fieldnames = list(rows[0].keys())
    if "ingested" not in fieldnames:
        fieldnames.append("ingested")

    to_append = []
    for r in rows:
        confirmed = r.get("confirmed", "0")
        label = r.get("label_hours", "")
        ingested = r.get("ingested", "0")
        if str(confirmed) in ("1", "True", "true") and label and str(ingested) not in ("1","True","true"):
            to_append.append(r)
            r["ingested"] = "1"

    if not to_append:
        print("No confirmed, un-ingested rows to import.")
        return

    append_to_data(DATA, to_append)
    save_rows(LOG, rows, fieldnames)
    print(f"Appended {len(to_append)} rows to {DATA} and updated {LOG} (backup saved).")

    if retrain:
        print("Retraining model...")
        subprocess.run(["python3", "training/train_text_model.py"], check=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Call training/train_text_model.py after ingesting")
    args = parser.parse_args()
    main(retrain=args.retrain)
