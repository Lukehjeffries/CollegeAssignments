"""For each unconfirmed row this script shows the text and 
predicted_hours and prompts you to enter a confirmed label_hours (number). 
Leave blank to skip. Entering s skips, q quits early. Confirmed rows will be 
written back to training/data_logged.csv with confirmed set to 1 and label_hours 
populated. simplify wording
"""
import csv
from pathlib import Path
import shutil

LOG = Path("training/data_logged.csv")

def load_rows(path):
    if not path.exists():
        print(path, "not found")
        return []
    with path.open(newline="") as f:
        reader = list(csv.DictReader(f))
    return reader

def save_rows(path, rows):
    if path.exists():
        shutil.copy(path, str(path) + ".bak")
    fieldnames = rows[0].keys() if rows else ["text","predicted_hours","label_hours","confirmed","timestamp","source"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def prompt_number(prompt):
    while True:
        v = input(prompt).strip()
        if v == "":
            return None
        if v.lower() in ("q","quit"):
            return "__QUIT__"
        if v.lower() in ("s","skip"):
            return "__SKIP__"
        try:
            return float(v)
        except ValueError:
            print("Please enter a number, or leave blank to skip, 's' to skip, 'q' to quit.")

def main():
    rows = load_rows(LOG)
    if not rows:
        print("No logged data found.")
        return

    updated = False
    for i, r in enumerate(rows):
        if r.get("confirmed", "0") in ("1", "True", "true"):
            continue
        print("\nEntry #{}:\n".format(i+1))
        print(r.get("text", ""))
        print("Predicted hours:", r.get("predicted_hours", ""))
        val = prompt_number("Enter confirmed label hours (blank=skip, s=skip, q=quit): ")
        if val == "__QUIT__":
            break
        if val == "__SKIP__" or val is None:
            continue
        r["label_hours"] = str(val)
        r["confirmed"] = "1"
        updated = True

    if updated:
        save_rows(LOG, rows)
        print("Updated", LOG, "(backup saved as .bak)")
    else:
        print("No changes made.")

if __name__ == '__main__':
    main()
