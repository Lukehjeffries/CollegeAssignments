from tasks import extract_subtasks, normalize_task_name
from model import WorkloadEstimator
from schedule import generate_schedule
import datetime
import argparse
import sys
import csv
import os
from pathlib import Path

def input_multiline():
    print("Paste assignment description. Press ENTER twice to finish.")
    lines = []
    blanks = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            blanks += 1
            if blanks == 2:
                break
        else:
            blanks = 0
        lines.append(line)
    return "\n".join(lines)

TRAIN_TASKS = [
    "Write a research paper",
    "Create an annotated bibliography",
    "Prepare a presentation",
    "Implement a Python program",
    "Edit and proofread the assignment"
]

TRAIN_HOURS = [18, 10, 3, 14, 1]

def main():
    print("Assignment Organizer \n")
    est = WorkloadEstimator()
    est.fit(TRAIN_TASKS, TRAIN_HOURS)
    parser = argparse.ArgumentParser(description="Assignment organizer")
    parser.add_argument("-n", "--count", type=int, help="number of assignments to enter")
    args, _ = parser.parse_known_args()

    try:
        if args.count is not None:
            n = args.count
        else:
            n = int(input("How many assignments do you want to enter? "))
    except (KeyboardInterrupt, EOFError):
        print("\nInput cancelled. Exiting.")
        sys.exit(1)
    tasks = []

    try:
        for _ in range(n):
            title = input("Assignment title: ")
            deadline = input("Deadline (YYYY-MM-DD): ")
            desc = input_multiline()
            
            subs = extract_subtasks(desc)
            for s in subs:
                tasks.append({
                    "task": normalize_task_name(s),
                    "assignment": title,
                    "deadline": deadline,
                    "hours": est.predict_hours(s)
                })
    except (KeyboardInterrupt, EOFError):
        print("\nInput cancelled during entry. Proceeding with collected tasks.")
        subs = extract_subtasks(desc)
        for s in subs:
            tasks.append({
                "task": normalize_task_name(s),
                "assignment": title,
                "deadline": deadline,
                "hours": est.predict_hours(s)
            })

    schedule = generate_schedule(tasks)
    print("\nAGENDA\n")
    for block in schedule:
        print(f"{block['day']} {block['start']}–{block['end']} → [{block['assignment']}] {block['task']}")

    def _append_record(text, predicted_hours, label_hours=None, confirmed=False, csv_path="training/data_logged.csv"):
        Path(os.path.dirname(csv_path)).mkdir(parents=True, exist_ok=True)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["text","predicted_hours","label_hours","confirmed","timestamp","source"])
            writer.writerow([text, predicted_hours, label_hours if label_hours is not None else "", int(bool(confirmed)), datetime.datetime.utcnow().isoformat(), "main_app"])
    for t in tasks:
        _append_record(t.get("task", ""), t.get("hours", ""))
    print('\nPredictions appended to training/data_logged.csv (unconfirmed).')

if __name__ == "__main__":
    main()
