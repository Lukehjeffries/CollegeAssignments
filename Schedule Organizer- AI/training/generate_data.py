"""Generate a dataset for task to hour estimate
Creates training/data.csv by expanding the original
examples with simple templates and small variations in the labels.
"""
import csv
import random

base_examples = [
    ("Write a research paper", 18),
    ("Create annotated bibliography", 10),
    ("Prepare presentation", 3),
]

task_templates = ["{verb} {object}", "{verb} the {object}"]


verbs = ["Write", "Create", "Prepare", "Fix", "Test"]
objects = ["paper", "bib", "slides", "pipeline", "script", "report", "docs", "tests", "deploy", "data"]

def augment_text(verb, obj, verb2=None, template_index=0):
    tmpl = task_templates[template_index % len(task_templates)]
    return tmpl.format(verb=verb, verb2=verb2 or verb, object=obj)

def main(out_path="training/data.csv", n_per_base=15, seed=42):
    random.seed(seed)
    rows = [("text", "label_hours")]

    for text, hours in base_examples:
        for i in range(n_per_base):
            verb = random.choice(verbs)
            obj = random.choice(objects)
            template_index = random.randrange(len(task_templates))
            text_aug = augment_text(verb, obj, random.choice(verbs), template_index)
          
            noise = random.uniform(-0.3, 0.3) * hours
            hours_aug = max(0.5, round(hours + noise, 1))
            rows.append((text_aug, hours_aug))

    additional = [
        ("Debug intermittent test failure", 2.5),
        ("Implement new feature for user signup", 6),
        ("Write integration tests for payments", 5),
        ("Conduct exploratory data analysis on sales data", 8),
        ("Run hyperparameter search and log results", 12),
        ("Clean and label 1000 data samples", 10),
        ("Prepare slides for weekly demo", 2),
        ("Rehearse the product demo presentation", 1.5),
        ("Deploy service to staging and validate", 3),
        ("Prepare release notes and changelog", 1.5),
        ("Design database schema for new feature", 4),
        ("Profile and optimize slow query", 3.5),
        ("Create synthetic data generator", 7),
        ("Perform model evaluation and error analysis", 6),
    ]

    for t, h in additional:
        rows.append((t, h))


    random.shuffle(rows[1:])
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Wrote {len(rows)-1} examples to {out_path}")

if __name__ == "__main__":
    main()
