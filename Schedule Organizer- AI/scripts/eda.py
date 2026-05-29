"""reads training/data.csv and outputs simple plots and a summary.
"""
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("training/data.csv")
OUT_DIR = Path("notebooks/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    print("No data found at", DATA)
    raise SystemExit(1)


df = pd.read_csv(DATA)
# ensure label numeric
df['label_hours'] = pd.to_numeric(df['label_hours'], errors='coerce')


summary = df['label_hours'].describe()
with open(OUT_DIR / 'eda_summary.md', 'w') as f:
    f.write('# EDA Summary\n\n')
    f.write('## Label hours summary\n\n')
    f.write(summary.to_markdown())


plt.figure(figsize=(6,4))
plt.hist(df['label_hours'].dropna(), bins=20)
plt.title('Label hours distribution')
plt.xlabel('hours')
plt.ylabel('count')
plt.tight_layout()
plt.savefig(OUT_DIR / 'label_hist.png')
plt.close()

df['text_len'] = df['text'].astype(str).apply(len)
plt.figure(figsize=(6,4))
plt.hist(df['text_len'], bins=20)
plt.title('Text length distribution')
plt.xlabel('characters')
plt.ylabel('count')
plt.tight_layout()
plt.savefig(OUT_DIR / 'textlen_hist.png')
plt.close()

print('EDA outputs written to', OUT_DIR)
