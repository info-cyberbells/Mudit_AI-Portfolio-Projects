import os
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV = os.path.join(BASE, "labeled_waste_dataset.csv")
OUT = os.path.join(BASE, "label_review_sample.csv")

df = pd.read_csv(DATA_CSV)
nums = pd.to_numeric(df["is_anomaly"], errors="coerce")
anom_idx = nums[nums == 1].index.tolist()
# include 3 neighbors per anomaly for context
rows = sorted(set(sum([[max(0,i-3)] + list(range(max(0,i-1), i+2)) + [i+1, i+3] for i in anom_idx], [])))
rows = [r for r in rows if r < len(df)]
sample = df.loc[rows].reset_index(drop=True)
sample.to_csv(OUT, index=False)
print("Wrote label_review_sample.csv with", len(sample), "rows:", OUT)
# open file on mac
try:
    os.system(f"open '{OUT}'")
except Exception:
    pass