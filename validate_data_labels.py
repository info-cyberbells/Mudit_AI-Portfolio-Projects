import os
import json
import ast
import re
from collections import Counter

import numpy as np
import pandas as pd
from shutil import copy2

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
DATA_CSV = os.path.join(BASE, "labeled_waste_dataset.csv")
BACKUP = DATA_CSV + ".bak"

def backup():
    if not os.path.exists(BACKUP):
        copy2(DATA_CSV, BACKUP)
        print("Backup created:", BACKUP)
    else:
        print("Backup already exists:", BACKUP)

def show_anomalies(n=20):
    df = pd.read_csv(DATA_CSV)
    if "is_anomaly" not in df.columns:
        print("No is_anomaly column")
        return
    nums = pd.to_numeric(df["is_anomaly"], errors="coerce")
    print("Label distribution:\n", nums.value_counts(dropna=False))
    idxs = nums[nums == 1].index.tolist()
    print(f"\nShowing up to {n} anomaly rows (indices):", idxs[:n])
    if len(idxs):
        display_cols = df.columns[:min(12, len(df.columns))]
        print(df.loc[idxs[:n], display_cols].to_string(index=True))
    else:
        print("No anomaly rows found.")

def normalize_labels(save=False):
    df = pd.read_csv(DATA_CSV)
    if "is_anomaly" not in df.columns:
        print("No is_anomaly column")
        return
    s = df["is_anomaly"].astype(str).str.strip().str.lower()
    mapping = {
        "true": 1, "t": 1, "1": 1, "yes": 1, "y": 1, "anomaly": 1,
        "false": 0, "f": 0, "0": 0, "no": 0, "n": 0, "normal": 0,
        "-1": 1
    }
    mapped = s.map(mapping)
    unmapped = mapped.isna() & ~s.isna()
    if unmapped.any():
        print("Found unmapped label variants (sample):", s[unmapped].unique()[:20])
    df["is_anomaly"] = mapped.fillna(pd.NA).astype("Int64")
    print("After mapping distribution:\n", df["is_anomaly"].value_counts(dropna=False))
    if save:
        backup()
        df.to_csv(DATA_CSV, index=False)
        print("Saved normalized labels to", DATA_CSV)

if __name__ == "__main__":
    backup()
    show_anomalies(20)
