import os
import pandas as pd

# Minimum required positive anomaly rows
DEFAULT_MIN = 5
MIN_ANOMALIES = int(os.environ.get("MIN_ANOMALIES", DEFAULT_MIN))

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV = os.path.join(BASE, "labeled_waste_dataset.csv")

def test_is_anomaly_column_exists():
    assert os.path.exists(DATA_CSV), f"Data CSV not found: {DATA_CSV}"
    df = pd.read_csv(DATA_CSV)
    assert "is_anomaly" in df.columns, "Column 'is_anomaly' is missing from dataset"

def test_is_anomaly_binary_and_min_count():
    df = pd.read_csv(DATA_CSV)
    col = df["is_anomaly"]
    # ensure no nulls
    assert col.notna().all(), "Null values found in 'is_anomaly' column"

    # coerce to numeric
    nums = pd.to_numeric(col, errors="coerce")
    assert nums.notna().all(), "Non-numeric / unparseable values found in 'is_anomaly'"

    uniques = set(nums.dropna().unique())
    assert uniques.issubset({0, 1}), f"Non-binary labels found in 'is_anomaly': {uniques}"

    anomalies = int((nums == 1).sum())
    if anomalies < MIN_ANOMALIES:
        sample_idxs = nums[nums == 1].index.tolist()[:20]
        msg = (
            f"Not enough anomaly rows: {anomalies} (minimum {MIN_ANOMALIES})."
            F"Sample anomaly indices (up to 20): {sample_idxs}. "
            "To bypass for now, set MIN_ANOMALIES env var, e.g. "
            "`export MIN_ANOMALIES=5` and re-run pytest, or add more labelled anomalies."
        )
        assert anomalies >= MIN_ANOMALIES, msg