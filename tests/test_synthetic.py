import os
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SYNTH_CSV = os.environ.get("SYNTHETIC_CSV", os.path.join(BASE, "synthetic_anomalies.csv"))

def test_synthetic_file_exists():
    assert os.path.exists(SYNTH_CSV), f"Synthetic CSV not found: {SYNTH_CSV}"

def test_synthetic_rows_flags():
    df = pd.read_csv(SYNTH_CSV)
    assert not df.empty, "Synthetic CSV is empty"
    assert "is_anomaly" in df.columns, "'is_anomaly' column missing in synthetic CSV"
    assert "is_synthetic" in df.columns, "'is_synthetic' column missing in synthetic CSV"

    # check all synthetic rows flagged as anomalies
    nums = pd.to_numeric(df["is_anomaly"], errors="coerce")
    assert nums.notna().all(), "Some 'is_anomaly' values are non-numeric / unparseable"
    assert (nums == 1).all(), "Not all synthetic rows have is_anomaly == 1"

    # check is_synthetic truthy
    synth_flags = df["is_synthetic"].astype(str).str.lower()
    assert synth_flags.isin(["true", "1"]).all(), "Not all rows marked as synthetic (is_synthetic not True/1)"