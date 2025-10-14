import os
import argparse
import numpy as np
import pandas as pd
from shutil import copy2

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_INPUT = os.path.join(BASE, "labeled_waste_dataset.csv")
DEFAULT_OUT_AUG = os.path.join(BASE, "labeled_waste_dataset.augmented.csv")
DEFAULT_OUT_SYN = os.path.join(BASE, "synthetic_anomalies.csv")
BACKUP = DEFAULT_INPUT + ".bak"

def backup(src=DEFAULT_INPUT):
    if not os.path.exists(BACKUP):
        copy2(src, BACKUP)
        print("Backup created:", BACKUP)

def choose_seed_rows(df, n_seeds=50, prefer_anomalies=True, random_state=42, targets=None):
    rng = np.random.default_rng(random_state)
    targets = targets or []
    if prefer_anomalies and "is_anomaly" in df.columns and df["is_anomaly"].astype(str).isin(["1", "True", "true"]).any():
        cand = df[df["is_anomaly"].astype(str).isin(["1", "True", "true"])]
        if not cand.empty:
            return cand.reset_index(drop=True)
    
    cand = pd.DataFrame()
    for t in targets:
        if t in ("vfd", "vfd_temp"):
            vfd_cols = [c for c in df.columns if ("VFD" in c or "vfd" in c or "max_vfd_temperature" in c)]
            if vfd_cols:
                df["_vfd_score"] = df[vfd_cols].max(axis=1, skipna=True)
                cand = pd.concat([cand, df.nlargest(n_seeds, "_vfd_score")], ignore_index=True)
        elif t == "tool_wear":
            if "tool_wear" in df.columns:
                cand = pd.concat([cand, df.nlargest(n_seeds, "tool_wear")], ignore_index=True)
        elif t == "vibration":
            if "vibration_level" in df.columns:
                cand = pd.concat([cand, df.nlargest(n_seeds, "vibration_level")], ignore_index=True)
        elif t == "waste":
            if "predicted_material_waste" in df.columns:
                cand = pd.concat([cand, df.nlargest(n_seeds, "predicted_material_waste")], ignore_index=True)
        elif t == "gripper":
            gr_cols = [c for c in df.columns if "Gripper_Load" in c or "gripper_load" in c or "gripper" in c]
            if gr_cols:
                df["_grip_score"] = df[gr_cols].sum(axis=1, skipna=True)
                cand = pd.concat([cand, df.nlargest(n_seeds, "_grip_score")], ignore_index=True)

    if cand.empty:
        if "predicted_material_waste" in df.columns:
            cand = df.nlargest(n_seeds, "predicted_material_waste")
        elif "tool_wear" in df.columns:
            cand = df.nlargest(n_seeds, "tool_wear")
        else:
            cand = df.sample(min(len(df), n_seeds), random_state=random_state)
    
    for tmp in ("_vfd_score", "_grip_score"):
        if tmp in df.columns:
            df.drop(columns=[tmp], inplace=True, errors=True)
    
    cand = cand.drop_duplicates().reset_index(drop=True)
    if cand.empty:
        cand = df.sample(min(len(df), n_seeds), random_state=random_state)
    return cand.reset_index(drop=True)

def perturb_row(row, numeric_cols, rng, strength=0.2, targets=None):
    r = row.copy()
    targets = targets or []
    # Add gaussian noise proportional to std of column
    for c in numeric_cols:
        val = r.get(c, np.nan)
        if pd.isna(val):
            continue
        scale = max(abs(val), 1.0)
        noise = rng.normal(loc=0.0, scale=strength * scale * 0.05)
        r[c] = val + noise

    # Boost key anomaly indicators if present
    if "tool_wear" in r and ("tool_wear" in targets or "wear" in targets):
        # accelerate wear drastically
        orig = float(r.get("tool_wear", 0) or 0)
        r["tool_wear"] = float(max(orig * (1 + rng.uniform(0.4, 1.5)), orig + rng.uniform(10, 40)))

    if ("vibration" in targets or "vibration_level" in targets) and "vibration_level" in r:
        orig = float(r.get("vibration_level", 0) or 0)
        # introduce spikes and increased variance
        r["vibration_level"] = float(orig * (1 + rng.uniform(1.0, 3.0)) + rng.normal(0, 0.5))

    if ("vfd" in targets or "vfd_temp" in targets):
        # increase VFD temps and set anomaly flags
        vfd_cols = [c for c in numeric_cols if ("VFD" in c or "vfd" in c or "Vfd" in c or "max_vfd_temperature" in c)]
        for c in vfd_cols:
            val = float(r.get(c, 0) or 0)
            # add a temperature spike proportional to column scale
            spike = rng.uniform(5.0, 25.0)
            r[c] = float(val + spike + rng.normal(0, 0.2 * max(abs(val), 1.0)))
        # set high-level vfd anomaly indicators if present
        r["vfd_temperature_anomaly"] = 1
        # recompute max_vfd_temperature if field exists or create it
        vfd_max_cols = [c for c in vfd_cols if "max" in c.lower()] or ["max_vfd_temperature"]
        if vfd_max_cols:
            r["max_vfd_temperature"] = max([float(r.get(c, 0) or 0) for c in vfd_cols])

    if ("waste" in targets or "predicted_material_waste" in targets) and "predicted_material_waste" in r:
        r["predicted_material_waste"] = float(min(1.0, r.get("predicted_material_waste", 0) + rng.uniform(0.2, 0.6)))

    if ("gripper" in targets or "gripper_load" in targets):
        gr_cols = [c for c in numeric_cols if "Gripper_Load" in c or "gripper_load" in c or "gripper" in c]
        for c in gr_cols:
            val = float(r.get(c, 0) or 0)
            r[c] = float(val * (1 + rng.uniform(0.3, 2.0)) + rng.normal(0, 0.1 * max(abs(val), 1.0)))
        r["gripper_overload_flag"] = 1
        # compute asymmetry if possible
        lefts = [c for c in gr_cols if "R01" in c or "R02" in c or "R03" in c]
        if lefts:
            vals = [float(r.get(c, 0) or 0) for c in lefts]
            if len(vals) >= 2:
                r["gripper_load_asymmetry"] = float(np.std(vals))
    # anomaly metadata
    r["is_anomaly"] = 1
    if "anomaly_score" in r:
        r["anomaly_score"] = float(min(1.0, float(r.get("anomaly_score", 0) or 0) + rng.uniform(0.5, 1.0)))
    if "anomaly_severity" in r and not r.get("anomaly_severity"):
        r["anomaly_severity"] = "synthetic_targeted"
    return r

def generate(n=10, input_csv=DEFAULT_INPUT, out_aug=DEFAULT_OUT_AUG, out_syn=DEFAULT_OUT_SYN, mode="separate", seed=42, strength=0.25, targets=None):
    assert mode in ("separate", "append"), "mode must be 'separate' or 'append'"
    df = pd.read_csv(input_csv)
    rng = np.random.default_rng(seed)

    targets = targets or []
    seeds = choose_seed_rows(df, n_seeds=min(100, max(10, n)), prefer_anomalies=True, random_state=seed, targets=targets)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    synth_list = []
    for i in range(n):
        base = seeds.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        s = perturb_row(base, numeric_cols, rng, strength=strength, targets=targets)
        # give unique synthetic id
        s["cycle_id"] = f"synth_{seed}_{i}"
        s["is_synthetic"] = True
        synth_list.append(s)

    synth_df = pd.DataFrame(synth_list)

    if mode == "append":
        backup(input_csv)
        out_df = pd.concat([df, synth_df], ignore_index=True)
        out_df.to_csv(out_aug, index=False)
        print(f"Appended {len(synth_df)} synthetic anomalies and wrote: {out_aug}")
    else:
        synth_df.to_csv(out_syn, index=False)
        print(f"Wrote {len(synth_df)} synthetic anomalies to: {out_syn}")

    # quick summary
    print("Original anomaly count:", int(df['is_anomaly'].astype(str).isin(['1', 'True', 'true']).sum()) if 'is_anomaly' in df.columns else "N/A")
    print("Synthetic added:", len(synth_df))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate synthetic anomaly rows for labeled_waste_dataset.csv")
    p.add_argument("-n", "--num", type=int, default=10, help="Number of synthetic anomalies to generate")
    p.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to labeled CSV")
    p.add_argument("--mode", choices=["separate", "append"], default="separate", help="'separate' -> write synthetic file; 'append' -> append to original (creates backup)")
    p.add_argument("--out", type=str, default=None, help="Output path (file). If not set uses defaults.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--strength", type=float, default=0.25, help="Perturbation strength (0-1)")
    p.add_argument("--targets", type=str, default="", help="Comma-separated targets: vfd,tool_wear,vibration,gripper,waste,random")
    args = p.parse_args()

    out = args.out
    if out is None:
        out = DEFAULT_OUT_AUG if args.mode == "append" else DEFAULT_OUT_SYN
    
    targets = [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    if "random" in targets:
        targets = []

    generate(n=args.num, input_csv=args.input, out_aug=out, out_syn=out, mode=args.mode, seed=args.seed, strength=args.strength, targets=targets)