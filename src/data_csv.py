# CSV loader
# auto-detects channels (excluding metadata)
# maps labels left=0 right=1, FILTERS OUT FOOT/TONGUE

# src/data_csv.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

# Acceptable column name aliases
TRIAL_COLS = ["trial", "epoch", "segment", "window_id"]
TIME_COLS  = ["time", "t", "sample", "time_idx"]
EVENT_COLS = ["label", "event", "y", "class", "target"]

# Known EEG channel names in BCI IV 2a (22 ch). CSVs may differ; we auto-detect numeric cols too.
KNOWN_EEG_22 = [
    "Fz","FC3","FC1","FCz","FC2","FC4","C5","C3","C1","Cz","C2","C4","C6",
    "CP3","CP1","CPz","CP2","CP4","P1","Pz","P2","POz"
]

# Map common labels to ints. We keep only left/right.
LEFT_TAGS  = {0, "0", "left", "Left", 769, "769"}
RIGHT_TAGS = {1, "1", "right", "Right", 770, "770"}
FOOT_TAGS  = {2, "2", "foot", "Foot", 771, "771"}
TONGUE_TAGS= {3, "3", "tongue", "Tongue", 772, "772"}

def _first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None

def _channel_columns(df: pd.DataFrame) -> List[str]:
    # Prefer known channel names if present; otherwise auto-detect numeric columns that are not metadata.
    cols = list(df.columns)
    chans = [c for c in KNOWN_EEG_22 if c in cols]
    if len(chans) >= 8:  # reasonable threshold
        return chans
    # Auto-detect: numeric columns minus any known meta
    meta = set(TRIAL_COLS + TIME_COLS + EVENT_COLS + ["subject","patient","subj","fs","Fs","FS"])
    numeric = [c for c in cols if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    # If too many, just take the first 22 numerics (common for 2a)
    return numeric[:22] if len(numeric) > 0 else []

def _to_lr_label(val) -> Optional[int]:
    if val in LEFT_TAGS:  return 0
    if val in RIGHT_TAGS: return 1
    if val in FOOT_TAGS or val in TONGUE_TAGS:
        return None  # we drop these
    # try string parse
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"left","l"}: return 0
        if v in {"right","r"}: return 1
        if v in {"foot","tongue"}: return None
    # unknown label -> drop
    return None

def load_subject_csv(
    csv_path: str | Path,
    fs: int = 250,
    tmin: float = 0.5,
    tmax: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns (X, y, info) where:
      X: (n_trials, n_channels, n_times)
      y: (n_trials,) with 0=left, 1=right
      info: dict with 'channels', 'fs'
    Supports:
      A) Pre-epoched long format: has a trial column and time column
      B) Continuous with event/label column: epochs are cut post-event [tmin, tmax]
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Identify columns
    cols = df.columns.tolist()
    trial_col = _first_existing(cols, TRIAL_COLS)
    time_col  = _first_existing(cols, TIME_COLS)
    event_col = _first_existing(cols, EVENT_COLS)
    chan_cols = _channel_columns(df)
    if not chan_cols:
        raise ValueError("Could not detect EEG channel columns. Please rename or specify a curated CSV.")

    # If the CSV already contains only L/R trials with per-trial rows (wide), try to handle:
    # Heuristic: if each row has label and many channel columns but NO time column -> likely pre-aggregated features.
    if trial_col is None and time_col is None and event_col in cols and df.duplicated(subset=[event_col]).any() is False:
        # This path is rare; most beginner CSVs are long-format. We assume this is not your case.
        raise ValueError("Detected a wide per-trial table without time. This loader expects long format or continuous.")

    # CASE A: Pre-epoched long format
    if trial_col is not None and time_col is not None:
        if event_col is None:
            raise ValueError("Pre-epoched long format detected but no label/event column found.")

        # Keep only Left/Right rows; drop Foot/Tongue
        df["_lr"] = df[event_col].apply(_to_lr_label)
        df = df[~df["_lr"].isna()]
        df["_lr"] = df["_lr"].astype(int)

        # Fixed-length crop per trial based on time and fs
        n_samp = int(round((tmax - tmin) * fs))
        start_idx = int(round(tmin * fs))
        end_idx = start_idx + n_samp  # exclusive

        X_list, y_list = [], []
        for tid, g in df.groupby(trial_col):
            g = g.sort_values(time_col)

            # Convert time (s) to integer sample indices robustly
            samp_idx = np.round(g[time_col].to_numpy() * fs).astype(int)
            mask = (samp_idx >= start_idx) & (samp_idx < end_idx)
            g_seg = g.loc[mask]

            # If we don't get the exact count, skip (or handle with padding/interp if you prefer)
            if len(g_seg) != n_samp:
                continue

            X_ct = g_seg[_channel_columns(df)].to_numpy().T  # (channels, time)
            lbls = g_seg["_lr"].unique()
            lbl = int(lbls[0]) if len(lbls) == 1 else int(g_seg["_lr"].mode().iat[0])

            X_list.append(X_ct)
            y_list.append(lbl)

        if not X_list:
            return np.empty((0, 0, 0)), np.array([], dtype=int), {"channels": [], "fs": fs, "schema": "pre-epoched-long"}

        X = np.stack(X_list, axis=0)  # (n_trials, n_channels, n_times)
        y = np.array(y_list, dtype=int)
        info = {"channels": _channel_columns(df), "fs": fs, "schema": "pre-epoched-long"}
        return X, y, info


    # CASE B: Continuous with event markers → epoch after events
    if event_col is None:
        raise ValueError("Continuous format requires an event/label column to cut epochs.")
    
    # Identify event onset rows (we consider only left/right)
    df["_lr"] = df[event_col].apply(_to_lr_label)
    event_idx = df.index[~df["_lr"].isna()].to_numpy()
    y_all = df.loc[event_idx, "_lr"].astype(int).to_numpy()

    # Build epochs for each event: samples [tmin, tmax] after the event row
    n_samp = int(round((tmax - tmin) * fs))
    offset = int(round(tmin * fs))
    X_list = []
    for i, idx in enumerate(event_idx):
        start = idx + offset
        end = start + n_samp
        if end > len(df):
            continue
        seg = df.iloc[start:end][chan_cols].to_numpy()  # (time, channels)
        if seg.shape[0] != n_samp:
            continue
        X_list.append(seg.T)  # (channels, time)
    
    X = np.stack(X_list, axis=0) if X_list else np.empty((0,len(chan_cols),0))
    y = y_all[:len(X_list)]
    info = {"channels": chan_cols, "fs": fs, "schema": "continuous-events"}
    return X, y, info

def load_subject_i(data_dir: str | Path, i: int, fs: int=250, tmin=0.5, tmax=3.5):
    p = Path(data_dir) / "raw" / "BCICIV_2a_csv" / f"BCICIV_2a_{i}.csv"
    return load_subject_csv(p, fs=fs, tmin=tmin, tmax=tmax)
