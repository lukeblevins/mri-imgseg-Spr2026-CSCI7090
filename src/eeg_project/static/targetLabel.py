from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd


SeizureInterval = Tuple[float, float]


@dataclass
class StaticBuildConfig:
    dataset_root: str | Path
    target_sfreq: int = 256
    epoch_duration: float = 10.0
    epoch_overlap: float = 0.0
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch_freq: float = 60.0

    # prediction mode only
    preictal_horizon_sec: float = 600.0
    postictal_exclusion_sec: float = 1800.0
    interictal_gap_sec: float = 300.0

    # for seizure prediction, these should stay True
    drop_non_prediction_windows: bool = True


# -----------------------------
# small helpers
# -----------------------------

def _normalize_name(name: str) -> str:
    return Path(name.strip()).name.lower()


def _extract_first_number(text: str) -> Optional[float]:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def _parse_time_to_seconds(text: str) -> Optional[float]:
    text = text.strip()

    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        if all(re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
            nums = [float(p) for p in parts]
            if len(nums) == 3:
                h, m, s = nums
                return h * 3600 + m * 60 + s
            if len(nums) == 2:
                m, s = nums
                return m * 60 + s

    return _extract_first_number(text)


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _infer_dataset_name(edf_path: Path) -> str:
    path_str = str(edf_path).lower()
    if "chbmit" in path_str:
        return "chbmit"
    if "siena" in path_str:
        return "siena"
    raise ValueError(f"Could not infer dataset from path: {edf_path}")


# -----------------------------
# dataset discovery
# -----------------------------

def discover_edf_files(dataset_root: str | Path) -> List[Path]:
    root = Path(dataset_root)
    return sorted(root.rglob("*.edf"))


# -----------------------------
# CHB-MIT parsing
# -----------------------------

def parse_chbmit_summary_file(summary_path: str | Path) -> Dict[str, List[SeizureInterval]]:
    """
    Example patterns:
      File Name: chb01_03.edf
      Seizure 1 Start Time: 2996 seconds
      Seizure 1 End Time: 3036 seconds
    """
    summary_path = Path(summary_path)
    out: Dict[str, List[SeizureInterval]] = {}

    current_file: Optional[str] = None
    starts: List[float] = []
    ends: List[float] = []

    lines = summary_path.read_text(errors="ignore").splitlines()

    def flush() -> None:
        nonlocal current_file, starts, ends
        if current_file is not None:
            intervals = list(zip(starts, ends))
            out.setdefault(_normalize_name(current_file), []).extend(intervals)
        current_file = None
        starts = []
        ends = []

    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()

        if lower.startswith("file name:"):
            flush()
            current_file = line.split(":", 1)[1].strip()
            continue

        if "start time" in lower and "seizure" in lower:
            sec = _parse_time_to_seconds(line)
            if sec is not None:
                starts.append(sec)
            continue

        if "end time" in lower and "seizure" in lower:
            sec = _parse_time_to_seconds(line)
            if sec is not None:
                ends.append(sec)
            continue

    flush()

    for key in out:
        out[key] = sorted(out[key], key=lambda x: x[0])

    return out


def load_all_chbmit_annotations(dataset_root: str | Path) -> Dict[str, List[SeizureInterval]]:
    root = Path(dataset_root)
    all_annotations: Dict[str, List[SeizureInterval]] = {}

    for txt_file in root.rglob("*summary*.txt"):
        if "chb" not in txt_file.name.lower():
            continue
        parsed = parse_chbmit_summary_file(txt_file)
        for edf_name, intervals in parsed.items():
            all_annotations.setdefault(edf_name, []).extend(intervals)

    for key in all_annotations:
        all_annotations[key] = sorted(all_annotations[key], key=lambda x: x[0])

    return all_annotations


# -----------------------------
# Siena parsing
# -----------------------------

def parse_siena_annotation_file(txt_path: str | Path) -> Dict[str, List[SeizureInterval]]:
    """
    Tolerant parser for Siena seizure text files such as Seizures-list-PN00.txt.
    """
    txt_path = Path(txt_path)
    out: Dict[str, List[SeizureInterval]] = {}

    current_file: Optional[str] = None
    pending_start: Optional[float] = None

    lines = txt_path.read_text(errors="ignore").splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()

        edf_match = re.search(r"([A-Za-z0-9_\-]+\.edf)", line, flags=re.IGNORECASE)
        if edf_match:
            current_file = _normalize_name(edf_match.group(1))
            out.setdefault(current_file, [])
            pending_start = None
            continue

        if current_file is None:
            continue

        if "start" in lower and "seiz" in lower:
            pending_start = _parse_time_to_seconds(line)
            continue

        if "end" in lower and "seiz" in lower:
            end_sec = _parse_time_to_seconds(line)
            if pending_start is not None and end_sec is not None:
                out[current_file].append((pending_start, end_sec))
                pending_start = None
            continue

    for key in out:
        out[key] = sorted(out[key], key=lambda x: x[0])

    return out


def load_all_siena_annotations(dataset_root: str | Path) -> Dict[str, List[SeizureInterval]]:
    root = Path(dataset_root)
    all_annotations: Dict[str, List[SeizureInterval]] = {}

    for txt_file in root.rglob("*.txt"):
        if "seizure" not in txt_file.name.lower():
            continue
        parsed = parse_siena_annotation_file(txt_file)
        for edf_name, intervals in parsed.items():
            all_annotations.setdefault(edf_name, []).extend(intervals)

    for key in all_annotations:
        all_annotations[key] = sorted(all_annotations[key], key=lambda x: x[0])

    return all_annotations


def load_all_annotations(dataset_root: str | Path) -> Dict[str, List[SeizureInterval]]:
    dataset_root = Path(dataset_root)

    combined: Dict[str, List[SeizureInterval]] = {}

    chb_root = dataset_root / "chbmit"
    siena_root = dataset_root / "siena"

    if chb_root.exists():
        chb = load_all_chbmit_annotations(chb_root)
        for edf_name, intervals in chb.items():
            combined.setdefault(edf_name, []).extend(intervals)

    if siena_root.exists():
        siena = load_all_siena_annotations(siena_root)
        for edf_name, intervals in siena.items():
            combined.setdefault(edf_name, []).extend(intervals)

    for key in combined:
        combined[key] = sorted(combined[key], key=lambda x: x[0])

    return combined


# -----------------------------
# preprocessing
# -----------------------------

def drop_mne_duplicate_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    dupes = [ch for ch in raw.ch_names if ch.endswith("-1")]
    if dupes:
        raw = raw.copy().drop_channels(dupes)
    return raw


def preprocess_raw(
    raw: mne.io.BaseRaw,
    dataset_name: str,
    target_sfreq: int,
    l_freq: float,
    h_freq: float,
    notch_freq: float,
) -> mne.io.BaseRaw:
    raw = raw.copy()
    raw.load_data()
    raw = drop_mne_duplicate_channels(raw)

    if dataset_name == "siena":
        try:
            raw.set_eeg_reference("average", projection=False)
        except Exception:
            pass

    if int(raw.info["sfreq"]) != int(target_sfreq):
        raw.resample(target_sfreq)

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    raw.notch_filter(freqs=[notch_freq], verbose="ERROR")
    return raw


def make_fixed_length_epochs(
    raw: mne.io.BaseRaw,
    duration: float,
    overlap: float,
) -> mne.Epochs:
    return mne.make_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=overlap,
        preload=True,
        verbose="ERROR",
    )


def extract_epoch_features(epochs: mne.Epochs) -> pd.DataFrame:
    """
    Basic time-domain features per epoch.
    """
    X = epochs.get_data()  # (n_epochs, n_channels, n_samples)
    sfreq = float(epochs.info["sfreq"])

    rows = []
    for i in range(X.shape[0]):
        epoch = X[i]
        flat = epoch.reshape(-1)

        rows.append(
            {
                "epoch_index": i,
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "min": float(np.min(flat)),
                "max": float(np.max(flat)),
                "range": float(np.max(flat) - np.min(flat)),
                "energy": float(np.sum(flat ** 2)),
                "rms": float(np.sqrt(np.mean(flat ** 2))),
                "abs_mean": float(np.mean(np.abs(flat))),
                "n_channels": int(epoch.shape[0]),
                "n_samples": int(epoch.shape[1]),
                "sfreq": sfreq,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# prediction labeling
# -----------------------------

def get_prediction_label(
    epoch_start: float,
    epoch_end: float,
    seizure_intervals: List[SeizureInterval],
    preictal_horizon_sec: float,
    postictal_exclusion_sec: float,
    interictal_gap_sec: float,
) -> Tuple[str, float]:
    """
    Prediction target:
      preictal -> 1
      interictal -> 0
      ictal/postictal/ambiguous -> NaN (excluded)
    """

    # 1. ictal windows: overlap seizure itself
    for sz_start, sz_end in seizure_intervals:
        if _overlaps(epoch_start, epoch_end, sz_start, sz_end):
            return "ictal", np.nan

    # 2. preictal windows: overlap the preictal horizon before seizure
    for sz_start, _ in seizure_intervals:
        pre_start = sz_start - preictal_horizon_sec
        pre_end = sz_start
        if _overlaps(epoch_start, epoch_end, pre_start, pre_end):
            return "preictal", 1.0

    # 3. postictal exclusion: windows right after seizure
    for _, sz_end in seizure_intervals:
        post_start = sz_end
        post_end = sz_end + postictal_exclusion_sec
        if _overlaps(epoch_start, epoch_end, post_start, post_end):
            return "postictal_exclude", np.nan

    # 4. ambiguous: too close to seizures to confidently call interictal
    for sz_start, sz_end in seizure_intervals:
        amb_start = sz_start - interictal_gap_sec
        amb_end = sz_end + interictal_gap_sec
        if _overlaps(epoch_start, epoch_end, amb_start, amb_end):
            return "ambiguous", np.nan

    # 5. otherwise safely interictal
    return "interictal", 0.0


# -----------------------------
# full builder
# -----------------------------

def build_labeled_static_prediction_dataset(config: StaticBuildConfig) -> pd.DataFrame:
    dataset_root = Path(config.dataset_root)
    edf_files = discover_edf_files(dataset_root)
    annotation_map = load_all_annotations(dataset_root)

    if not edf_files:
        raise FileNotFoundError(f"No EDF files found under {dataset_root}")

    all_rows: List[pd.DataFrame] = []

    for edf_path in edf_files:
        dataset_name = _infer_dataset_name(edf_path)
        edf_name = _normalize_name(edf_path.name)
        seizure_intervals = annotation_map.get(edf_name, [])

        print(f"[INFO] Processing {edf_path}")

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
            raw = preprocess_raw(
                raw=raw,
                dataset_name=dataset_name,
                target_sfreq=config.target_sfreq,
                l_freq=config.l_freq,
                h_freq=config.h_freq,
                notch_freq=config.notch_freq,
            )

            epochs = make_fixed_length_epochs(
                raw=raw,
                duration=config.epoch_duration,
                overlap=config.epoch_overlap,
            )

            features = extract_epoch_features(epochs)

            stride = config.epoch_duration - config.epoch_overlap
            if stride <= 0:
                raise ValueError("epoch_duration - epoch_overlap must be > 0")

            epoch_starts = np.arange(len(features), dtype=float) * stride
            epoch_ends = epoch_starts + config.epoch_duration

            labels: List[str] = []
            targets: List[float] = []

            for start_sec, end_sec in zip(epoch_starts, epoch_ends):
                label, target = get_prediction_label(
                    epoch_start=float(start_sec),
                    epoch_end=float(end_sec),
                    seizure_intervals=seizure_intervals,
                    preictal_horizon_sec=config.preictal_horizon_sec,
                    postictal_exclusion_sec=config.postictal_exclusion_sec,
                    interictal_gap_sec=config.interictal_gap_sec,
                )
                labels.append(label)
                targets.append(target)

            features["dataset"] = dataset_name
            features["edf_path"] = str(edf_path)
            features["edf_name"] = edf_path.name
            features["epoch_start_sec"] = epoch_starts
            features["epoch_end_sec"] = epoch_ends
            features["n_seizures_in_file"] = len(seizure_intervals)
            features["label"] = labels
            features["target"] = targets

            if config.drop_non_prediction_windows:
                features = features.dropna(subset=["target"]).reset_index(drop=True)

            all_rows.append(features)

        except Exception as exc:
            print(f"[WARN] Failed on {edf_path}: {exc}")

    if not all_rows:
        raise RuntimeError("No files were successfully processed.")

    out = pd.concat(all_rows, ignore_index=True)
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out["target"] = out["target"].astype(np.int64)
    return out


def split_features_and_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_cols = [
        "mean",
        "std",
        "min",
        "max",
        "range",
        "energy",
        "rms",
        "abs_mean",
        "n_channels",
        "n_samples",
        "sfreq",
    ]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.int64)
    meta = df.drop(columns=feature_cols).copy()
    return X, y, meta


def save_static_dataset(
    df: pd.DataFrame,
    output_root: str | Path,
    stem: str = "static_prediction_dataset",
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    X, y, meta = split_features_and_target(df)

    df.to_csv(output_root / f"{stem}.csv", index=False)
    np.save(output_root / f"{stem}_X.npy", X)
    np.save(output_root / f"{stem}_y.npy", y)
    meta.to_csv(output_root / f"{stem}_meta.csv", index=False)

    print(f"[INFO] Saved full table to: {output_root / f'{stem}.csv'}")
    print(f"[INFO] Saved X to:         {output_root / f'{stem}_X.npy'}")
    print(f"[INFO] Saved y to:         {output_root / f'{stem}_y.npy'}")
    print(f"[INFO] Saved meta to:      {output_root / f'{stem}_meta.csv'}")