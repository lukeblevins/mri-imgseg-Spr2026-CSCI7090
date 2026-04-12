from __future__ import annotations

import re
import time
import traceback
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

    preictal_horizon_sec: float = 600.0
    postictal_exclusion_sec: float = 1800.0
    interictal_gap_sec: float = 300.0

    drop_non_prediction_windows: bool = True
    feature_version: str = "v1"
    channel_schema: str = "raw_mne_channels"
    warn_on_missing_annotations: bool = True

    # practical large-file safety controls
    max_file_size_mb: Optional[float] = 1000.0
    skip_files_without_annotations: bool = False
    verbose_timing: bool = True


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


def _line_looks_like_start(line_lower: str) -> bool:
    seizure_tokens = ("seiz", "sz", "crisis")
    start_tokens = ("start", "onset", "begin", "beginning")
    return any(tok in line_lower for tok in seizure_tokens) and any(tok in line_lower for tok in start_tokens)


def _line_looks_like_end(line_lower: str) -> bool:
    seizure_tokens = ("seiz", "sz", "crisis")
    end_tokens = ("end", "offset", "stop", "ending")
    return any(tok in line_lower for tok in seizure_tokens) and any(tok in line_lower for tok in end_tokens)


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _infer_dataset_name(edf_path: Path) -> str:
    path_str = str(edf_path).lower()
    if "chbmit" in path_str:
        return "chbmit"
    if "siena" in path_str:
        return "siena"
    raise ValueError(f"Could not infer dataset from path: {edf_path}")


def _infer_subject_id(edf_path: Path, dataset_name: str) -> str:
    parts = [p.lower() for p in edf_path.parts]

    if dataset_name == "chbmit":
        for part in parts:
            if re.fullmatch(r"chb\d+", part):
                return part

    if dataset_name == "siena":
        for part in parts:
            if re.fullmatch(r"pn\d+", part):
                return part
        for part in parts:
            if re.search(r"pn\d+", part):
                match = re.search(r"(pn\d+)", part)
                if match:
                    return match.group(1)
        parent = edf_path.parent.name.lower()
        if parent:
            return parent

    return "unknown"


def _infer_session_id(edf_path: Path) -> str:
    return edf_path.stem.lower()


def _make_record_id(
    dataset_name: str,
    subject_id: str,
    session_id: str,
    epoch_index: int,
) -> str:
    return f"{dataset_name}:{subject_id}:{session_id}:epoch_{epoch_index:06d}"


def _channel_signature(raw: mne.io.BaseRaw) -> str:
    return "|".join(ch.lower() for ch in raw.ch_names)


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


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
    txt_path = Path(txt_path)
    out: Dict[str, List[SeizureInterval]] = {}

    current_file: Optional[str] = None
    pending_start: Optional[float] = None
    saw_possible_siena_content = False

    lines = txt_path.read_text(errors="ignore").splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()

        edf_match = re.search(r"([A-Za-z0-9_\-]+\.edf)", line, flags=re.IGNORECASE)
        if edf_match:
            current_file = _normalize_name(edf_match.group(1))
            out.setdefault(current_file, [])
            pending_start = None
            saw_possible_siena_content = True
            continue

        if any(tok in lower for tok in ("seiz", "onset", "offset", "crisis", "edf")):
            saw_possible_siena_content = True

        if current_file is None:
            continue

        if _line_looks_like_start(lower):
            pending_start = _parse_time_to_seconds(line)
            continue

        if _line_looks_like_end(lower):
            end_sec = _parse_time_to_seconds(line)
            if pending_start is not None and end_sec is not None:
                if end_sec >= pending_start:
                    out[current_file].append((pending_start, end_sec))
                pending_start = None
            continue

    for key in out:
        out[key] = sorted(out[key], key=lambda x: x[0])

    out = {k: v for k, v in out.items() if v}

    if saw_possible_siena_content and not out:
        print(f"[DEBUG] Siena annotation file had possible content but no parsed intervals: {txt_path}", flush=True)

    return out


def load_all_siena_annotations(dataset_root: str | Path) -> Dict[str, List[SeizureInterval]]:
    root = Path(dataset_root)
    all_annotations: Dict[str, List[SeizureInterval]] = {}

    for txt_file in root.rglob("*.txt"):
        try:
            print(f"[DEBUG] Reading Siena annotation file: {txt_file}", flush=True)
            parsed = parse_siena_annotation_file(txt_file)

            if parsed:
                print(
                    f"[DEBUG] Parsed {sum(len(v) for v in parsed.values())} Siena seizure intervals from {txt_file}",
                    flush=True,
                )

            for edf_name, intervals in parsed.items():
                all_annotations.setdefault(edf_name, []).extend(intervals)

        except Exception as exc:
            print(f"[WARN] Failed to parse Siena annotation file {txt_file}: {exc}", flush=True)
            traceback.print_exc()
            continue

    for key in all_annotations:
        all_annotations[key] = sorted(all_annotations[key], key=lambda x: x[0])

    print(f"[DEBUG] Total Siena annotation EDF entries loaded: {len(all_annotations)}", flush=True)
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


def keep_only_eeg_channels_if_possible(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw = raw.copy()
    try:
        eeg_picks = mne.pick_types(
            raw.info,
            eeg=True,
            meg=False,
            ecg=False,
            eog=False,
            emg=False,
            stim=False,
            misc=False,
        )
        if len(eeg_picks) > 0 and len(eeg_picks) < len(raw.ch_names):
            raw.pick(eeg_picks)
    except Exception:
        pass
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
    raw = keep_only_eeg_channels_if_possible(raw)

    if dataset_name == "siena":
        try:
            raw.set_eeg_reference("average", projection=False)
        except Exception as exc:
            print(f"[WARN] Could not set average EEG reference for Siena file: {exc}", flush=True)

    if int(raw.info["sfreq"]) != int(target_sfreq):
        raw.resample(target_sfreq)

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

    try:
        raw.notch_filter(freqs=[notch_freq], verbose="ERROR")
    except Exception as exc:
        print(f"[WARN] Notch filter failed: {exc}", flush=True)

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
    X = epochs.get_data()
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
    for sz_start, sz_end in seizure_intervals:
        if _overlaps(epoch_start, epoch_end, sz_start, sz_end):
            return "ictal", np.nan

    for sz_start, _ in seizure_intervals:
        pre_start = sz_start - preictal_horizon_sec
        pre_end = sz_start
        if _overlaps(epoch_start, epoch_end, pre_start, pre_end):
            return "preictal", 1.0

    for _, sz_end in seizure_intervals:
        post_start = sz_end
        post_end = sz_end + postictal_exclusion_sec
        if _overlaps(epoch_start, epoch_end, post_start, post_end):
            return "postictal_exclude", np.nan

    for sz_start, sz_end in seizure_intervals:
        amb_start = sz_start - interictal_gap_sec
        amb_end = sz_end + interictal_gap_sec
        if _overlaps(epoch_start, epoch_end, amb_start, amb_end):
            return "ambiguous", np.nan

    return "interictal", 0.0


# -----------------------------
# canonical schema helpers
# -----------------------------

CANONICAL_COLUMNS = [
    "source_type",
    "dataset",
    "subject_id",
    "session_id",
    "record_id",
    "edf_path",
    "source_file_name",
    "epoch_index",
    "window_start_sec",
    "window_end_sec",
    "duration_sec",
    "label",
    "target",
    "n_seizures_in_file",
    "feature_version",
    "channel_schema",
    "channel_signature",
    "annotation_source",
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


def ensure_canonical_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[CANONICAL_COLUMNS]


def merge_dataset_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    valid_frames = [ensure_canonical_schema(df) for df in frames if df is not None and not df.empty]
    if not valid_frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    merged = pd.concat(valid_frames, ignore_index=True)
    merged = merged.sort_values(
        by=["source_type", "dataset", "subject_id", "session_id", "window_start_sec"],
        kind="stable",
    ).reset_index(drop=True)
    return merged


# -----------------------------
# static builder
# -----------------------------

def build_labeled_static_prediction_dataset(config: StaticBuildConfig) -> pd.DataFrame:
    dataset_root = Path(config.dataset_root)
    edf_files = discover_edf_files(dataset_root)

    print("[DEBUG] About to load annotations", flush=True)
    annotation_map = load_all_annotations(dataset_root)
    print("[DEBUG] Finished loading annotations", flush=True)
    print(f"[DEBUG] Total annotation keys loaded: {len(annotation_map)}", flush=True)

    if not edf_files:
        raise FileNotFoundError(f"No EDF files found under {dataset_root}")

    all_rows: List[pd.DataFrame] = []

    print(f"[DEBUG] Total EDF files found: {len(edf_files)}", flush=True)
    for edf_path in edf_files:
        print(f"[DEBUG] Next EDF path: {edf_path}", flush=True)
        dataset_name = _infer_dataset_name(edf_path)
        print(f"[DEBUG] Dataset inferred: {dataset_name}", flush=True)

        try:
            size_mb = _file_size_mb(edf_path)
            print(f"[DEBUG] EDF file size: {size_mb:.2f} MB", flush=True)

            if config.max_file_size_mb is not None and size_mb > config.max_file_size_mb:
                print(
                    f"[WARN] Skipping oversized EDF ({size_mb:.2f} MB > {config.max_file_size_mb:.2f} MB): {edf_path}",
                    flush=True,
                )
                continue
        except Exception as exc:
            print(f"[WARN] Could not determine file size for {edf_path}: {exc}", flush=True)

        edf_name = _normalize_name(edf_path.name)
        seizure_intervals = annotation_map.get(edf_name, [])
        subject_id = _infer_subject_id(edf_path, dataset_name)
        session_id = _infer_session_id(edf_path)

        if config.warn_on_missing_annotations and edf_name not in annotation_map:
            print(f"[WARN] No annotation entry found for {edf_name}", flush=True)

        if config.skip_files_without_annotations and edf_name not in annotation_map:
            print(f"[WARN] Skipping file without annotation entry: {edf_path}", flush=True)
            continue

        print(f"[INFO] Processing {edf_path}", flush=True)

        try:
            t0 = time.time()
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
            if config.verbose_timing:
                print(f"[DEBUG] read_raw_edf took {time.time() - t0:.2f}s", flush=True)

            print(
                f"[DEBUG] Loaded raw EDF: sfreq={raw.info['sfreq']}, n_channels={len(raw.ch_names)}",
                flush=True,
            )

            t0 = time.time()
            raw = preprocess_raw(
                raw=raw,
                dataset_name=dataset_name,
                target_sfreq=config.target_sfreq,
                l_freq=config.l_freq,
                h_freq=config.h_freq,
                notch_freq=config.notch_freq,
            )
            if config.verbose_timing:
                print(f"[DEBUG] preprocess_raw took {time.time() - t0:.2f}s", flush=True)

            print(
                f"[DEBUG] After preprocessing: sfreq={raw.info['sfreq']}, n_channels={len(raw.ch_names)}",
                flush=True,
            )

            t0 = time.time()
            epochs = make_fixed_length_epochs(
                raw=raw,
                duration=config.epoch_duration,
                overlap=config.epoch_overlap,
            )
            if config.verbose_timing:
                print(f"[DEBUG] make_fixed_length_epochs took {time.time() - t0:.2f}s", flush=True)

            t0 = time.time()
            features = extract_epoch_features(epochs)
            if config.verbose_timing:
                print(f"[DEBUG] extract_epoch_features took {time.time() - t0:.2f}s", flush=True)

            stride = config.epoch_duration - config.epoch_overlap
            if stride <= 0:
                raise ValueError("epoch_duration - epoch_overlap must be > 0")

            epoch_starts = np.arange(len(features), dtype=float) * stride
            epoch_ends = epoch_starts + config.epoch_duration

            labels: List[str] = []
            targets: List[float] = []
            record_ids: List[str] = []

            for epoch_index, (start_sec, end_sec) in enumerate(zip(epoch_starts, epoch_ends)):
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
                record_ids.append(
                    _make_record_id(
                        dataset_name=dataset_name,
                        subject_id=subject_id,
                        session_id=session_id,
                        epoch_index=epoch_index,
                    )
                )

            features["source_type"] = "static"
            features["dataset"] = dataset_name
            features["subject_id"] = subject_id
            features["session_id"] = session_id
            features["record_id"] = record_ids
            features["edf_path"] = str(edf_path)
            features["source_file_name"] = edf_path.name
            features["window_start_sec"] = epoch_starts
            features["window_end_sec"] = epoch_ends
            features["duration_sec"] = config.epoch_duration
            features["n_seizures_in_file"] = len(seizure_intervals)
            features["label"] = labels
            features["target"] = targets
            features["feature_version"] = config.feature_version
            features["channel_schema"] = config.channel_schema
            features["channel_signature"] = _channel_signature(raw)
            features["annotation_source"] = dataset_name

            if config.drop_non_prediction_windows:
                features = features.dropna(subset=["target"]).reset_index(drop=True)

            all_rows.append(ensure_canonical_schema(features))

            # free some large objects before next file
            del epochs
            del raw

        except Exception as exc:
            print(f"[WARN] Failed on {edf_path}: {exc}", flush=True)
            traceback.print_exc()

    if not all_rows:
        raise RuntimeError("No files were successfully processed.")

    out = pd.concat(all_rows, ignore_index=True)
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    out["target"] = out["target"].astype(np.int64)

    return out


def build_static_only_master_dataframe(config: StaticBuildConfig) -> pd.DataFrame:
    static_df = build_labeled_static_prediction_dataset(config)
    return merge_dataset_frames(static_df)


def append_realtime_dataframe(master_df: pd.DataFrame, realtime_df: pd.DataFrame) -> pd.DataFrame:
    return merge_dataset_frames(master_df, realtime_df)


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


def summarize_dataset(df: pd.DataFrame) -> None:
    print("\n[INFO] Dataset breakdown:")
    print(df["dataset"].value_counts(dropna=False))

    print("\n[INFO] Label counts:")
    print(df["label"].value_counts(dropna=False))

    print("\n[INFO] Target counts:")
    print(df["target"].value_counts(dropna=False))

    print(f"\n[INFO] Total rows: {len(df)}")


def save_dataset_bundle(
    df: pd.DataFrame,
    output_root: str | Path,
    stem: str = "master_eeg_dataset",
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    X, y, meta = split_features_and_target(df)

    df.to_csv(output_root / f"{stem}.csv", index=False)

    try:
        df.to_parquet(output_root / f"{stem}.parquet", index=False)
        print(f"[INFO] Saved full table to: {output_root / f'{stem}.parquet'}")
    except Exception as exc:
        print(f"[WARN] Could not save parquet file: {exc}")

    np.save(output_root / f"{stem}_X.npy", X)
    np.save(output_root / f"{stem}_y.npy", y)
    meta.to_csv(output_root / f"{stem}_meta.csv", index=False)

    print(f"[INFO] Saved full table to: {output_root / f'{stem}.csv'}")
    print(f"[INFO] Saved X to:         {output_root / f'{stem}_X.npy'}")
    print(f"[INFO] Saved y to:         {output_root / f'{stem}_y.npy'}")
    print(f"[INFO] Saved meta to:      {output_root / f'{stem}_meta.csv'}")