from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .builder import RealtimePaths, get_realtime_paths

PREICTAL_SEC = 10 * 60
POSTICTAL_EXCLUSION_SEC = 30 * 60
INTERICTAL_GAP_SEC = 5 * 60
KEEP_ONLY_QC_PASS = True
TRAINING_LABELS = {"preictal", "interictal"}


@dataclass
class SeizureEvent:
    session_id: str
    seizure_start: float
    seizure_end: float


def safe_json_load(value):
    if pd.isna(value):
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def load_manifest(manifest_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    for column in ["channel_order", "qc_details", "skipped_pairs"]:
        if column in df.columns:
            df[column] = df[column].apply(safe_json_load)

    for column in ["timestamp_start", "timestamp_end", "fs", "n_channels", "n_samples"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "qc_pass" in df.columns:
        df["qc_pass"] = (
            df["qc_pass"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )

    return df


def load_annotations(annotations_path: Union[str, Path]) -> pd.DataFrame:
    annotations = pd.read_csv(annotations_path)
    required = {"session_id", "seizure_start", "seizure_end"}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Missing required annotation columns: {missing}")

    annotations["seizure_start"] = pd.to_numeric(annotations["seizure_start"], errors="coerce")
    annotations["seizure_end"] = pd.to_numeric(annotations["seizure_end"], errors="coerce")
    annotations = annotations.dropna(subset=["session_id", "seizure_start", "seizure_end"]).copy()
    annotations = annotations.sort_values(["session_id", "seizure_start"]).reset_index(drop=True)
    return annotations


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and a_end > b_start


def label_window(
    window_start: float,
    window_end: float,
    seizure_starts: list[float],
    seizure_ends: list[float],
    *,
    preictal_sec: int = PREICTAL_SEC,
    postictal_exclusion_sec: int = POSTICTAL_EXCLUSION_SEC,
    interictal_gap_sec: int = INTERICTAL_GAP_SEC,
) -> str:
    for seizure_start, seizure_end in zip(seizure_starts, seizure_ends):
        if overlaps(window_start, window_end, seizure_start, seizure_end):
            return "ictal"

    for seizure_start in seizure_starts:
        if seizure_start - preictal_sec <= window_start < seizure_start:
            return "preictal"

    for seizure_end in seizure_ends:
        if seizure_end <= window_start < seizure_end + postictal_exclusion_sec:
            return "exclude"

    far_from_all = True
    for seizure_start, seizure_end in zip(seizure_starts, seizure_ends):
        if abs(window_start - seizure_start) < interictal_gap_sec:
            far_from_all = False
            break
        if abs(window_end - seizure_end) < interictal_gap_sec:
            far_from_all = False
            break

    if far_from_all:
        return "interictal"

    return "exclude"


def label_manifest(manifest_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
    if "session_id" not in manifest_df.columns:
        raise ValueError("Manifest must contain a session_id column for session-aware labeling.")

    labeled_rows: list[dict] = []
    for session_id, session_df in manifest_df.groupby("session_id"):
        annotations = annotations_df[annotations_df["session_id"] == session_id]
        seizure_starts = annotations["seizure_start"].tolist()
        seizure_ends = annotations["seizure_end"].tolist()

        for _, row in session_df.iterrows():
            row_dict = row.to_dict()
            if not seizure_starts:
                row_dict["label"] = "interictal"
                row_dict["label_reason"] = "no_seizures_in_session"
            else:
                row_dict["label"] = label_window(
                    window_start=row_dict["timestamp_start"],
                    window_end=row_dict["timestamp_end"],
                    seizure_starts=seizure_starts,
                    seizure_ends=seizure_ends,
                )
                row_dict["label_reason"] = "rule_based_time_alignment"
            labeled_rows.append(row_dict)

    labeled_df = pd.DataFrame(labeled_rows)
    return labeled_df.sort_values(["session_id", "timestamp_start"]).reset_index(drop=True)


def build_training_manifest(
    labeled_df: pd.DataFrame,
    *,
    keep_only_qc_pass: bool = KEEP_ONLY_QC_PASS,
    training_labels: Optional[set[str]] = None,
) -> pd.DataFrame:
    training_labels = training_labels or TRAINING_LABELS
    df = labeled_df.copy()

    if keep_only_qc_pass and "qc_pass" in df.columns:
        df = df[df["qc_pass"]].copy()

    df = df[df["label"].isin(training_labels)].copy()
    df["target"] = (df["label"] == "preictal").astype(int)
    df["window_duration_sec"] = df["timestamp_end"] - df["timestamp_start"]
    df["window_midpoint"] = (df["timestamp_start"] + df["timestamp_end"]) / 2.0
    return df.reset_index(drop=True)


def load_feature_matrix(training_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([np.load(path) for path in training_df["feature_path"]])
    y = training_df["target"].values.astype(int)
    return x, y


def load_raw_window_matrix(training_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([np.load(path, allow_pickle=True)["window"] for path in training_df["file_path"]])
    y = training_df["target"].values.astype(int)
    return x, y


def summarize_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    summary = {
        "dataset": name,
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values_total": int(df.isna().sum().sum()),
    }

    if "label" in df.columns:
        for label, count in df["label"].value_counts(dropna=False).to_dict().items():
            summary[f"label_{label}"] = int(count)

    if "target" in df.columns:
        for target, count in df["target"].value_counts(dropna=False).to_dict().items():
            summary[f"target_{target}"] = int(count)

    return pd.DataFrame([summary])


def run_phase2_labeling(
    *,
    output_root: Optional[Union[str, Path]] = None,
    keep_only_qc_pass: bool = KEEP_ONLY_QC_PASS,
    training_labels: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths: RealtimePaths = get_realtime_paths(output_root, create=True)
    manifest_df = load_manifest(paths.manifest_path)
    annotations_df = load_annotations(paths.annotations_path)
    labeled_df = label_manifest(manifest_df, annotations_df)
    training_df = build_training_manifest(
        labeled_df,
        keep_only_qc_pass=keep_only_qc_pass,
        training_labels=training_labels,
    )

    labeled_df.to_csv(paths.labeled_manifest_path, index=False)
    training_df.to_csv(paths.training_manifest_path, index=False)

    print(f"Saved labeled manifest: {paths.labeled_manifest_path}")
    print(f"Saved training manifest: {paths.training_manifest_path}")
    print("\n--- Summary: Raw Manifest ---")
    print(summarize_dataframe(manifest_df, "raw_manifest").to_string(index=False))
    print("\n--- Summary: Labeled Manifest ---")
    print(summarize_dataframe(labeled_df, "labeled_manifest").to_string(index=False))
    print("\n--- Summary: Training Manifest ---")
    print(summarize_dataframe(training_df, "training_manifest").to_string(index=False))

    return manifest_df, annotations_df, labeled_df, training_df
