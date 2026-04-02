from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import mne
import numpy as np
import pandas as pd

from eeg_project.common.paths import CHB_RELATIVE_PATH, SIENA_RELATIVE_PATH, resolve_dataset_root

mne.set_log_level("WARNING")


def get_example_dataset_paths(
    dataset_root: Optional[Union[str, Path]] = None,
) -> tuple[Path, Path, Path]:
    resolved_root, _ = resolve_dataset_root(dataset_root)
    return (
        resolved_root,
        resolved_root / CHB_RELATIVE_PATH,
        resolved_root / SIENA_RELATIVE_PATH,
    )


def drop_mne_duplicates(raw: mne.io.BaseRaw) -> None:
    to_drop = [channel for channel in raw.ch_names if channel.endswith("-1")]
    if to_drop:
        raw.drop_channels(to_drop)


def preprocess_eeg(
    raw: mne.io.BaseRaw,
    dataset_name: str,
    *,
    target_sfreq: int = 256,
) -> mne.io.BaseRaw:
    raw = raw.copy()
    raw.load_data()

    non_eeg_keywords = [
        "EKG",
        "ECG",
        "SPO2",
        "HR",
        "PHOTIC",
        "IBI",
        "BURSTS",
        "SUPPR",
        "MK",
    ]
    drop_now = []

    for channel in raw.ch_names:
        channel_upper = channel.upper()
        if any(keyword in channel_upper for keyword in non_eeg_keywords):
            drop_now.append(channel)

    if drop_now:
        raw.drop_channels(drop_now)

    drop_mne_duplicates(raw)

    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq)

    raw.set_channel_types({channel: "eeg" for channel in raw.ch_names})

    if dataset_name.lower() == "siena":
        raw.set_eeg_reference("average", projection=False)

    raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    raw.notch_filter(freqs=60, verbose=False)
    return raw


def make_epochs(raw: mne.io.BaseRaw, *, duration: float = 2.0, overlap: float = 1.0) -> mne.Epochs:
    return mne.make_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=overlap,
        preload=True,
        verbose=False,
    )


def extract_epoch_features(
    epoch_data: np.ndarray,
    dataset_name: str,
    sfreq: float,
    *,
    duration: float = 2.0,
    overlap: float = 1.0,
) -> pd.DataFrame:
    rows = []
    step_sec = duration - overlap

    for epoch_index, epoch in enumerate(epoch_data):
        flat = epoch.flatten()
        rows.append(
            {
                "dataset": dataset_name,
                "epoch_index": epoch_index,
                "epoch_start_sec": epoch_index * step_sec,
                "mean": np.mean(flat),
                "std": np.std(flat),
                "min": np.min(flat),
                "max": np.max(flat),
                "range": np.max(flat) - np.min(flat),
                "energy": np.sum(flat**2),
                "rms": np.sqrt(np.mean(flat**2)),
                "abs_mean": np.mean(np.abs(flat)),
                "channel_count": epoch.shape[0],
                "samples_per_epoch": epoch.shape[1],
                "sampling_rate": sfreq,
            }
        )

    return pd.DataFrame(rows)


def run_static_feature_workflow(
    dataset_root: Optional[Union[str, Path]] = None,
    *,
    duration: float = 2.0,
    overlap: float = 1.0,
    target_sfreq: int = 256,
) -> dict[str, object]:
    resolved_root, chb_path, siena_path = get_example_dataset_paths(dataset_root)

    raw_chb = mne.io.read_raw_edf(chb_path, preload=False, verbose=False)
    raw_siena = mne.io.read_raw_edf(siena_path, preload=False, verbose=False)

    raw_chb_clean = preprocess_eeg(raw_chb, "CHB", target_sfreq=target_sfreq)
    raw_siena_clean = preprocess_eeg(raw_siena, "Siena", target_sfreq=target_sfreq)

    epochs_chb = make_epochs(raw_chb_clean, duration=duration, overlap=overlap)
    epochs_siena = make_epochs(raw_siena_clean, duration=duration, overlap=overlap)

    x_chb = epochs_chb.get_data()
    x_siena = epochs_siena.get_data()

    df_chb = extract_epoch_features(
        x_chb,
        "CHB",
        raw_chb_clean.info["sfreq"],
        duration=duration,
        overlap=overlap,
    )
    df_siena = extract_epoch_features(
        x_siena,
        "Siena",
        raw_siena_clean.info["sfreq"],
        duration=duration,
        overlap=overlap,
    )
    df_all = pd.concat([df_chb, df_siena], ignore_index=True)

    return {
        "dataset_root": resolved_root,
        "chb_path": chb_path,
        "siena_path": siena_path,
        "raw_chb_clean": raw_chb_clean,
        "raw_siena_clean": raw_siena_clean,
        "epochs_chb": epochs_chb,
        "epochs_siena": epochs_siena,
        "x_chb": x_chb,
        "x_siena": x_siena,
        "features_chb": df_chb,
        "features_siena": df_siena,
        "features": df_all,
    }
