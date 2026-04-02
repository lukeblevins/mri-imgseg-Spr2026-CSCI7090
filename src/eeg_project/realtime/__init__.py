"""Realtime EEG dataset-building and labeling workflows."""

from .builder import (
    LiveEEGDatasetBuilder,
    RealtimePaths,
    WindowRecord,
    get_realtime_paths,
    run_live_dataset_builder,
)
from .labeling import (
    build_training_manifest,
    label_manifest,
    load_annotations,
    load_feature_matrix,
    load_manifest,
    load_raw_window_matrix,
    run_phase2_labeling,
    summarize_dataframe,
)

__all__ = [
    "LiveEEGDatasetBuilder",
    "RealtimePaths",
    "WindowRecord",
    "build_training_manifest",
    "get_realtime_paths",
    "label_manifest",
    "load_annotations",
    "load_feature_matrix",
    "load_manifest",
    "load_raw_window_matrix",
    "run_live_dataset_builder",
    "run_phase2_labeling",
    "summarize_dataframe",
]
