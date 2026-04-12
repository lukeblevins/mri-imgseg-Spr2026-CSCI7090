"""Static EEG preprocessing, feature extraction, and labeling workflows."""

from .workflow import (
    extract_epoch_features as workflow_extract_epoch_features,
    get_example_dataset_paths,
    make_epochs,
    preprocess_eeg,
    run_static_feature_workflow,
)

from .target_label import (
    CANONICAL_COLUMNS,
    StaticBuildConfig,
    append_realtime_dataframe,
    build_labeled_static_prediction_dataset,
    build_static_only_master_dataframe,
    discover_edf_files,
    ensure_canonical_schema,
    extract_epoch_features,
    get_prediction_label,
    load_all_annotations,
    merge_dataset_frames,
    save_dataset_bundle,
    split_features_and_target,
    summarize_dataset,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "StaticBuildConfig",
    "append_realtime_dataframe",
    "build_labeled_static_prediction_dataset",
    "build_static_only_master_dataframe",
    "discover_edf_files",
    "ensure_canonical_schema",
    "extract_epoch_features",
    "get_example_dataset_paths",
    "get_prediction_label",
    "load_all_annotations",
    "make_epochs",
    "merge_dataset_frames",
    "preprocess_eeg",
    "run_static_feature_workflow",
    "save_dataset_bundle",
    "split_features_and_target",
    "summarize_dataset",
    "workflow_extract_epoch_features",
]