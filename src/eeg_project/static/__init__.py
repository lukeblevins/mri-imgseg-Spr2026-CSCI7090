"""Static EEG preprocessing and feature workflows."""

from .workflow import (
    extract_epoch_features,
    get_example_dataset_paths,
    make_epochs,
    preprocess_eeg,
    run_static_feature_workflow,
)

__all__ = [
    "extract_epoch_features",
    "get_example_dataset_paths",
    "make_epochs",
    "preprocess_eeg",
    "run_static_feature_workflow",
]
