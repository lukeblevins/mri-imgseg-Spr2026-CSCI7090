from __future__ import annotations

import argparse

from eeg_project.static import (
    StaticBuildConfig,
    build_static_only_master_dataframe,
    save_dataset_bundle,
    summarize_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build merged static seizure prediction dataset"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root containing chbmit/ and/or siena/",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="artifacts/static_merged",
        help="Directory for generated outputs",
    )
    parser.add_argument("--epoch-duration", type=float, default=10.0)
    parser.add_argument("--epoch-overlap", type=float, default=0.0)
    parser.add_argument("--target-sfreq", type=int, default=256)
    parser.add_argument("--l-freq", type=float, default=0.5)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=60.0)
    parser.add_argument("--preictal-horizon-sec", type=float, default=600.0)
    parser.add_argument("--postictal-exclusion-sec", type=float, default=1800.0)
    parser.add_argument("--interictal-gap-sec", type=float, default=300.0)
    parser.add_argument(
        "--keep-non-prediction-windows",
        action="store_true",
        help="Keep ictal/postictal/ambiguous rows instead of dropping them",
    )

    args = parser.parse_args()

    config = StaticBuildConfig(
        dataset_root=args.dataset_root,
        target_sfreq=args.target_sfreq,
        epoch_duration=args.epoch_duration,
        epoch_overlap=args.epoch_overlap,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch_freq=args.notch_freq,
        preictal_horizon_sec=args.preictal_horizon_sec,
        postictal_exclusion_sec=args.postictal_exclusion_sec,
        interictal_gap_sec=args.interictal_gap_sec,
        drop_non_prediction_windows=not args.keep_non_prediction_windows,
        feature_version="v1",
        channel_schema="raw_mne_channels",
    )

    df = build_static_only_master_dataframe(config)
    save_dataset_bundle(df, args.output_root, stem="master_eeg_dataset")
    summarize_dataset(df)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()