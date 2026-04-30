"""End-to-end ML pipeline: dataset build, train/test split, RF training, evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit

from eeg_project.common.paths import resolve_dataset_root
from eeg_project.static.target_label import (
    StaticBuildConfig,
    build_static_only_master_dataframe,
    save_dataset_bundle,
    split_features_and_target,
    summarize_dataset,
)

FEATURE_COLS = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "median",
    "iqr",
    "energy",
    "rms",
    "abs_mean",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "line_length",
    "zero_crossing_rate",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "rel_delta_power",
    "rel_theta_power",
    "rel_alpha_power",
    "rel_beta_power",
    "rel_gamma_power",
    "spectral_entropy",
]


def prepare_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subject-grouped train/test split to prevent data leakage."""
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.int64)
    groups = df["subject_id"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    print(f"[INFO] Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    print(
        f"[INFO] Shared subjects between splits: "
        f"{len(set(groups[train_idx]) & set(groups[test_idx]))}"
    )

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest with GridSearchCV. Falls back to defaults when only one class is present in the training set."""
    train_classes = np.unique(y_train)

    if len(train_classes) < 2:
        print(
            f"[WARN] Training set contains only class(es) {train_classes.tolist()}. "
            "Skipping grid search — add more annotated recordings to enable tuning."
        )
        rf = RandomForestClassifier(
            class_weight="balanced_subsample", random_state=random_state, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        print("[INFO] Fitted default RandomForest (no grid search).")
        return rf

    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [10, 20],
        "min_samples_leaf": [1, 5],
    }

    rf = RandomForestClassifier(
        class_weight="balanced_subsample", random_state=random_state, n_jobs=-1
    )
    grid = GridSearchCV(rf, param_grid, scoring="recall", cv=3, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)

    print(f"[INFO] Best hyperparameters: {grid.best_params_}")
    return grid.best_estimator_


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate model and print metrics. Returns a results dict."""
    y_pred = model.predict(X_test)

    proba_matrix = model.predict_proba(X_test)
    if proba_matrix.shape[1] < 2:
        print("[WARN] Model only saw one class during training — probabilities unavailable.")
        y_proba = np.zeros(len(X_test))
        auc = None
    else:
        y_proba = proba_matrix[:, 1]
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else None

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n[INFO] Model Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    if auc is not None:
        print(f"ROC AUC:  {auc:.4f}")
    else:
        print("ROC AUC:  N/A (only one class in test set)")

    print("\nConfusion Matrix:")
    print(cm)

    seizure_recall = None
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        seizure_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")
        print(f"Seizure Recall: {seizure_recall:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "auc": auc,
        "confusion_matrix": cm,
        "seizure_recall": seizure_recall,
    }


def tune_threshold(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    *,
    target_recall: float = 0.50,
    n_thresholds: int = 100,
) -> dict:
    """Search probability thresholds to hit target seizure recall with fewest false positives."""
    if len(np.unique(y_test)) < 2:
        print("[WARN] Skipping threshold tuning: only one class in test set.")
        return {}

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    rows = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        rows.append(
            {
                "threshold": t,
                "accuracy": accuracy_score(y_test, y_pred_t),
                "precision": precision_score(y_test, y_pred_t, zero_division=0),
                "recall": recall_score(y_test, y_pred_t, zero_division=0),
                "f1": f1_score(y_test, y_pred_t, zero_division=0),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

    threshold_df = pd.DataFrame(rows)
    candidates = threshold_df[threshold_df["recall"] >= target_recall]
    best_row = (
        candidates.sort_values("FP").iloc[0]
        if len(candidates) > 0
        else threshold_df.sort_values("recall", ascending=False).iloc[0]
    )
    best_threshold = float(best_row["threshold"])

    print(f"\n[INFO] Threshold Tuning (target recall ≥ {target_recall})")
    print(f"Selected threshold: {best_threshold:.4f}")
    print(f"Seizure Recall: {best_row['recall']:.4f}  FP: {int(best_row['FP'])}")

    y_pred_best = (y_proba >= best_threshold).astype(int)
    print("\nFinal Classification Report (tuned threshold):")
    print(classification_report(y_test, y_pred_best, zero_division=0))

    return {
        "threshold_df": threshold_df,
        "best_threshold": best_threshold,
        "best_row": best_row,
        "y_pred_tuned": y_pred_best,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end EEG seizure prediction ML pipeline"
    )
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="artifacts/static")
    parser.add_argument("--epoch-duration", type=float, default=10.0)
    parser.add_argument("--epoch-overlap", type=float, default=0.0)
    parser.add_argument("--target-sfreq", type=int, default=256)
    parser.add_argument("--l-freq", type=float, default=0.5)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=60.0)
    parser.add_argument("--preictal-horizon-sec", type=float, default=600.0)
    parser.add_argument("--postictal-exclusion-sec", type=float, default=1800.0)
    parser.add_argument("--interictal-gap-sec", type=float, default=300.0)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--target-recall", type=float, default=0.5)
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip dataset build and load from --output-root instead",
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)

    # Step 1: Build (or load) the dataset
    if args.skip_build:
        csv_path = output_root / "master_eeg_dataset.csv"
        parquet_path = output_root / "master_eeg_dataset.parquet"
        if parquet_path.exists():
            print(f"[INFO] Loading dataset from {parquet_path}")
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            print(f"[INFO] Loading dataset from {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"No dataset found in {output_root}. Run without --skip-build first."
            )
    else:
        dataset_root, _ = resolve_dataset_root(args.dataset_root)
        config = StaticBuildConfig(
            dataset_root=dataset_root,
            target_sfreq=args.target_sfreq,
            epoch_duration=args.epoch_duration,
            epoch_overlap=args.epoch_overlap,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            notch_freq=args.notch_freq,
            preictal_horizon_sec=args.preictal_horizon_sec,
            postictal_exclusion_sec=args.postictal_exclusion_sec,
            interictal_gap_sec=args.interictal_gap_sec,
            feature_version="v2_expanded_eeg",
            channel_schema="raw_mne_channels",
        )
        print("[INFO] Building static dataset...")
        df = build_static_only_master_dataframe(config)
        save_dataset_bundle(df, output_root, stem="master_eeg_dataset")

    summarize_dataset(df)

    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df, test_size=args.test_size
    )

    # Step 3: Train
    print("\n[INFO] Training Random Forest...")
    model = train_random_forest(X_train, y_train)

    # Step 4: Evaluate
    results = evaluate_model(model, X_test, y_test)

    # Step 5: Threshold tuning
    tune_threshold(y_test, results["y_proba"], target_recall=args.target_recall)

    print("\n[INFO] Pipeline complete.")


if __name__ == "__main__":
    main()
