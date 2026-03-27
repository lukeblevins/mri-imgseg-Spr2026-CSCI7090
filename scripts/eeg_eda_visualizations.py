"""
Visualization Script for EEG Dataset Analysis
Dataset: master_eeg_dataset.csv (CHB-MIT + Siena, 354,478 epochs, 10s windows, 256 Hz)
Columns used: dataset, subject_id, label, target, epoch_index, window_start_sec,
              mean, std, min, max, range, energy, rms, abs_mean,
              n_channels, n_samples, sfreq, channel_signature, record_id

Run: python eeg_eda_visualizations.py
Outputs: PNG files saved to ./plots/
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


DATA_PATH = "master_eeg_dataset.csv"  # adjust path if needed
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

PALETTE = {"interictal": "#4C72B0", "preictal": "#DD8452"}
LABEL_ORDER = ["interictal", "preictal"]
FEATURES = ["mean", "std", "min", "max", "range", "energy", "rms", "abs_mean"]

print("Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

chb = df[df["dataset"] == "chbmit"].copy()
siena = df[df["dataset"] == "siena"].copy()


def plot_class_imbalance():
    """Bar chart of interictal vs preictal epoch counts, global and per-dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Class Imbalance: Interictal vs Pre-ictal Epochs",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (title, subset) in zip(
        axes,
        [
            ("All data", df),
            ("CHB-MIT", chb),
            ("Siena", siena),
        ],
    ):
        counts = subset["label"].value_counts().reindex(LABEL_ORDER)
        bars = ax.bar(
            counts.index,
            counts.values,
            color=[PALETTE[l] for l in counts.index],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Epoch count")
        ax.set_yscale("log")
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.1,
                f"{val:,}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ratio = counts["interictal"] / counts.get("preictal", 1)
        ax.set_xlabel(f"Imbalance ratio  {ratio:.0f}:1", fontsize=10)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_class_imbalance.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_class_imbalance.png")


def plot_per_subject_counts():
    """Stacked bar: interictal + preictal epochs per subject, ordered by total."""
    subj_label = (
        df.groupby(["dataset", "subject_id", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    subj_label["total"] = subj_label.get("interictal", 0) + subj_label.get(
        "preictal", 0
    )
    subj_label = subj_label.sort_values(["dataset", "total"], ascending=[True, False])
    subj_label["color"] = subj_label["dataset"].map(
        {"chbmit": "#4e79a7", "siena": "#59a14f"}
    )

    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(subj_label))
    ax.bar(x, subj_label["interictal"], color=PALETTE["interictal"], label="interictal")
    ax.bar(
        x,
        subj_label.get("preictal", 0),
        bottom=subj_label["interictal"],
        color=PALETTE["preictal"],
        label="preictal",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(subj_label["subject_id"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Epoch count (10 s windows)")
    ax.set_title(
        "Epoch Counts per Subject  (stacked interictal + pre-ictal)", fontsize=13
    )
    ax.legend()

    # dataset separator
    n_chb = (subj_label["dataset"] == "chbmit").sum()
    ax.axvline(n_chb - 0.5, color="gray", linestyle="--", linewidth=1.2)
    ax.text(
        n_chb / 2,
        ax.get_ylim()[1] * 0.95,
        "CHB-MIT",
        ha="center",
        fontsize=10,
        color="#4e79a7",
    )
    ax.text(
        n_chb + (len(subj_label) - n_chb) / 2,
        ax.get_ylim()[1] * 0.95,
        "Siena",
        ha="center",
        fontsize=10,
        color="#59a14f",
    )

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_per_subject_epoch_counts.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_per_subject_epoch_counts.png")


def plot_preictal_density():
    """Horizontal bar: % of epochs that are pre-ictal, per subject."""
    grp = df.groupby("subject_id")["target"].agg(["sum", "count"])
    grp["pct"] = grp["sum"] / grp["count"] * 100
    grp["dataset"] = df.groupby("subject_id")["dataset"].first()
    grp = grp.sort_values("pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = grp["dataset"].map({"chbmit": "#4e79a7", "siena": "#59a14f"})
    bars = ax.barh(grp.index, grp["pct"], color=colors, edgecolor="white")
    ax.set_xlabel("Pre-ictal epochs (%)")
    ax.set_title("Pre-ictal Epoch Density per Subject", fontsize=13)

    for bar, (_, row) in zip(bars, grp.iterrows()):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f'{row["sum"]:.0f} epochs',
            va="center",
            fontsize=8,
        )

    patches = [
        mpatches.Patch(color="#4e79a7", label="CHB-MIT"),
        mpatches.Patch(color="#59a14f", label="Siena"),
    ]
    ax.legend(handles=patches)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "03_preictal_density_per_subject.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_preictal_density_per_subject.png")


def plot_feature_distributions():
    """KDE overlay plots for each feature, interictal vs preictal."""
    # Use log-scale for energy/rms (very right-skewed)
    log_features = {"energy", "rms", "abs_mean"}

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        "Feature Distributions: Interictal vs Pre-ictal", fontsize=14, fontweight="bold"
    )

    for ax, feat in zip(axes.flat, FEATURES):
        for label in LABEL_ORDER:
            vals = df.loc[df["label"] == label, feat].dropna()
            if feat in log_features:
                vals = np.log10(vals.clip(lower=1e-20))
            vals.plot.kde(ax=ax, label=label, color=PALETTE[label], linewidth=2)
        ax.set_title(feat + (" (log₁₀)" if feat in log_features else ""), fontsize=11)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "04_feature_distributions.png", dpi=150)
    plt.close(fig)
    print("  ✓ 04_feature_distributions.png")


def plot_violin():
    """Violin + boxplot for each feature, faceted by label."""
    # Clip extreme outliers for readability (keep 1–99th percentile)
    plot_df = df[FEATURES + ["label"]].copy()
    for feat in FEATURES:
        lo, hi = plot_df[feat].quantile([0.01, 0.99])
        plot_df[feat] = plot_df[feat].clip(lo, hi)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        "Violin Plots: Feature Spread by Class", fontsize=14, fontweight="bold"
    )

    for ax, feat in zip(axes.flat, FEATURES):
        sns.violinplot(
            data=plot_df,
            x="label",
            y=feat,
            order=LABEL_ORDER,
            palette=PALETTE,
            inner="box",
            ax=ax,
            linewidth=0.8,
        )
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "05_violin_plots.png", dpi=150)
    plt.close(fig)
    print("  ✓ 05_violin_plots.png")


def plot_correlation_matrix():
    """Heatmap of Pearson correlations between all numeric features."""
    corr = df[FEATURES].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix (lower triangle)", fontsize=13)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "06_feature_correlation_matrix.png", dpi=150)
    plt.close(fig)
    print("  ✓ 06_feature_correlation_matrix.png")


def plot_correlation_by_class():
    """Side-by-side heatmaps: interictal vs preictal feature correlations."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Feature Correlations by Class", fontsize=14, fontweight="bold")

    for ax, label in zip(axes, LABEL_ORDER):
        corr = df[df["label"] == label][FEATURES].corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(label.capitalize(), fontsize=12)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "07_correlation_by_class.png", dpi=150)
    plt.close(fig)
    print("  ✓ 07_correlation_by_class.png")


def plot_feature_evolution():
    """
    For each subject with preictal epochs, plot rolling-mean of a feature
    vs epoch index, with vertical marker at transition to preictal.
    Shows whether features ramp up before seizure.
    """
    subjects_with_pre = (
        df[df["label"] == "preictal"]["subject_id"]
        .value_counts()
        .head(6)  # pick top-6 by preictal count
        .index.tolist()
    )

    feature = "std"  # most interpretable amplitude proxy; change as desired

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        f'Feature Evolution Around Seizure Onset  (feature: "{feature}")',
        fontsize=14,
        fontweight="bold",
    )

    for ax, subj in zip(axes.flat, subjects_with_pre):
        sub = df[df["subject_id"] == subj].sort_values("epoch_index").copy()
        # Smooth with rolling window
        sub["smooth"] = (
            sub[feature].rolling(window=10, min_periods=1, center=True).mean()
        )

        interictal_mask = sub["label"] == "interictal"
        preictal_mask = sub["label"] == "preictal"

        ax.fill_between(
            sub["epoch_index"],
            sub["smooth"],
            where=interictal_mask.values,
            color=PALETTE["interictal"],
            alpha=0.4,
            label="interictal",
        )
        ax.fill_between(
            sub["epoch_index"],
            sub["smooth"],
            where=preictal_mask.values,
            color=PALETTE["preictal"],
            alpha=0.6,
            label="preictal",
        )
        ax.plot(
            sub["epoch_index"], sub["smooth"], color="black", linewidth=0.8, alpha=0.7
        )

        # Mark first preictal epoch
        first_pre_idx = sub.loc[preictal_mask, "epoch_index"].min()
        if not np.isnan(first_pre_idx):
            ax.axvline(
                first_pre_idx,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"onset @ epoch {int(first_pre_idx)}",
            )

        ax.set_title(subj, fontsize=11)
        ax.set_xlabel("Epoch index")
        ax.set_ylabel(feature)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "08_feature_evolution_around_onset.png", dpi=150)
    plt.close(fig)
    print("  ✓ 08_feature_evolution_around_onset.png")


def plot_epoch_feature_heatmap():
    """
    Heatmap with epochs as rows (sorted by label), features as columns.
    Shows cluster separation at a glance.
    Downsampled for speed.
    """
    sample_n = 5000
    sub = df[FEATURES + ["label"]].copy()
    sub = pd.concat(
        [
            sub[sub["label"] == "preictal"],
            sub[sub["label"] == "interictal"].sample(
                min(sample_n, len(sub[sub["label"] == "interictal"])), random_state=42
            ),
        ]
    ).sort_values("label")

    # Z-score normalise each column
    from scipy.stats import zscore

    mat = sub[FEATURES].apply(zscore).clip(-3, 3).values

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=-3, vmax=3)
    plt.colorbar(im, ax=ax, label="Z-score (clipped ±3)")

    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=30, ha="right")
    ax.set_ylabel("Epoch (interictal top, pre-ictal bottom)")
    ax.set_title("Epoch × Feature Heatmap  (z-scored, sampled)", fontsize=13)

    # Draw class boundary line
    n_interictal = (sub["label"] == "interictal").sum()
    ax.axhline(n_interictal - 0.5, color="red", linewidth=2, label="class boundary")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "09_epoch_feature_heatmap.png", dpi=150)
    plt.close(fig)
    print("  ✓ 09_epoch_feature_heatmap.png")


def plot_dataset_comparison():
    """Box plots comparing each feature between CHB-MIT and Siena."""
    plot_df = df[FEATURES + ["dataset"]].copy()
    for feat in FEATURES:
        lo, hi = plot_df[feat].quantile([0.005, 0.995])
        plot_df[feat] = plot_df[feat].clip(lo, hi)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        "CHB-MIT vs Siena: Feature Distributions", fontsize=14, fontweight="bold"
    )

    ds_palette = {"chbmit": "#4e79a7", "siena": "#59a14f"}
    for ax, feat in zip(axes.flat, FEATURES):
        sns.boxplot(
            data=plot_df,
            x="dataset",
            y=feat,
            palette=ds_palette,
            ax=ax,
            width=0.5,
            fliersize=2,
        )
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "10_dataset_comparison.png", dpi=150)
    plt.close(fig)
    print("  ✓ 10_dataset_comparison.png")


def plot_recording_timeline():
    """Gantt chart showing recording duration and pre-ictal periods per subject."""
    subj_stats = []
    for subj, grp in df.groupby("subject_id"):
        total_epochs = len(grp)
        total_sec = total_epochs * 10
        pre_epochs = grp[grp["label"] == "preictal"]
        pre_windows = []
        if len(pre_epochs):
            pre_starts = pre_epochs["window_start_sec"].values
            pre_windows = [(s, s + 10) for s in pre_starts]
        subj_stats.append(
            {
                "subject": subj,
                "dataset": grp["dataset"].iloc[0],
                "total_sec": total_sec,
                "pre_windows": pre_windows,
            }
        )
    subj_stats.sort(key=lambda x: (x["dataset"], x["total_sec"]))

    fig, ax = plt.subplots(figsize=(18, 12))
    yticks, ylabels = [], []

    for i, s in enumerate(subj_stats):
        color = "#4e79a7" if s["dataset"] == "chbmit" else "#59a14f"
        ax.barh(i, s["total_sec"] / 3600, color=color, alpha=0.25, height=0.7)
        for start, end in s["pre_windows"]:
            ax.barh(
                i,
                (end - start) / 3600,
                left=start / 3600,
                color=PALETTE["preictal"],
                height=0.7,
                alpha=0.9,
            )
        yticks.append(i)
        ylabels.append(s["subject"])

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Recording time (hours)")
    ax.set_title(
        "Recording Timeline per Subject  (orange = pre-ictal epochs)", fontsize=13
    )

    patches = [
        mpatches.Patch(color="#4e79a7", alpha=0.4, label="CHB-MIT recording"),
        mpatches.Patch(color="#59a14f", alpha=0.4, label="Siena recording"),
        mpatches.Patch(color=PALETTE["preictal"], label="Pre-ictal window"),
    ]
    ax.legend(handles=patches, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "11_recording_timeline.png", dpi=150)
    plt.close(fig)
    print("  ✓ 11_recording_timeline.png")


def plot_channel_schema():
    """Count plot of n_channels per subject — flags schema inconsistencies."""
    grp = df.groupby(["subject_id", "n_channels"]).size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(14, 5))
    for n_ch, color in {22: "#4e79a7", 23: "#f28e2b", 29: "#59a14f"}.items():
        sub = grp[grp["n_channels"] == n_ch]
        ax.bar(
            sub["subject_id"],
            sub["count"],
            label=f"{n_ch} channels",
            color=color,
            alpha=0.8,
        )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Epoch count")
    ax.set_title("n_channels per Subject  (consistency check)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "12_channel_schema_check.png", dpi=150)
    plt.close(fig)
    print("  ✓ 12_channel_schema_check.png")


def plot_std_histogram():
    """Histogram of epoch std (mean EEG amplitude) split by label, log x-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Epoch Amplitude (std) Distribution by Class", fontsize=13, fontweight="bold"
    )

    for ax, (title, subset) in zip(axes, [("All data", df), ("CHB-MIT only", chb)]):
        for label in LABEL_ORDER:
            vals = subset[subset["label"] == label]["std"].dropna()
            vals = np.log10(vals.clip(lower=1e-20))
            ax.hist(
                vals,
                bins=80,
                alpha=0.55,
                color=PALETTE[label],
                label=label,
                density=True,
            )
        ax.set_xlabel("log₁₀(std) — proxy for amplitude")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "13_std_histogram.png", dpi=150)
    plt.close(fig)
    print("  ✓ 13_std_histogram.png")


def plot_scatter_2d():
    """2D scatter of two strong features, sampled for speed."""
    n_pre = df[df["label"] == "preictal"]
    n_intr = df[df["label"] == "interictal"].sample(
        min(5000, len(df[df["label"] == "interictal"])), random_state=42
    )
    plot_df = pd.concat([n_pre, n_intr])

    fig, ax = plt.subplots(figsize=(10, 7))
    for label in LABEL_ORDER:
        sub = plot_df[plot_df["label"] == label]
        ax.scatter(
            np.log10(sub["energy"].clip(lower=1e-20)),
            np.log10(sub["std"].clip(lower=1e-20)),
            c=PALETTE[label],
            label=label,
            alpha=0.4,
            s=8,
        )

    ax.set_xlabel("log₁₀(energy)")
    ax.set_ylabel("log₁₀(std)")
    ax.set_title("Energy vs Std  (sampled, log-scaled)", fontsize=13)
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "14_scatter_energy_vs_std.png", dpi=150)
    plt.close(fig)
    print("  ✓ 14_scatter_energy_vs_std.png")


if __name__ == "__main__":
    print("\nGenerating visualizations …\n")
    plot_class_imbalance()
    plot_per_subject_counts()
    plot_preictal_density()
    plot_feature_distributions()
    plot_violin()
    plot_correlation_matrix()
    plot_correlation_by_class()
    plot_feature_evolution()
    plot_epoch_feature_heatmap()
    plot_dataset_comparison()
    plot_recording_timeline()
    plot_channel_schema()
    plot_std_histogram()
    plot_scatter_2d()
    print(f"\nAll plots saved to ./{PLOTS_DIR}/")
