from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne


DATA_DIRS = [
    Path("data/chbmit"),
    Path("data/siena"),
]

OUTPUT_DIR = Path("artifacts/static/raw_windows")
MANIFEST_PATH = Path("artifacts/static/raw_windows_manifest.csv")

WINDOW_SEC = 10
SFREQ_TARGET = 256
LOW_FREQ = 0.5
HIGH_FREQ = 40.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_first_number(text):
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    return float(nums[0]) if nums else None


def get_dataset_type(edf_path):
    path_str = str(edf_path).lower()

    if "chbmit" in path_str:
        return "chbmit"
    if "siena" in path_str:
        return "siena"

    return "unknown"


def get_chbmit_intervals(edf_path):
    """
    Example:
    data/chbmit/chb01/chb01_03.edf
    data/chbmit/chb01/chb01-summary.txt
    """

    subject_id = edf_path.parent.name
    summary_path = edf_path.parent / f"{subject_id}-summary.txt"

    intervals = []

    if not summary_path.exists():
        print(f"[WARN] Missing CHB-MIT summary file: {summary_path}")
        return intervals

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_file = None
    start = None

    for line in lines:
        lower = line.lower()

        if "file name" in lower:
            current_file = line.split(":", 1)[1].strip()

        if current_file == edf_path.name:
            if "seizure start time" in lower:
                start = extract_first_number(line)

            elif "seizure end time" in lower:
                end = extract_first_number(line)

                if start is not None and end is not None:
                    intervals.append((start, end))
                    start = None

    return intervals


def get_siena_intervals(edf_path):
    """
    Example:
    data/siena/PN01/PN01-1.edf
    data/siena/PN01/Seizures-list-PN01.txt
    """

    patient_id = edf_path.parent.name
    txt_path = edf_path.parent / f"Seizures-list-{patient_id}.txt"

    intervals = []

    if not txt_path.exists():
        print(f"[WARN] Missing Siena seizure list: {txt_path}")
        return intervals

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_file = None
    start = None

    for line in lines:
        lower = line.lower().strip()

        if ".edf" in lower:
            current_file = line.strip()

        if current_file and edf_path.name.lower() in current_file.lower():

            if "start" in lower:
                start = extract_first_number(line)

            elif "end" in lower:
                end = extract_first_number(line)

                if start is not None and end is not None:
                    intervals.append((start, end))
                    start = None

    return intervals


def get_seizure_intervals(edf_path):
    dataset = get_dataset_type(edf_path)

    if dataset == "chbmit":
        return get_chbmit_intervals(edf_path)

    if dataset == "siena":
        return get_siena_intervals(edf_path)

    return []


def window_overlaps_seizure(window_start, window_end, seizure_intervals):
    for seizure_start, seizure_end in seizure_intervals:
        if window_start < seizure_end and window_end > seizure_start:
            return True

    return False


def process_edf(edf_path):
    dataset = get_dataset_type(edf_path)

    print(f"[INFO] Processing {dataset}: {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    raw.resample(SFREQ_TARGET)
    raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)

    data = raw.get_data().astype(np.float32)

    sfreq = int(raw.info["sfreq"])
    ch_names = raw.ch_names

    window_size = WINDOW_SEC * sfreq
    total_samples = data.shape[1]

    seizure_intervals = get_seizure_intervals(edf_path)

    rows = []

    epoch_index = 0
    start_sample = 0

    while start_sample + window_size <= total_samples:
        end_sample = start_sample + window_size

        window = data[:, start_sample:end_sample]

        window_start_sec = start_sample / sfreq
        window_end_sec = end_sample / sfreq

        is_seizure = window_overlaps_seizure(
            window_start_sec,
            window_end_sec,
            seizure_intervals,
        )

        target = 1 if is_seizure else 0
        label = "seizure" if is_seizure else "interictal"

        out_name = (
            f"{dataset}_{edf_path.parent.name}_{edf_path.stem}_"
            f"epoch_{epoch_index:06d}.npz"
        )

        out_path = OUTPUT_DIR / out_name

        np.savez_compressed(
            out_path,
            window=window,
            fs=sfreq,
            channel_order=np.array(ch_names, dtype=object),
            dataset=dataset,
            subject_id=edf_path.parent.name,
            session_id=edf_path.stem,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            target=target,
            label=label,
        )

        rows.append({
            "file_path": str(out_path),
            "source_edf": str(edf_path),
            "source_file_name": edf_path.name,
            "dataset": dataset,
            "subject_id": edf_path.parent.name,
            "session_id": edf_path.stem,
            "epoch_index": epoch_index,
            "window_start_sec": window_start_sec,
            "window_end_sec": window_end_sec,
            "duration_sec": WINDOW_SEC,
            "label": label,
            "target": target,
            "n_channels": window.shape[0],
            "n_samples": window.shape[1],
            "sfreq": sfreq,
            "channel_order": "|".join(ch_names),
        })

        start_sample += window_size
        epoch_index += 1

    return rows


def main():
    all_edf_files = []

    for data_dir in DATA_DIRS:
        found = list(data_dir.rglob("*.edf"))
        print(f"[INFO] Found {len(found)} EDF files in {data_dir}")
        all_edf_files.extend(found)

    all_rows = []

    for edf_path in all_edf_files:
        try:
            rows = process_edf(edf_path)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[ERROR] Failed on {edf_path}: {e}")

    manifest = pd.DataFrame(all_rows)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print("\n[DONE] Saved raw windows to:", OUTPUT_DIR)
    print("[DONE] Saved manifest to:", MANIFEST_PATH)

    if not manifest.empty:
        print("\nClass counts:")
        print(manifest["target"].value_counts())
        print("\nLabel counts:")
        print(manifest["label"].value_counts())


if __name__ == "__main__":
    main()