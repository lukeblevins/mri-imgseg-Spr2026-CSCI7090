
from __future__ import annotations

import time
import os
import math
import numpy as np
import uuid
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import pandas as pd

from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, resample_poly

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

TARGET_FS = 256
WINDOW_SEC = 30
STRIDE_SEC = 5
BUFFER_SEC = 5 * 60

TARGET_BIPOLAR_CHANNELS = [
    "FZ-CZ",
    "CZ-PZ",
    "F3-C3",
    "F4-C4",
    "F7-F3",
    "F4-F8",
    "PO7-OZ",
    "OZ-PO8",
]
OUTPUT_DIR = "live_eeg_dataset"
WINDOWS_DIR = os.path.join(OUTPUT_DIR, "windows")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.csv")

os.makedirs(WINDOWS_DIR, exist_ok=True)

@dataclass
class WindowRecord:
    window_id: str
    session_id: str
    timestamp_end: float
    timestamp_start: float
    file_path: str
    feature_path: str
    fs: int
    n_channels: int
    n_samples: int
    channel_order: List[str]
    qc_pass: bool
    qc_details: Dict[str, bool]
    skipped_pairs: List[str]
    notes: str

OLD_TO_NEW = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "FZ": "FZ",
    "CZ": "CZ",
    "PZ": "PZ",
}

JUNK_EXACT = {
    "EKG", "ECG", "EMG", "PHOTIC", "IBI", "BURSTS", "SUPPR"
}

JUNK_PREFIXES = (
    "DC", "EVENT", "MARK", "TRIG", "STATUS", "ACC", "GYRO", "AUX"
)
def clean_basic_name(name):
    name = name.upper().strip()

    for token in ["EEG ", "EEG", "POL ", "POL", "REF", "-REF", " LE", "-LE"]:
        name = name.replace(token, "")

    name = name.replace("--", "-")
    name = name.replace(" ", "")

    for old, new in OLD_TO_NEW.items():
        name = name.replace(old, new)

    return name


def is_junk_channel(name):
    if name in JUNK_EXACT:
        return True
    return any(name.startswith(prefix) for prefix in JUNK_PREFIXES)


def clean_channel_dict(chunk):
    cleaned = {}

    for raw_name, signal in chunk.items():
        new_name = clean_basic_name(raw_name)

        if is_junk_channel(new_name):
            continue

        if new_name not in cleaned:
            cleaned[new_name] = signal.astype(np.float32)

    return cleaned

def build_bipolar_chunk(
    mono_chunk, target_pairs):
    bipolar = {}
    skipped = []

    for pair in target_pairs:
        left, right = pair.split("-")

        if left in mono_chunk and right in mono_chunk:
            bipolar[pair] = mono_chunk[left] - mono_chunk[right]
        else:
            skipped.append(pair)

    return bipolar, skipped

def reorder_channels(chunk, target_order):
    if not all(ch in chunk for ch in target_order):
        return None

    return np.vstack([chunk[ch] for ch in target_order]).astype(np.float32)

def resample_chunk(data, old_fs, new_fs) -> np.ndarray:
    if old_fs == new_fs:
        return data.astype(np.float32)

    return resample_poly(data, up=new_fs, down=old_fs, axis=1).astype(np.float32)

def design_bandpass(fs, low: float = 0.5, high: float = 40.0, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a

def design_notch(fs: int, freq: float = 60.0, q: float = 30.0):
    b, a = iirnotch(freq / (0.5 * fs), q)
    return b, a

class StreamingFilter:
    def __init__(self, fs: int, n_channels: int):
        self.fs = fs
        self.n_channels = n_channels

        self.bp_b, self.bp_a = design_bandpass(fs, low=0.5, high=40.0, order=4)
        self.notch_b, self.notch_a = design_notch(fs, freq=60.0, q=30.0)

        self.notch_state = np.tile(
            lfilter_zi(self.notch_b, self.notch_a),
            (n_channels, 1)
        )
        self.bp_state = np.tile(
            lfilter_zi(self.bp_b, self.bp_a),
            (n_channels, 1)
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        notch_out = np.zeros_like(data, dtype=np.float32)
        out = np.zeros_like(data, dtype=np.float32)

        for ch in range(self.n_channels):
            notch_out[ch], self.notch_state[ch] = lfilter(
                self.notch_b,
                self.notch_a,
                data[ch],
                zi=self.notch_state[ch]
            )

        for ch in range(self.n_channels):
            out[ch], self.bp_state[ch] = lfilter(
                self.bp_b,
                self.bp_a,
                notch_out[ch],
                zi=self.bp_state[ch]
            )

        return out.astype(np.float32)

class RollingEEGBuffer:
    def __init__(self, n_channels, fs, buffer_sec):
        self.n_channels = n_channels
        self.fs = fs
        self.max_samples = fs * buffer_sec
        self.data = np.zeros((n_channels, 0), dtype=np.float32)

    def append(self, chunk):
        self.data = np.concatenate([self.data, chunk], axis=1)

        if self.data.shape[1] > self.max_samples:
            self.data = self.data[:, -self.max_samples:]

    def has_window(self, window_sec) -> bool:
        return self.data.shape[1] >= int(window_sec * self.fs)

    def latest_window(self, window_sec) -> np.ndarray:
        win_len = int(window_sec * self.fs)
        return self.data[:, -win_len:].copy()

def qc_check(window):
    checks = {}

    checks["has_nan"] = bool(np.isnan(window).any())
    checks["flatline"] = bool(np.any(np.std(window, axis=1) < 1e-6))
    checks["extreme_amplitude"] = bool(np.any(np.max(np.abs(window), axis=1) > 1e4))

    checks["pass"] = not any([
        checks["has_nan"],
        checks["flatline"],
        checks["extreme_amplitude"]
    ])

    return checks

def zscore_per_channel(window, eps: float = 1e-8):
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True)
    return ((window - mean) / (std + eps)).astype(np.float32)

def bandpower_approx(x):
    return float(np.mean(x ** 2))

def line_length(x):
    return float(np.sum(np.abs(np.diff(x))))

def hjorth_activity(x):
    return float(np.var(x))

def hjorth_mobility(x, eps: float = 1e-8):
    dx = np.diff(x)
    return float(np.sqrt(np.var(dx) / (np.var(x) + eps)))

def hjorth_complexity(x, eps: float = 1e-8):
    dx = np.diff(x)
    ddx = np.diff(dx)
    mobility_x = np.sqrt(np.var(dx) / (np.var(x) + eps))
    mobility_dx = np.sqrt(np.var(ddx) / (np.var(dx) + eps))
    return float(mobility_dx / (mobility_x + eps))

def extract_features(window):
    feats = []

    for ch in range(window.shape[0]):
        x = window[ch]

        feats.extend([
            float(np.mean(x)),
            float(np.std(x)),
            float(np.var(x)),
            float(np.max(x)),
            float(np.min(x)),
            bandpower_approx(x),
            line_length(x),
            hjorth_activity(x),
            hjorth_mobility(x),
            hjorth_complexity(x),
        ])

    return np.asarray(feats, dtype=np.float32)

def save_window_example(
    window: np.ndarray,
    session_id: str,
    features: np.ndarray,
    timestamp_end: float,
    fs: int,
    channel_order: List[str],
    qc_pass: bool,
    qc_details: Dict[str, bool],
    skipped_pairs: List[str],
    notes: str = ""
):
    window_id = str(uuid.uuid4())

    window_path = os.path.join(WINDOWS_DIR, f"{window_id}_window.npz")
    feature_path = os.path.join(WINDOWS_DIR, f"{window_id}_features.npy")

    timestamp_start = timestamp_end - WINDOW_SEC

    np.savez_compressed(
        window_path,
        window=window,
        fs=fs,
        channel_order=np.array(channel_order, dtype=object),
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end
    )

    np.save(feature_path, features)

    return WindowRecord(
        window_id=window_id,
        session_id=session_id,
        timestamp_end=timestamp_end,
        timestamp_start=timestamp_start,
        file_path=window_path,
        feature_path=feature_path,
        fs=fs,
        n_channels=window.shape[0],
        n_samples=window.shape[1],
        channel_order=channel_order,
        qc_pass=qc_pass,
        qc_details=qc_details,
        skipped_pairs=skipped_pairs,
        notes=notes
    )

def append_to_manifest(record, manifest_path: str = MANIFEST_PATH):
    row = asdict(record)

    # store nested fields as JSON strings for CSV compatibility
    row["channel_order"] = json.dumps(row["channel_order"])
    row["qc_details"] = json.dumps(row["qc_details"])
    row["skipped_pairs"] = json.dumps(row["skipped_pairs"])

    df_row = pd.DataFrame([row])

    if os.path.exists(manifest_path):
        df_row.to_csv(manifest_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(manifest_path, mode="w", header=True, index=False)

class BrainFlowStreamer:
    def __init__(self, board_id, params: BrainFlowInputParams):
        self.board_id = board_id
        self.params = params
        self.board = BoardShim(board_id, params)

        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.eeg_names = BoardShim.get_eeg_names(board_id)

    def start(self):
        self.board.prepare_session()
        self.board.start_stream()

    def stop(self):
        try:
            self.board.stop_stream()
        finally:
            self.board.release_session()

    def get_chunk_dict(self, duration_sec: float = 1.0) -> Dict[str, np.ndarray]:
        num_points = max(1, int(self.sampling_rate * duration_sec))
        data = self.board.get_current_board_data(num_points)

        if data.shape[1] == 0:
            return {}

        eeg_data = data[self.eeg_channels, :]

        chunk = {}
        for i, ch_name in enumerate(self.eeg_names):
            if i < eeg_data.shape[0]:
                chunk[ch_name] = eeg_data[i].astype(np.float32)

        return chunk

class LiveEEGDatasetBuilder:
    def __init__(
        self,
        target_fs = TARGET_FS,
        target_channels = TARGET_BIPOLAR_CHANNELS
    ):
        self.target_fs = target_fs
        self.target_channels = target_channels
        self.n_channels = len(target_channels)

        self.filter = StreamingFilter(fs=target_fs, n_channels=self.n_channels)
        self.buffer = RollingEEGBuffer(
            n_channels=self.n_channels,
            fs=target_fs,
            buffer_sec=BUFFER_SEC
        )

        self.last_save_time = 0.0

    def process_chunk(self, chunk, chunk_fs, timestamp):

        if not chunk:
            return None

        # 1. Clean channel names
        cleaned = clean_channel_dict(chunk)

        # 2. Convert to bipolar montage
        bipolar, skipped_pairs = build_bipolar_chunk(cleaned, self.target_channels)

        # 3. Reorder channels
        ordered = reorder_channels(bipolar, self.target_channels)
        if ordered is None:
          return None

        # 4. Resample
        ordered = resample_chunk(ordered, chunk_fs, self.target_fs)

        # 5. Filter
        ordered = self.filter.apply(ordered)

        # 6. Append to rolling buffer
        self.buffer.append(ordered)

        # 7. Save only every STRIDE_SEC
        if (timestamp - self.last_save_time) < STRIDE_SEC:
            return None

        # 8. Need a full fixed window first
        if not self.buffer.has_window(WINDOW_SEC):
            return None

        self.last_save_time = timestamp

        # 9. Extract latest window
        window = self.buffer.latest_window(WINDOW_SEC)

        # 10. QC
        qc = qc_check(window)

        # 11. Normalize
        norm_window = zscore_per_channel(window)

        # 12. Features
        feats = extract_features(norm_window)

        # 13. Save
        record = save_window_example(
            window=norm_window,
            session_id="session_01",
            features=feats,
            timestamp_end=timestamp,
            fs=self.target_fs,
            channel_order=self.target_channels,
            qc_pass=qc["pass"],
            qc_details=qc,
            skipped_pairs=skipped_pairs,
            notes="Saved live training window."
        )

        append_to_manifest(record)
        return record

def run_live_dataset_builder(
    board_id: int,
    params: BrainFlowInputParams,
    poll_duration_sec: float = 1.0,
    sleep_sec: float = 0.2
):
    BoardShim.enable_dev_board_logger()

    streamer = BrainFlowStreamer(board_id=board_id, params=params)
    builder = LiveEEGDatasetBuilder()

    print("Starting BrainFlow stream...")
    streamer.start()
    print("Live EEG dataset builder running... Press Ctrl+C to stop.")

    try:
        while True:
            ts = time.time()
            chunk = streamer.get_chunk_dict(duration_sec=poll_duration_sec)

            record = builder.process_chunk(
                chunk=chunk,
                chunk_fs=streamer.sampling_rate,
                timestamp=ts
            )

            if record is not None:
                print(
                    f"Saved window {record.window_id} | "
                    f"start={record.timestamp_start:.2f} | "
                    f"end={record.timestamp_end:.2f} | "
                    f"qc={record.qc_pass}"
                )
            else:
                print("Streaming... no window saved this cycle.")

            time.sleep(sleep_sec)

    except KeyboardInterrupt:
        print("Stopping stream...")

    finally:
        streamer.stop()
        print("Stream stopped.")

if __name__ == "__main__":
    params = BrainFlowInputParams()

    # Start with Synthetic Board for testing
    board_id = BoardIds.SYNTHETIC_BOARD.value

    run_live_dataset_builder(
        board_id=board_id,
        params=params,
        poll_duration_sec=1.0,
        sleep_sec=0.2
    )