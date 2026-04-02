from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, resample_poly

from eeg_project.common.paths import get_realtime_output_root

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


@dataclass(frozen=True)
class RealtimePaths:
    output_root: Path
    windows_dir: Path
    manifest_path: Path
    annotations_path: Path
    labeled_manifest_path: Path
    training_manifest_path: Path


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
    channel_order: list[str]
    qc_pass: bool
    qc_details: dict[str, bool]
    skipped_pairs: list[str]
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

JUNK_EXACT = {"EKG", "ECG", "EMG", "PHOTIC", "IBI", "BURSTS", "SUPPR"}
JUNK_PREFIXES = ("DC", "EVENT", "MARK", "TRIG", "STATUS", "ACC", "GYRO", "AUX")


def get_realtime_paths(
    output_root: Optional[Union[str, Path]] = None,
    *,
    create: bool = True,
) -> RealtimePaths:
    root = get_realtime_output_root(output_root, create=create)
    windows_dir = root / "windows"
    if create:
        windows_dir.mkdir(parents=True, exist_ok=True)

    return RealtimePaths(
        output_root=root,
        windows_dir=windows_dir,
        manifest_path=root / "manifest.csv",
        annotations_path=root / "seizure_annotations.csv",
        labeled_manifest_path=root / "labeled_manifest.csv",
        training_manifest_path=root / "training_manifest.csv",
    )


def clean_basic_name(name: str) -> str:
    name = name.upper().strip()
    for token in ["EEG ", "EEG", "POL ", "POL", "REF", "-REF", " LE", "-LE"]:
        name = name.replace(token, "")

    name = name.replace("--", "-")
    name = name.replace(" ", "")

    for old, new in OLD_TO_NEW.items():
        name = name.replace(old, new)

    return name


def is_junk_channel(name: str) -> bool:
    if name in JUNK_EXACT:
        return True
    return any(name.startswith(prefix) for prefix in JUNK_PREFIXES)


def clean_channel_dict(chunk: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cleaned: dict[str, np.ndarray] = {}
    for raw_name, signal in chunk.items():
        new_name = clean_basic_name(raw_name)
        if is_junk_channel(new_name):
            continue
        if new_name not in cleaned:
            cleaned[new_name] = signal.astype(np.float32)
    return cleaned


def build_bipolar_chunk(
    mono_chunk: dict[str, np.ndarray],
    target_pairs: list[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    bipolar: dict[str, np.ndarray] = {}
    skipped: list[str] = []

    for pair in target_pairs:
        left, right = pair.split("-")
        if left in mono_chunk and right in mono_chunk:
            bipolar[pair] = mono_chunk[left] - mono_chunk[right]
        else:
            skipped.append(pair)

    return bipolar, skipped


def reorder_channels(chunk: dict[str, np.ndarray], target_order: list[str]) -> Optional[np.ndarray]:
    if not all(channel in chunk for channel in target_order):
        return None
    return np.vstack([chunk[channel] for channel in target_order]).astype(np.float32)


def resample_chunk(data: np.ndarray, old_fs: int, new_fs: int) -> np.ndarray:
    if old_fs == new_fs:
        return data.astype(np.float32)
    return resample_poly(data, up=new_fs, down=old_fs, axis=1).astype(np.float32)


def design_bandpass(fs: int, low: float = 0.5, high: float = 40.0, order: int = 4):
    nyquist = 0.5 * fs
    return butter(order, [low / nyquist, high / nyquist], btype="band")


def design_notch(fs: int, freq: float = 60.0, q: float = 30.0):
    return iirnotch(freq / (0.5 * fs), q)


class StreamingFilter:
    def __init__(self, fs: int, n_channels: int):
        self.n_channels = n_channels
        self.bp_b, self.bp_a = design_bandpass(fs, low=0.5, high=40.0, order=4)
        self.notch_b, self.notch_a = design_notch(fs, freq=60.0, q=30.0)
        self.notch_state = np.tile(lfilter_zi(self.notch_b, self.notch_a), (n_channels, 1))
        self.bp_state = np.tile(lfilter_zi(self.bp_b, self.bp_a), (n_channels, 1))

    def apply(self, data: np.ndarray) -> np.ndarray:
        notch_out = np.zeros_like(data, dtype=np.float32)
        out = np.zeros_like(data, dtype=np.float32)

        for channel in range(self.n_channels):
            notch_out[channel], self.notch_state[channel] = lfilter(
                self.notch_b,
                self.notch_a,
                data[channel],
                zi=self.notch_state[channel],
            )

        for channel in range(self.n_channels):
            out[channel], self.bp_state[channel] = lfilter(
                self.bp_b,
                self.bp_a,
                notch_out[channel],
                zi=self.bp_state[channel],
            )

        return out.astype(np.float32)


class RollingEEGBuffer:
    def __init__(self, n_channels: int, fs: int, buffer_sec: int):
        self.fs = fs
        self.max_samples = fs * buffer_sec
        self.data = np.zeros((n_channels, 0), dtype=np.float32)

    def append(self, chunk: np.ndarray) -> None:
        self.data = np.concatenate([self.data, chunk], axis=1)
        if self.data.shape[1] > self.max_samples:
            self.data = self.data[:, -self.max_samples :]

    def has_window(self, window_sec: int) -> bool:
        return self.data.shape[1] >= int(window_sec * self.fs)

    def latest_window(self, window_sec: int) -> np.ndarray:
        win_len = int(window_sec * self.fs)
        return self.data[:, -win_len:].copy()


def qc_check(window: np.ndarray) -> dict[str, bool]:
    checks = {
        "has_nan": bool(np.isnan(window).any()),
        "flatline": bool(np.any(np.std(window, axis=1) < 1e-6)),
        "extreme_amplitude": bool(np.any(np.max(np.abs(window), axis=1) > 1e4)),
    }
    checks["pass"] = not any(checks.values())
    return checks


def zscore_per_channel(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True)
    return ((window - mean) / (std + eps)).astype(np.float32)


def bandpower_approx(x: np.ndarray) -> float:
    return float(np.mean(x**2))


def line_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def hjorth_activity(x: np.ndarray) -> float:
    return float(np.var(x))


def hjorth_mobility(x: np.ndarray, eps: float = 1e-8) -> float:
    dx = np.diff(x)
    return float(np.sqrt(np.var(dx) / (np.var(x) + eps)))


def hjorth_complexity(x: np.ndarray, eps: float = 1e-8) -> float:
    dx = np.diff(x)
    ddx = np.diff(dx)
    mobility_x = np.sqrt(np.var(dx) / (np.var(x) + eps))
    mobility_dx = np.sqrt(np.var(ddx) / (np.var(dx) + eps))
    return float(mobility_dx / (mobility_x + eps))


def extract_features(window: np.ndarray) -> np.ndarray:
    features: list[float] = []
    for channel in range(window.shape[0]):
        x = window[channel]
        features.extend(
            [
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
            ]
        )
    return np.asarray(features, dtype=np.float32)


def save_window_example(
    window: np.ndarray,
    session_id: str,
    features: np.ndarray,
    timestamp_end: float,
    fs: int,
    channel_order: list[str],
    qc_pass: bool,
    qc_details: dict[str, bool],
    skipped_pairs: list[str],
    paths: RealtimePaths,
    *,
    notes: str = "",
) -> WindowRecord:
    window_id = str(uuid.uuid4())
    window_path = paths.windows_dir / f"{window_id}_window.npz"
    feature_path = paths.windows_dir / f"{window_id}_features.npy"
    timestamp_start = timestamp_end - WINDOW_SEC

    np.savez_compressed(
        window_path,
        window=window,
        fs=fs,
        channel_order=np.array(channel_order, dtype=object),
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
    )
    np.save(feature_path, features)

    return WindowRecord(
        window_id=window_id,
        session_id=session_id,
        timestamp_end=timestamp_end,
        timestamp_start=timestamp_start,
        file_path=str(window_path),
        feature_path=str(feature_path),
        fs=fs,
        n_channels=window.shape[0],
        n_samples=window.shape[1],
        channel_order=channel_order,
        qc_pass=qc_pass,
        qc_details=qc_details,
        skipped_pairs=skipped_pairs,
        notes=notes,
    )


def append_to_manifest(record: WindowRecord, manifest_path: Union[str, Path]) -> None:
    row = asdict(record)
    row["channel_order"] = json.dumps(row["channel_order"])
    row["qc_details"] = json.dumps(row["qc_details"])
    row["skipped_pairs"] = json.dumps(row["skipped_pairs"])

    df_row = pd.DataFrame([row])
    manifest_path = Path(manifest_path)

    if manifest_path.exists():
        df_row.to_csv(manifest_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(manifest_path, mode="w", header=True, index=False)


class BrainFlowStreamer:
    def __init__(self, board_id: int, params: BrainFlowInputParams):
        self.board = BoardShim(board_id, params)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.eeg_names = BoardShim.get_eeg_names(board_id)

    def start(self) -> None:
        self.board.prepare_session()
        self.board.start_stream()

    def stop(self) -> None:
        try:
            self.board.stop_stream()
        finally:
            self.board.release_session()

    def get_chunk_dict(self, duration_sec: float = 1.0) -> dict[str, np.ndarray]:
        num_points = max(1, int(self.sampling_rate * duration_sec))
        data = self.board.get_current_board_data(num_points)
        if data.shape[1] == 0:
            return {}

        eeg_data = data[self.eeg_channels, :]
        chunk: dict[str, np.ndarray] = {}
        for index, channel_name in enumerate(self.eeg_names):
            if index < eeg_data.shape[0]:
                chunk[channel_name] = eeg_data[index].astype(np.float32)
        return chunk


class LiveEEGDatasetBuilder:
    def __init__(
        self,
        *,
        target_fs: int = TARGET_FS,
        target_channels: list[str] = TARGET_BIPOLAR_CHANNELS,
        session_id: str = "session_01",
        output_root: Optional[Union[str, Path]] = None,
    ):
        self.target_fs = target_fs
        self.target_channels = target_channels
        self.session_id = session_id
        self.filter = StreamingFilter(fs=target_fs, n_channels=len(target_channels))
        self.buffer = RollingEEGBuffer(
            n_channels=len(target_channels),
            fs=target_fs,
            buffer_sec=BUFFER_SEC,
        )
        self.paths = get_realtime_paths(output_root, create=True)
        self.last_save_time = 0.0

    def process_chunk(
        self,
        chunk: dict[str, np.ndarray],
        chunk_fs: int,
        timestamp: float,
    ) -> Optional[WindowRecord]:
        if not chunk:
            return None

        cleaned = clean_channel_dict(chunk)
        bipolar, skipped_pairs = build_bipolar_chunk(cleaned, self.target_channels)
        ordered = reorder_channels(bipolar, self.target_channels)
        if ordered is None:
            return None

        ordered = resample_chunk(ordered, chunk_fs, self.target_fs)
        ordered = self.filter.apply(ordered)
        self.buffer.append(ordered)

        if (timestamp - self.last_save_time) < STRIDE_SEC:
            return None
        if not self.buffer.has_window(WINDOW_SEC):
            return None

        self.last_save_time = timestamp
        window = self.buffer.latest_window(WINDOW_SEC)
        qc = qc_check(window)
        norm_window = zscore_per_channel(window)
        features = extract_features(norm_window)

        record = save_window_example(
            window=norm_window,
            session_id=self.session_id,
            features=features,
            timestamp_end=timestamp,
            fs=self.target_fs,
            channel_order=self.target_channels,
            qc_pass=qc["pass"],
            qc_details=qc,
            skipped_pairs=skipped_pairs,
            paths=self.paths,
            notes="Saved live training window.",
        )
        append_to_manifest(record, self.paths.manifest_path)
        return record


def run_live_dataset_builder(
    board_id: int = BoardIds.SYNTHETIC_BOARD.value,
    params: Optional[BrainFlowInputParams] = None,
    *,
    poll_duration_sec: float = 1.0,
    sleep_sec: float = 0.2,
    session_id: str = "session_01",
    output_root: Optional[Union[str, Path]] = None,
) -> None:
    params = params or BrainFlowInputParams()
    BoardShim.enable_dev_board_logger()

    streamer = BrainFlowStreamer(board_id=board_id, params=params)
    builder = LiveEEGDatasetBuilder(
        session_id=session_id,
        output_root=output_root,
    )

    print(f"Starting BrainFlow stream for session '{session_id}'...")
    print(f"Writing artifacts to: {builder.paths.output_root}")
    print("Live EEG dataset builder running. Press Ctrl+C to stop.")
    streamer.start()

    try:
        while True:
            timestamp = time.time()
            chunk = streamer.get_chunk_dict(duration_sec=poll_duration_sec)
            record = builder.process_chunk(
                chunk=chunk,
                chunk_fs=streamer.sampling_rate,
                timestamp=timestamp,
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
