from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union

CHB_RELATIVE_PATH = Path("chbmit/chb01/chb01_01.edf")
SIENA_RELATIVE_PATH = Path("siena/pn00/PN00-1.edf")
DEFAULT_REALTIME_OUTPUT_ROOT = Path("artifacts/realtime")


def has_expected_dataset_files(root: Path) -> bool:
    return (root / CHB_RELATIVE_PATH).exists() and (root / SIENA_RELATIVE_PATH).exists()


def _infer_root_from_match(file_path: Path, marker_dir: str) -> Optional[Path]:
    parts = file_path.parts
    if marker_dir not in parts:
        return None

    marker_index = parts.index(marker_dir)
    if marker_index == 0:
        return Path(file_path.anchor)

    return Path(*parts[:marker_index])


def _candidate_dataset_roots(cwd: Optional[Path] = None) -> list[Path]:
    cwd = cwd or Path.cwd()
    candidates: list[Path] = []

    env_override = os.environ.get("EEG_DATA_ROOT")
    if env_override:
        candidates.append(Path(env_override).expanduser())

    for candidate in [cwd, *cwd.parents]:
        candidates.append(candidate / "data")

    candidates.extend(
        [
            Path("/content/data"),
            Path("/content/drive/MyDrive/data"),
            Path("/content/drive/MyDrive/Colab Notebooks/data"),
        ]
    )
    return candidates


def _recursive_drive_candidates() -> Iterable[Path]:
    search_roots = [
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Shareddrives"),
    ]

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for chb_match in search_root.rglob(CHB_RELATIVE_PATH.name):
            inferred_root = _infer_root_from_match(chb_match, "chbmit")
            if inferred_root is not None:
                yield inferred_root

        for siena_match in search_root.rglob(SIENA_RELATIVE_PATH.name):
            inferred_root = _infer_root_from_match(siena_match, "siena")
            if inferred_root is not None:
                yield inferred_root


def _running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_dataset_root(
    dataset_root: Optional[Union[str, Path]] = None,
    *,
    mount_google_drive_if_needed: bool = False,
) -> tuple[Path, list[Path]]:
    seen: set[str] = set()
    checked: list[Path] = []

    def register(candidate: Path) -> Optional[Path]:
        expanded = candidate.expanduser()
        key = str(expanded)
        if key in seen:
            return None
        seen.add(key)
        checked.append(expanded)
        return expanded

    if dataset_root is not None:
        candidate = register(Path(dataset_root))
        if candidate is not None and has_expected_dataset_files(candidate):
            return candidate.resolve(), checked

    for candidate in _candidate_dataset_roots():
        expanded = register(candidate)
        if expanded is not None and has_expected_dataset_files(expanded):
            return expanded.resolve(), checked

    if mount_google_drive_if_needed and _running_in_colab():
        from google.colab import drive

        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")

    for candidate in _recursive_drive_candidates():
        expanded = register(candidate)
        if expanded is not None and has_expected_dataset_files(expanded):
            return expanded.resolve(), checked

    checked_display = "\n".join(f" - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not locate the EEG dataset files.\n"
        f"Working directory: {Path.cwd()}\n"
        f"Checked locations:\n{checked_display}\n\n"
        "Set EEG_DATA_ROOT to the folder that directly contains chbmit/ and siena/ "
        "or place those dataset directories under ./data."
    )


def get_realtime_output_root(
    output_root: Optional[Union[str, Path]] = None,
    *,
    create: bool = False,
) -> Path:
    raw_root = output_root or os.environ.get("EEG_OUTPUT_ROOT") or DEFAULT_REALTIME_OUTPUT_ROOT
    path = Path(raw_root).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path
