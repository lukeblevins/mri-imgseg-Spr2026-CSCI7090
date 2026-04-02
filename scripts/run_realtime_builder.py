from __future__ import annotations

import argparse

from brainflow.board_shim import BoardIds, BrainFlowInputParams

from eeg_project.realtime import run_live_dataset_builder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build realtime EEG windows from a BrainFlow stream.")
    parser.add_argument(
        "--board-id",
        type=int,
        default=BoardIds.SYNTHETIC_BOARD.value,
        help="BrainFlow board id. Defaults to the synthetic board.",
    )
    parser.add_argument("--session-id", default="session_01", help="Session identifier written to the manifest.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override EEG_OUTPUT_ROOT. Defaults to artifacts/realtime/.",
    )
    parser.add_argument("--poll-duration-sec", type=float, default=1.0, help="Seconds of board data to sample per poll.")
    parser.add_argument("--sleep-sec", type=float, default=0.2, help="Seconds to sleep between polls.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_live_dataset_builder(
        board_id=args.board_id,
        params=BrainFlowInputParams(),
        poll_duration_sec=args.poll_duration_sec,
        sleep_sec=args.sleep_sec,
        session_id=args.session_id,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
