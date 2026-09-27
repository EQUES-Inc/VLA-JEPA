#!/usr/bin/env python3
#python scripts/MATTERIX/convert_npz_to_lerobot_v21.py   --input-dir /home/ubuntu/dataset/matterix_beaker_dataset_mixed_front/successes   --output-dir /home/ubuntu/dataset/MATTERIX_LEROBOT/matterix_beaker   --fps 20 --with-state

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CHUNK_SIZE = 1000
STATE_DIM = 8


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def compute_stats(array: np.ndarray) -> dict:
    x = np.asarray(array, dtype=np.float32)

    if x.ndim == 1:
        x = x[:, None]

    return {
        "min": np.min(x, axis=0).astype(np.float32).tolist(),
        "max": np.max(x, axis=0).astype(np.float32).tolist(),
        "mean": np.mean(x, axis=0).astype(np.float32).tolist(),
        "std": np.std(x, axis=0).astype(np.float32).tolist(),
        "count": [int(x.shape[0])],
    }


def encode_video_ffmpeg(
    frames: np.ndarray,
    output_path: Path,
    fps: float,
) -> None:
    frames = np.asarray(frames)

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected [T,H,W,3], got {frames.shape}"
        )

    if frames.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 frames, got {frames.dtype}"
        )

    _, h, w, _ = frames.shape

    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(
            f"Video dimensions must be even, got {h}x{w}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = np.ascontiguousarray(frames)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        input=frames.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode(
                "utf-8",
                errors="replace",
            )
        )


def write_episode_parquet(
    output_path: Path,
    actions: np.ndarray,
    states: np.ndarray,
    episode_index: int,
    global_start_index: int,
    task_index: int,
    fps: float,
) -> np.ndarray:
    actions = np.asarray(
        actions,
        dtype=np.float32,
    )

    t = actions.shape[0]

    states = np.asarray(
        states,
        dtype=np.float32,
    )

    if states.shape != (t, STATE_DIM):
        raise ValueError(
            "states must have shape "
            f"({t}, {STATE_DIM}), got {states.shape}"
        )

    if not np.all(np.isfinite(states)):
        raise ValueError(
            "states contain NaN/Inf"
        )

    action_values = pa.array(
        actions.reshape(-1),
        type=pa.float32(),
    )

    action_column = pa.FixedSizeListArray.from_arrays(
        action_values,
        list_size=7,
    )

    state_values = pa.array(
        states.reshape(-1),
        type=pa.float32(),
    )

    state_column = pa.FixedSizeListArray.from_arrays(
        state_values,
        list_size=STATE_DIM,
    )

    timestamp = (
        np.arange(t, dtype=np.float32)
        / np.float32(fps)
    )

    table = pa.table(
        {
            "observation.state": state_column,

            "action": action_column,

            "timestamp": pa.array(
                timestamp,
                type=pa.float32(),
            ),

            "frame_index": pa.array(
                np.arange(t, dtype=np.int64),
                type=pa.int64(),
            ),

            "episode_index": pa.array(
                np.full(
                    t,
                    episode_index,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            ),

            "index": pa.array(
                np.arange(
                    global_start_index,
                    global_start_index + t,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            ),

            "task_index": pa.array(
                np.full(
                    t,
                    task_index,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            ),
        }
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pq.write_table(
        table,
        output_path,
        compression="zstd",
    )

    return states


def validate_episode(
    path: Path,
    primary: np.ndarray,
    wrist: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray,
) -> None:
    if primary.ndim != 4 or primary.shape[-1] != 3:
        raise ValueError(
            f"{path}: invalid primary_images shape "
            f"{primary.shape}"
        )

    if wrist.ndim != 4 or wrist.shape[-1] != 3:
        raise ValueError(
            f"{path}: invalid wrist_images shape "
            f"{wrist.shape}"
        )

    if primary.dtype != np.uint8:
        raise ValueError(
            f"{path}: primary dtype must be uint8"
        )

    if wrist.dtype != np.uint8:
        raise ValueError(
            f"{path}: wrist dtype must be uint8"
        )

    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            f"{path}: actions must be [T,7], "
            f"got {actions.shape}"
        )

    t = actions.shape[0]

    if primary.shape[0] != t:
        raise ValueError(
            f"{path}: primary/action length mismatch"
        )

    if wrist.shape[0] != t:
        raise ValueError(
            f"{path}: wrist/action length mismatch"
        )

    if t < 8:
        raise ValueError(
            f"{path}: episode has only {t} frames; "
            "VLA-JEPA video horizon is 8"
        )

    if not np.all(np.isfinite(actions)):
        raise ValueError(
            f"{path}: actions contain NaN/Inf"
        )

    if states.ndim != 2 or states.shape != (t, STATE_DIM):
        raise ValueError(
            f"{path}: states must be [T,{STATE_DIM}], "
            f"got {states.shape}"
        )

    if not np.all(np.isfinite(states)):
        raise ValueError(
            f"{path}: states contain NaN/Inf"
        )


def convert(
    input_dir: Path,
    output_dir: Path,
    fps: float,
    overwrite: bool,
    with_state: bool,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be > 0")

    print(
        "[CONVERTER] observation.state source:",
        (
            "real 8D collector states"
            if with_state
            else "dummy zero 8D state"
        ),
    )
    if with_state:
        print(
            "[CONVERTER] state mapping: "
            "[x,y,z,roll,pitch,yaw,pad,gripper] = "
            "[EE xyz, EE axis-angle xyz, gripper qpos 1/2]"
        )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not available in PATH"
        )

    npz_files = sorted(
        input_dir.glob("episode_*.npz")
    )

    if not npz_files:
        npz_files = sorted(
            input_dir.glob("*.npz")
        )

    if not npz_files:
        raise FileNotFoundError(
            f"No npz files found in {input_dir}"
        )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. "
                "Use --overwrite to replace it."
            )

        shutil.rmtree(output_dir)

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    task_to_index: dict[str, int] = {}

    global_index = 0
    total_frames = 0
    total_videos = 0

    all_actions = []
    all_states = []

    primary_shape = None
    wrist_shape = None

    for episode_index, npz_path in enumerate(npz_files):
        print(
            f"[{episode_index + 1}/{len(npz_files)}] "
            f"{npz_path.name}"
        )

        with np.load(
            npz_path,
            allow_pickle=False,
        ) as data:
            required = {
                "primary_images",
                "wrist_images",
                "actions",
                "language",
            }

            missing = required - set(data.files)

            if missing:
                raise KeyError(
                    f"{npz_path}: missing keys "
                    f"{sorted(missing)}"
                )

            primary = np.asarray(
                data["primary_images"]
            )

            wrist = np.asarray(
                data["wrist_images"]
            )

            actions = np.asarray(
                data["actions"],
                dtype=np.float32,
            )

            language_array = np.asarray(
                data["language"]
            )

            if with_state:
                if "states" not in data.files:
                    raise KeyError(
                        f"{npz_path}: --with-state was requested, "
                        "but this episode has no 'states' array. "
                        "Re-collect the episode with the collector's "
                        "--with-state option."
                    )

                states = np.asarray(
                    data["states"],
                    dtype=np.float32,
                )
            else:
                states = np.zeros(
                    (actions.shape[0], STATE_DIM),
                    dtype=np.float32,
                )

            if language_array.size != 1:
                raise ValueError(
                    f"{npz_path}: language must be scalar"
                )

            language = str(
                language_array.item()
            ).strip()

        validate_episode(
            npz_path,
            primary,
            wrist,
            actions,
            states,
        )

        t = actions.shape[0]

        if primary_shape is None:
            primary_shape = list(
                primary.shape[1:]
            )

        if wrist_shape is None:
            wrist_shape = list(
                wrist.shape[1:]
            )

        if list(primary.shape[1:]) != primary_shape:
            raise ValueError(
                f"{npz_path}: inconsistent primary shape"
            )

        if list(wrist.shape[1:]) != wrist_shape:
            raise ValueError(
                f"{npz_path}: inconsistent wrist shape"
            )

        if language not in task_to_index:
            task_to_index[language] = len(
                task_to_index
            )

        task_index = task_to_index[language]

        chunk_index = (
            episode_index // CHUNK_SIZE
        )

        parquet_path = (
            output_dir
            / "data"
            / f"chunk-{chunk_index:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )

        write_episode_parquet(
            output_path=parquet_path,
            actions=actions,
            states=states,
            episode_index=episode_index,
            global_start_index=global_index,
            task_index=task_index,
            fps=fps,
        )

        primary_video_path = (
            output_dir
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / "observation.images.image"
            / f"episode_{episode_index:06d}.mp4"
        )

        wrist_video_path = (
            output_dir
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / "observation.images.wrist_image"
            / f"episode_{episode_index:06d}.mp4"
        )

        # No additional image flip here.
        encode_video_ffmpeg(
            primary,
            primary_video_path,
            fps,
        )

        encode_video_ffmpeg(
            wrist,
            wrist_video_path,
            fps,
        )

        append_jsonl(
            meta_dir / "episodes.jsonl",
            {
                "episode_index": episode_index,
                "tasks": [language],
                "length": t,
            },
        )

        append_jsonl(
            meta_dir / "episodes_stats.jsonl",
            {
                "episode_index": episode_index,
                "stats": {
                    "observation.state": compute_stats(states),
                    "action": compute_stats(actions),
                },
            },
        )

        all_states.append(states)
        all_actions.append(actions)

        global_index += t
        total_frames += t
        total_videos += 2

    for language, task_index in sorted(
        task_to_index.items(),
        key=lambda x: x[1],
    ):
        append_jsonl(
            meta_dir / "tasks.jsonl",
            {
                "task_index": task_index,
                "task": language,
            },
        )

    all_actions = np.concatenate(
        all_actions,
        axis=0,
    ).astype(np.float32)

    all_states = np.concatenate(
        all_states,
        axis=0,
    ).astype(np.float32)

    write_json(
        meta_dir / "stats.json",
        {
            "observation.state": compute_stats(all_states),
            "action": compute_stats(all_actions),
        },
    )

    expected_state_dim = 8
    if all_states.ndim != 2 or all_states.shape[1] != expected_state_dim:
        raise RuntimeError(
            "Internal state aggregation error: expected "
            f"[N,{expected_state_dim}], got {all_states.shape}"
        )

    if with_state and np.allclose(all_states, 0.0):
        raise RuntimeError(
            "--with-state was requested, but all converted states are zero. "
            "Check that the collector was run with --with-state and that "
            "the NPZ files contain real 'states' arrays."
        )

    modality = {
        "state": {
            # Names intentionally follow VLA-JEPA's existing state_keys
            # convention. Numerical semantics are:
            #   roll/pitch/yaw -> EE axis-angle x/y/z
            #   pad/gripper    -> gripper qpos 1/2
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5},
            "yaw": {"start": 5, "end": 6},
            "pad": {"start": 6, "end": 7},
            "gripper": {"start": 7, "end": 8},
        },

        "action": {
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5},
            "yaw": {"start": 5, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },

        "video": {
            "primary_image": {
                "original_key":
                    "observation.images.image"
            },

            "wrist_image": {
                "original_key":
                    "observation.images.wrist_image"
            },
        },

        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index"
            }
        },
    }

    write_json(
        meta_dir / "modality.json",
        modality,
    )

    total_chunks = math.ceil(
        len(npz_files) / CHUNK_SIZE
    )

    info = {
        "codebase_version": "v2.1",

        "robot_type": "matterix_franka",
        "with_state": bool(with_state),
        "state_source": (
            "collector_npz.states"
            if with_state
            else "dummy_zeros"
        ),
        "state_numeric_semantics": [
            "ee_world_x",
            "ee_world_y",
            "ee_world_z",
            "ee_axis_angle_x",
            "ee_axis_angle_y",
            "ee_axis_angle_z",
            "gripper_qpos_1",
            "gripper_qpos_2",
        ],

        "total_episodes": len(npz_files),
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_videos": total_videos,

        "total_chunks": total_chunks,
        "chunks_size": CHUNK_SIZE,

        "fps": float(fps),

        "splits": {
            "train": f"0:{len(npz_files)}"
        },

        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),

        "video_path": (
            "videos/chunk-{episode_chunk:03d}/"
            "{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),

        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": [
                    "x",
                    "y",
                    "z",
                    "roll",
                    "pitch",
                    "yaw",
                    "pad",
                    "gripper",
                ],
            },

            "observation.images.image": {
                "dtype": "video",
                "shape": primary_shape,
                "names": [
                    "height",
                    "width",
                    "channel",
                ],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },

            "observation.images.wrist_image": {
                "dtype": "video",
                "shape": wrist_shape,
                "names": [
                    "height",
                    "width",
                    "channel",
                ],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },

            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": [
                    "x",
                    "y",
                    "z",
                    "roll",
                    "pitch",
                    "yaw",
                    "gripper",
                ],
            },

            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },

            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },

            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
    }

    write_json(
        meta_dir / "info.json",
        info,
    )

    print()
    print("Conversion complete")
    print("output:", output_dir)
    print("episodes:", len(npz_files))
    print("frames:", total_frames)
    print("tasks:", task_to_index)
    print(
        "action min:",
        np.round(all_actions.min(axis=0), 6),
    )
    print(
        "action max:",
        np.round(all_actions.max(axis=0), 6),
    )
    print(
        "state mean:",
        np.round(all_states.mean(axis=0), 6),
    )
    print(
        "state std:",
        np.round(all_states.std(axis=0), 6),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fps",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--with-state",
        action="store_true",
        help=(
            "Use each episode NPZ's real 8D 'states' array as "
            "LeRobot observation.state. Without this flag, preserve "
            "the previous behavior and write an all-zero 8D dummy state."
        ),
    )

    args = parser.parse_args()

    convert(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        fps=args.fps,
        overwrite=args.overwrite,
        with_state=args.with_state,
    )


if __name__ == "__main__":
    main()