#!/usr/bin/env python3
"""
Collect paired beaker / blue-cylinder scripted demonstrations from MATTERIX.

This client expects the current matterix_server.py protocol:

    reset request:
        {"cmd": "reset", "episode_idx": int}

    step request:
        {"cmd": "step", "action": float32[7]}

    observation response keys:
        primary_image: uint8[H,W,3]
        wrist_image: uint8[H,W,3]
        state: array whose first 3 values are EE world xyz
        beaker_position: float[3]
        beaker_lift: float
        blue_cylinder_position: float[3]
        blue_cylinder_lift: float
        success: bool                  # legacy beaker success
        blue_cylinder_success: bool    # required for cylinder episodes
        done: bool

The scripted teacher uses privileged EE/target-object positions only to compute labels.
Each successful scene seed is collected twice: once for the beaker and once for
the blue cylinder. The saved language instruction and teacher target are always
kept consistent. Policy inputs remain RGB (+ optional proprioceptive state) +
language; the saved target is the same raw 7D action sent to the server.

With --with-state, each pre-action observation additionally saves the 8D
VLA-JEPA robot state returned by MATTERIX:
    [EE xyz, EE axis-angle xyz, gripper qpos 1, gripper qpos 2]
Without --with-state, the saved dataset format remains compatible with the
previous collector and no "states" array is written to the episode NPZ.

Important:
- Set --position-axis-sign and --position-scale to exactly the values used by
  matterix_server.py.
- The default controller keeps rotation_delta at zero, so the reset EE
  orientation must already be suitable for top-down grasping.
- Tune --grasp-z-offset for your specific beaker asset and Franka control frame.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# TCP protocol
# ---------------------------------------------------------------------------

def send_message(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("MATTERIX server closed the connection.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Any:
    (length,) = struct.unpack("!Q", recv_exact(sock, 8))
    return pickle.loads(recv_exact(sock, length))


def check_server_error(response: Any) -> None:
    if isinstance(response, dict) and response.get("server_error", False):
        raise RuntimeError(
            f"MATTERIX server error: "
            f"{response.get('error_type', 'UnknownError')}: "
            f"{response.get('error_message', 'unknown error')}"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CollectorConfig:
    beaker_instruction: str
    cylinder_instruction: str
    position_axis_sign: tuple[float, float, float]
    position_scale: float
    max_world_step: float
    approach_z_offset: float
    beaker_grasp_z_offset: float
    cylinder_grasp_z_offset: float
    lift_z_offset: float
    approach_xy_tolerance: float
    approach_z_tolerance: float
    grasp_xy_tolerance: float
    grasp_z_tolerance: float
    close_steps: int
    max_steps: int
    flip_primary: bool
    flip_wrist: bool
    with_state: bool


def parse_axis_sign(text: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3 or any(abs(abs(x) - 1.0) > 1e-6 for x in parts):
        raise argparse.ArgumentTypeError(
            "Expected three comma-separated signs, e.g. 1,1,1 or -1,-1,1."
        )
    return (parts[0], parts[1], parts[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect paired beaker / blue-cylinder demonstrations."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--num-successes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("./matterix_beaker_cylinder_dataset"))
    parser.add_argument("--beaker-instruction", default="Pick up the beaker")
    parser.add_argument("--cylinder-instruction", default="Pick up the blue cylinder")
    parser.add_argument(
        "--first-target",
        choices=("beaker", "blue_cylinder"),
        default="beaker",
        help=(
            "Target for the first successful episode. Successes alternate targets, "
            "and every pair reuses the same reset seed."
        ),
    )

    # These must match matterix_server.py.
    parser.add_argument(
        "--position-axis-sign",
        type=parse_axis_sign,
        default=(1.0, 1.0, 1.0),
        help="Server xyz sign mapping, comma-separated. Default: 1,1,1",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=0.05,
        help="Server position_scale. Default: 0.05",
    )

    # Teacher trajectory in world coordinates.
    parser.add_argument("--max-world-step", type=float, default=0.006)
    parser.add_argument("--approach-z-offset", type=float, default=0.12)
    parser.add_argument(
        "--beaker-grasp-z-offset",
        type=float,
        default=0.040,
        help="EE control-frame height above the beaker origin during grasp.",
    )
    parser.add_argument(
        "--cylinder-grasp-z-offset",
        type=float,
        default=0.040,
        help="EE control-frame height above the blue-cylinder origin during grasp.",
    )
    parser.add_argument("--lift-z-offset", type=float, default=0.18)

    parser.add_argument("--approach-xy-tolerance", type=float, default=0.008)
    parser.add_argument("--approach-z-tolerance", type=float, default=0.008)
    parser.add_argument("--grasp-xy-tolerance", type=float, default=0.006)
    parser.add_argument("--grasp-z-tolerance", type=float, default=0.004)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=250)

    parser.add_argument(
        "--flip-primary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rotate primary image by 180 degrees before saving.",
    )
    parser.add_argument(
        "--flip-wrist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rotate wrist image by 180 degrees before saving.",
    )
    parser.add_argument(
        "--keep-failures",
        action="store_true",
        help="Also save failed attempts under output_dir/failures.",
    )
    parser.add_argument(
        "--with-state",
        action="store_true",
        help=(
            "Save the 8D VLA-JEPA robot state for every observation: "
            "[EE x, y, z, axis-angle x, y, z, gripper qpos 1, gripper qpos 2]. "
            "Without this flag, collection behavior matches the previous version "
            "and state is used only internally by the scripted teacher."
        ),
    )
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument(
        "--save-debug-video",
        action="store_true",
        help="Save per-attempt front/wrist debug MP4 videos.",
    )
    parser.add_argument(
        "--debug-video-fps",
        type=float,
        default=20.0,
        help=(
            "FPS for per-attempt debug MP4 videos when "
            "--save-debug-video is enabled. Default: 20"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Observation and action helpers
# ---------------------------------------------------------------------------

def as_rgb(image: Any, *, flip_180: bool) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}.")
    arr = np.ascontiguousarray(arr[..., :3].astype(np.uint8, copy=False))
    if flip_180:
        arr = np.ascontiguousarray(arr[::-1, ::-1])
    return arr


def extract_vla_state(obs: dict[str, Any]) -> np.ndarray:
    """
    Extract and validate the 8D robot state expected by VLA-JEPA:

        [EE x, EE y, EE z,
         EE axis-angle x, EE axis-angle y, EE axis-angle z,
         gripper qpos 1, gripper qpos 2]

    The MATTERIX server is expected to construct obs["state"] in this order.
    """
    if obs.get("state") is None:
        raise KeyError(
            "Observation has no state. The MATTERIX server must return an "
            "8D robot state when --with-state is used."
        )

    state = np.asarray(
        obs["state"],
        dtype=np.float32,
    ).reshape(-1)

    if state.shape != (8,):
        raise ValueError(
            "Expected MATTERIX/VLA-JEPA state shape (8,), "
            f"got {state.shape}. Values: {state}"
        )

    if not np.all(np.isfinite(state)):
        raise ValueError(
            f"State contains NaN/Inf values: {state}"
        )

    return state.copy()


def extract_geometry(
    obs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return EE, beaker, and blue-cylinder world positions."""
    if obs.get("state") is None:
        raise KeyError(
            "Observation has no state. The server must include robot state; "
            "the first three values must be EE world xyz."
        )

    for key in ("beaker_position", "blue_cylinder_position"):
        if key not in obs:
            raise KeyError(
                f"Observation has no {key}. Add it to matterix_server.make_response()."
            )

    state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    if state.size < 3:
        raise ValueError(f"state must contain EE xyz, got shape {state.shape}.")

    ee_pos = state[:3].copy()
    beaker_pos = np.asarray(obs["beaker_position"], dtype=np.float32).reshape(3).copy()
    cylinder_pos = np.asarray(
        obs["blue_cylinder_position"], dtype=np.float32
    ).reshape(3).copy()
    return ee_pos, beaker_pos, cylinder_pos


def select_target(
    target_object: str,
    beaker_pos: np.ndarray,
    cylinder_pos: np.ndarray,
) -> np.ndarray:
    if target_object == "beaker":
        return beaker_pos
    if target_object == "blue_cylinder":
        return cylinder_pos
    raise ValueError(f"Unknown target_object={target_object!r}")


def instruction_for_target(cfg: CollectorConfig, target_object: str) -> str:
    if target_object == "beaker":
        return cfg.beaker_instruction
    if target_object == "blue_cylinder":
        return cfg.cylinder_instruction
    raise ValueError(f"Unknown target_object={target_object!r}")


def target_success(obs: dict[str, Any], target_object: str) -> bool:
    """Use target-specific success. Never accept beaker success for cylinder data."""
    if target_object == "beaker":
        return bool(obs.get("beaker_success", obs.get("success", False)))
    if target_object == "blue_cylinder":
        if "blue_cylinder_success" not in obs:
            raise KeyError(
                "Cylinder episode requires obs['blue_cylinder_success']. "
                "Add a target-specific cylinder success signal to matterix_server.py."
            )
        return bool(obs["blue_cylinder_success"])
    raise ValueError(f"Unknown target_object={target_object!r}")


def target_lift(obs: dict[str, Any], target_object: str) -> float:
    if target_object == "beaker":
        return float(obs.get("beaker_lift", 0.0))
    if target_object == "blue_cylinder":
        if "blue_cylinder_lift" not in obs:
            raise KeyError(
                "Cylinder episode requires obs['blue_cylinder_lift']. "
                "Add it to matterix_server.py."
            )
        return float(obs["blue_cylinder_lift"])
    raise ValueError(f"Unknown target_object={target_object!r}")


def clipped_world_delta(
    current: np.ndarray,
    target: np.ndarray,
    max_step: float,
) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float32) - np.asarray(
        current, dtype=np.float32
    )
    norm = float(np.linalg.norm(delta))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    if norm > max_step:
        delta = delta * (max_step / norm)
    return delta.astype(np.float32)


def world_delta_to_raw_action(
    desired_world_delta: np.ndarray,
    *,
    position_axis_sign: tuple[float, float, float],
    position_scale: float,
    gripper: float,
) -> np.ndarray:
    """
    Invert the server mapping:

        world_delta = raw_delta * position_axis_sign * position_scale

    Since each sign is +/-1, its inverse is itself.
    """
    if position_scale <= 0:
        raise ValueError("position_scale must be positive.")

    signs = np.asarray(position_axis_sign, dtype=np.float32)
    raw_delta = desired_world_delta * signs / np.float32(position_scale)

    action = np.zeros(7, dtype=np.float32)
    action[:3] = raw_delta
    action[3:6] = 0.0
    action[6] = np.float32(gripper)
    return action


def validate_observation(obs: Any) -> dict[str, Any]:
    check_server_error(obs)
    if not isinstance(obs, dict):
        raise TypeError(f"Expected dict observation, got {type(obs)}.")
    required = {
        "primary_image",
        "wrist_image",
        "state",
        "beaker_position",
        "blue_cylinder_position",
        "success",
        "done",
    }
    missing = required - set(obs)
    if missing:
        raise KeyError(f"Observation missing keys: {sorted(missing)}")
    return obs


# ---------------------------------------------------------------------------
# Scripted teacher
# ---------------------------------------------------------------------------

class ObjectTeacher:
    def __init__(self, cfg: CollectorConfig) -> None:
        self.cfg = cfg
        self.phase = "approach"
        self.close_counter = 0
        self.initial_target_pos: np.ndarray | None = None

    def reset(self, target_pos: np.ndarray) -> None:
        self.phase = "approach"
        self.close_counter = 0
        self.initial_target_pos = target_pos.copy()

    def action(
        self,
        ee_pos: np.ndarray,
        target_pos: np.ndarray,
        *,
        grasp_z_offset: float,
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        cfg = self.cfg
        xy_error = float(np.linalg.norm(ee_pos[:2] - target_pos[:2]))

        if self.phase == "approach":
            target = target_pos.copy()

            # target[0] -= 0.075
            # target[1] -= 0.025
            target[2] = (
                target_pos[2]
                + cfg.approach_z_offset
            )

            xy_error = float(
                np.linalg.norm(
                    ee_pos[:2] - target[:2]
                )
            )

            z_error = abs(
                float(
                    ee_pos[2] - target[2]
                )
            )

            if (
                xy_error
                <= cfg.approach_xy_tolerance
                and z_error
                <= cfg.approach_z_tolerance
            ):
                self.phase = "descend"

        if self.phase == "descend":
            target = target_pos.copy()

            # target[0] -= 0.075
            # target[1] -= 0.025
            target[2] = (
                target_pos[2]
                + grasp_z_offset
            )

            xy_error = float(
                np.linalg.norm(
                    ee_pos[:2] - target[:2]
                )
            )

            z_error = abs(
                float(
                    ee_pos[2] - target[2]
                )
            )

            print(
                "[DESCEND DEBUG]",
                f"ee_z={ee_pos[2]:.5f}",
                f"target_z={target[2]:.5f}",
                f"z_error={z_error:.5f}",
                f"xy_error={xy_error:.5f}",
                flush=True,
            )

            if (
                xy_error
                <= cfg.grasp_xy_tolerance
                and z_error
                <= cfg.grasp_z_tolerance
            ):
                self.phase = "close"
                self.close_counter = 0

        if self.phase == "close":
            target = ee_pos.copy()
            self.close_counter += 1
            if self.close_counter >= cfg.close_steps:
                self.phase = "lift"

        if self.phase == "lift":
            target = target_pos.copy()
            assert self.initial_target_pos is not None
            target[2] = self.initial_target_pos[2] + cfg.lift_z_offset

        gripper = -1.0 if self.phase in {"close", "lift"} else 1.0
        desired_world_delta = clipped_world_delta(
            current=ee_pos,
            target=target,
            max_step=cfg.max_world_step,
        )

        action = world_delta_to_raw_action(
            desired_world_delta,
            position_axis_sign=cfg.position_axis_sign,
            position_scale=cfg.position_scale,
            gripper=gripper,
        )

        debug = {
            "phase": self.phase,
            "xy_error": xy_error,
            "z_error": abs(float(ee_pos[2] - target[2])),
            "desired_dx": float(desired_world_delta[0]),
            "desired_dy": float(desired_world_delta[1]),
            "desired_dz": float(desired_world_delta[2]),
        }
        return action, debug


# ---------------------------------------------------------------------------
# Episode storage
# ---------------------------------------------------------------------------

def save_episode(
    directory: Path,
    *,
    episode_id: int,
    attempt_id: int,
    success: bool,
    cfg: CollectorConfig,
    target_object: str,
    instruction: str,
    frames: list[dict[str, Any]],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"episode_{episode_id:05d}.npz"
    tmp_path = path.with_suffix(".npz.tmp")

    phases = np.asarray([frame["phase"] for frame in frames], dtype="U16")
    primary = np.stack([frame["primary_image"] for frame in frames])
    wrist = np.stack([frame["wrist_image"] for frame in frames])
    actions = np.stack([frame["action"] for frame in frames]).astype(np.float32)

    states = None
    if cfg.with_state:
        states = np.stack(
            [frame["state"] for frame in frames]
        ).astype(np.float32)

        if states.ndim != 2 or states.shape[1] != 8:
            raise ValueError(
                "Expected saved states with shape (T, 8), "
                f"got {states.shape}"
            )

    ee_positions = np.stack([frame["ee_position"] for frame in frames]).astype(np.float32)
    beaker_positions = np.stack(
        [frame["beaker_position"] for frame in frames]
    ).astype(np.float32)
    cylinder_positions = np.stack(
        [frame["blue_cylinder_position"] for frame in frames]
    ).astype(np.float32)
    target_positions = np.stack(
        [frame["target_position"] for frame in frames]
    ).astype(np.float32)
    target_lifts = np.asarray(
        [frame["target_lift"] for frame in frames], dtype=np.float32
    )

    metadata = {
        "episode_id": episode_id,
        "attempt_id": attempt_id,
        "success": success,
        "target_object": target_object,
        "instruction": instruction,
        "num_steps": len(frames),
        "collector_config": asdict(cfg),
        "with_state": bool(cfg.with_state),
        "state_semantics": (
            [
                "ee_world_x",
                "ee_world_y",
                "ee_world_z",
                "ee_axis_angle_x",
                "ee_axis_angle_y",
                "ee_axis_angle_z",
                "gripper_qpos_1",
                "gripper_qpos_2",
            ]
            if cfg.with_state
            else None
        ),
        "action_semantics": [
            "raw_dx",
            "raw_dy",
            "raw_dz",
            "axis_angle_dx",
            "axis_angle_dy",
            "axis_angle_dz",
            "gripper_open_plus1_close_minus1",
        ],
        "observation_action_alignment": (
            "Each image/state at index t is the observation before action[t]."
        ),
    }

    save_payload: dict[str, Any] = {
        "primary_images": primary,
        "wrist_images": wrist,
        "actions": actions,
        "ee_positions": ee_positions,
        "beaker_positions": beaker_positions,
        "blue_cylinder_positions": cylinder_positions,
        "target_positions": target_positions,
        "target_lifts": target_lifts,
        "phases": phases,
        "target_object": np.asarray(target_object),
        "language": np.asarray(instruction),
        "metadata_json": np.asarray(
            json.dumps(metadata, ensure_ascii=False)
        ),
    }

    if cfg.with_state:
        assert states is not None
        save_payload["states"] = states

    with tmp_path.open("wb") as file:
        np.savez_compressed(
            file,
            **save_payload,
        )
        file.flush()
        os.fsync(file.fileno())

    tmp_path.replace(path)
    return path


def save_debug_videos(
    debug_dir: Path,
    *,
    attempt_id: int,
    frames: list[dict[str, Any]],
    fps: float,
    final_primary: np.ndarray | None = None,
    final_wrist: np.ndarray | None = None,
) -> tuple[Path | None, Path | None]:
    """Save full per-step front/wrist debug videos for one attempt.

    Uses OpenCV VideoWriter instead of imageio/ffmpeg. Front and wrist
    videos are written independently so a failure in one camera does not
    prevent the other from being saved.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for debug video saving. "
            "Install it with: pip install opencv-python"
        ) from exc

    if not frames:
        print("[DEBUG VIDEO] no frames, skip", flush=True)
        return None, None

    if fps <= 0:
        raise ValueError("debug video fps must be positive.")

    debug_dir.mkdir(parents=True, exist_ok=True)

    front_frames = [
        np.asarray(frame["primary_image"], dtype=np.uint8)
        for frame in frames
    ]
    wrist_frames = [
        np.asarray(frame["wrist_image"], dtype=np.uint8)
        for frame in frames
    ]

    # Include the terminal post-step observation when available so the
    # video visibly reaches the success/failure end state.
    if final_primary is not None:
        front_frames.append(np.asarray(final_primary, dtype=np.uint8))
    if final_wrist is not None:
        wrist_frames.append(np.asarray(final_wrist, dtype=np.uint8))

    front_path = debug_dir / f"attempt_{attempt_id:04d}_front.mp4"
    wrist_path = debug_dir / f"attempt_{attempt_id:04d}_wrist.mp4"

    def write_mp4(
        path: Path,
        video_frames: list[np.ndarray],
    ) -> Path | None:
        if not video_frames:
            print(f"[DEBUG VIDEO] no frames for {path}", flush=True)
            return None

        first = np.asarray(video_frames[0])

        print(
            "[DEBUG VIDEO]",
            f"opening={path}",
            f"frames={len(video_frames)}",
            f"first_shape={first.shape}",
            f"dtype={first.dtype}",
            flush=True,
        )

        if first.ndim != 3 or first.shape[2] != 3:
            raise ValueError(
                f"Invalid first frame for {path.name}: {first.shape}"
            )

        height, width = first.shape[:2]

        # mp4v is usually available in standard OpenCV installations and
        # avoids the libx264/imageio path that caused the previous failure.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            float(fps),
            (width, height),
        )

        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"cv2.VideoWriter failed to open: {path}"
            )

        written = 0

        try:
            for i, frame in enumerate(video_frames):
                arr = np.asarray(frame)

                if arr.ndim != 3 or arr.shape[2] != 3:
                    print(
                        "[DEBUG VIDEO] "
                        f"skip invalid frame path={path.name} "
                        f"idx={i} shape={arr.shape}",
                        flush=True,
                    )
                    continue

                if arr.shape[:2] != (height, width):
                    print(
                        "[DEBUG VIDEO] "
                        f"skip size mismatch path={path.name} "
                        f"idx={i} shape={arr.shape} "
                        f"expected=({height}, {width}, 3)",
                        flush=True,
                    )
                    continue

                arr = np.ascontiguousarray(arr[..., :3], dtype=np.uint8)

                # Saved observations are RGB; OpenCV expects BGR.
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
                written += 1

                if i == 0 or i == len(video_frames) - 1 or i % 100 == 0:
                    print(
                        "[DEBUG VIDEO FRAME]",
                        f"path={path.name}",
                        f"idx={i}",
                        f"shape={arr.shape}",
                        f"written={written}",
                        flush=True,
                    )
        finally:
            writer.release()

        exists = path.exists()
        size = path.stat().st_size if exists else 0

        print(
            "[DEBUG VIDEO]",
            f"released={path}",
            f"written={written}",
            f"exists={exists}",
            f"size={size}",
            flush=True,
        )

        if written == 0:
            raise RuntimeError(
                f"No valid frames were written to {path}"
            )

        if not exists or size <= 0:
            raise RuntimeError(
                f"Video file was not created correctly: {path}"
            )

        print(f"[DEBUG VIDEO] saved {path}", flush=True)
        return path

    front_result: Path | None = None
    wrist_result: Path | None = None

    # Save independently: one camera failing must not suppress the other.
    try:
        front_result = write_mp4(front_path, front_frames)
    except Exception as exc:
        print(
            "[DEBUG VIDEO] "
            f"FRONT failed: {type(exc).__name__}: {exc}",
            flush=True,
        )

    try:
        wrist_result = write_mp4(wrist_path, wrist_frames)
    except Exception as exc:
        print(
            "[DEBUG VIDEO] "
            f"WRIST failed: {type(exc).__name__}: {exc}",
            flush=True,
        )

    return front_result, wrist_result

def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def run() -> int:
    args = build_parser().parse_args()

    cfg = CollectorConfig(
        beaker_instruction=args.beaker_instruction,
        cylinder_instruction=args.cylinder_instruction,
        position_axis_sign=args.position_axis_sign,
        position_scale=args.position_scale,
        max_world_step=args.max_world_step,
        approach_z_offset=args.approach_z_offset,
        beaker_grasp_z_offset=args.beaker_grasp_z_offset,
        cylinder_grasp_z_offset=args.cylinder_grasp_z_offset,
        lift_z_offset=args.lift_z_offset,
        approach_xy_tolerance=args.approach_xy_tolerance,
        approach_z_tolerance=args.approach_z_tolerance,
        grasp_xy_tolerance=args.grasp_xy_tolerance,
        grasp_z_tolerance=args.grasp_z_tolerance,
        close_steps=args.close_steps,
        max_steps=args.max_steps,
        flip_primary=args.flip_primary,
        flip_wrist=args.flip_wrist,
        with_state=args.with_state,
    )

    output_dir: Path = args.output_dir
    success_dir = output_dir / "successes"
    failure_dir = output_dir / "failures"
    debug_dir = output_dir / "debug_frames"
    manifest_path = output_dir / "manifest.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    with (
        output_dir / "collector_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(cfg),
            file,
            ensure_ascii=False,
            indent=2,
        )

    teacher = ObjectTeacher(cfg)
    success_count = 0

    print(
        "[COLLECTOR] VLA-JEPA state saving: "
        f"{'ENABLED (8D)' if cfg.with_state else 'DISABLED'}"
    )

    print(
        f"[COLLECTOR] Connecting to "
        f"{args.host}:{args.port}..."
    )

    with socket.create_connection(
        (args.host, args.port),
        timeout=args.socket_timeout,
    ) as sock:
        sock.settimeout(args.socket_timeout)
        print("[COLLECTOR] Connected.")

        for attempt_id in range(args.max_attempts):
            if success_count >= args.num_successes:
                break

            # Successful episodes alternate targets. Every pair uses the same
            # scene seed, creating a counterfactual pair: nearly identical visual
            # scene / initial robot state, different language and target action.
            pair_index = success_count // 2
            first = args.first_target
            second = "blue_cylinder" if first == "beaker" else "beaker"
            target_object = first if success_count % 2 == 0 else second
            instruction = instruction_for_target(cfg, target_object)
            episode_seed = args.seed_offset + pair_index

            send_message(
                sock,
                {
                    "cmd": "reset",
                    "episode_idx": episode_seed,
                },
            )

            obs = validate_observation(
                recv_message(sock)
            )

            ee_pos, beaker_pos, cylinder_pos = extract_geometry(obs)
            target_pos = select_target(target_object, beaker_pos, cylinder_pos)
            teacher.reset(target_pos)

            frames: list[dict[str, Any]] = []
            terminal_primary: np.ndarray | None = None
            terminal_wrist: np.ndarray | None = None
            attempt_success = target_success(obs, target_object)

            last_phase = teacher.phase
            previous_phase: str | None = None

            print(
                f"[ATTEMPT {attempt_id:04d}] "
                f"seed={episode_seed} "
                f"target={target_object} "
                f"instruction={instruction!r} "
                f"beaker={np.round(beaker_pos, 4)} "
                f"cylinder={np.round(cylinder_pos, 4)}"
            )

            for step_idx in range(cfg.max_steps):
                ee_pos, beaker_pos, cylinder_pos = extract_geometry(obs)
                target_pos = select_target(target_object, beaker_pos, cylinder_pos)
                grasp_z_offset = (
                    cfg.beaker_grasp_z_offset
                    if target_object == "beaker"
                    else cfg.cylinder_grasp_z_offset
                )

                action, debug = teacher.action(
                    ee_pos,
                    target_pos,
                    grasp_z_offset=grasp_z_offset,
                )

                last_phase = str(debug["phase"])

                print(
                    "[TARGET TRACK]",
                    f"step={step_idx:03d}",
                    f"phase={last_phase}",
                    f"target={target_object}",
                    f"target_pos={np.round(target_pos, 4)}",
                    f"ee={np.round(ee_pos, 4)}",
                    flush=True,
                )

                primary_image = as_rgb(
                    obs["primary_image"],
                    flip_180=cfg.flip_primary,
                )

                wrist_image = as_rgb(
                    obs["wrist_image"],
                    flip_180=cfg.flip_wrist,
                )

                frame_record: dict[str, Any] = {
                    "primary_image": primary_image,
                    "wrist_image": wrist_image,
                    "action": action.copy(),
                    "ee_position": ee_pos.copy(),
                    "beaker_position": beaker_pos.copy(),
                    "blue_cylinder_position": cylinder_pos.copy(),
                    "target_position": target_pos.copy(),
                    "target_lift": target_lift(obs, target_object),
                    "target_object": target_object,
                    "instruction": instruction,
                    "phase": last_phase,
                }

                if cfg.with_state:
                    frame_record["state"] = extract_vla_state(obs)

                frames.append(frame_record)

                # ---------------------------------------------------------
                # Debug image saving
                # ---------------------------------------------------------

                phase_changed = (
                    previous_phase is None
                    or last_phase != previous_phase
                )

                periodic_debug_frame = (
                    step_idx % 5 == 0
                )

                should_save_debug = (
                    last_phase
                    in {
                        "approach",
                        "descend",
                        "close",
                        "lift",
                    }
                    and (
                        phase_changed
                        or periodic_debug_frame
                    )
                )

                if should_save_debug:
                    prefix = (
                        f"attempt_{attempt_id:04d}_"
                        f"step_{step_idx:03d}_"
                        f"{last_phase}"
                    )

                    front_path = (
                        debug_dir
                        / f"{prefix}_front.png"
                    )

                    wrist_path = (
                        debug_dir
                        / f"{prefix}_wrist.png"
                    )

                    Image.fromarray(
                        primary_image
                    ).save(front_path)

                    Image.fromarray(
                        wrist_image
                    ).save(wrist_path)

                    print(
                        "[DEBUG IMAGE] "
                        f"saved {front_path}"
                    )

                    print(
                        "[DEBUG IMAGE] "
                        f"saved {wrist_path}"
                    )

                # state末尾を参考値として表示。
                # 本当にfinger jointかはserver側のstate定義次第。
                if last_phase in {"close", "lift"}:
                    state_array = np.asarray(
                        obs["state"],
                        dtype=np.float32,
                    ).reshape(-1)

                    if state_array.size >= 2:
                        print(
                            "[GRIPPER DEBUG] "
                            f"step={step_idx:03d} "
                            f"phase={last_phase} "
                            f"state_last2="
                            f"{state_array[-2:]}"
                        )

                send_message(
                    sock,
                    {
                        "cmd": "step",
                        "action": action,
                    },
                )

                obs = validate_observation(
                    recv_message(sock)
                )

                if (
                    step_idx % 10 == 0
                    or last_phase in {
                        "close",
                        "lift",
                    }
                ):
                    print(
                        f"  step={step_idx:03d} "
                        f"phase={last_phase:8s} "
                        f"ee={np.round(ee_pos, 4)} "
                        f"xy_err="
                        f"{float(debug['xy_error']):.4f} "
                        f"z_err="
                        f"{float(debug['z_error']):.4f} "
                        f"target_lift="
                        f"{target_lift(obs, target_object):.4f} "
                        f"gripper_cmd={float(action[6]):+.1f}"
                    )

                previous_phase = last_phase

                if target_success(obs, target_object):
                    attempt_success = True

                    # 成功直後の観測も保存
                    success_primary = as_rgb(
                        obs["primary_image"],
                        flip_180=cfg.flip_primary,
                    )

                    success_wrist = as_rgb(
                        obs["wrist_image"],
                        flip_180=cfg.flip_wrist,
                    )
                    terminal_primary = success_primary
                    terminal_wrist = success_wrist

                    success_prefix = (
                        f"attempt_{attempt_id:04d}_"
                        f"step_{step_idx:03d}_"
                        "success"
                    )

                    Image.fromarray(
                        success_primary
                    ).save(
                        debug_dir
                        / f"{success_prefix}_front.png"
                    )

                    Image.fromarray(
                        success_wrist
                    ).save(
                        debug_dir
                        / f"{success_prefix}_wrist.png"
                    )

                    break

                if bool(obs.get("done", False)):
                    print(
                        f"[ATTEMPT {attempt_id:04d}] "
                        "Server returned done=True."
                    )
                    break

            if attempt_success:
                path = save_episode(
                    success_dir,
                    episode_id=success_count,
                    attempt_id=attempt_id,
                    success=True,
                    cfg=cfg,
                    target_object=target_object,
                    instruction=instruction,
                    frames=frames,
                )

                success_count += 1
                status = "success"

                print(
                    f"[ATTEMPT {attempt_id:04d}] "
                    f"SUCCESS "
                    f"({success_count}/"
                    f"{args.num_successes}) "
                    f"-> {path}"
                )

            else:
                status = "failure"
                path = None

                # 失敗終了時の最終観測を必ず保存
                try:
                    final_primary = as_rgb(
                        obs["primary_image"],
                        flip_180=cfg.flip_primary,
                    )

                    final_wrist = as_rgb(
                        obs["wrist_image"],
                        flip_180=cfg.flip_wrist,
                    )
                    terminal_primary = final_primary
                    terminal_wrist = final_wrist

                    final_prefix = (
                        f"attempt_{attempt_id:04d}_"
                        f"final_{last_phase}_failure"
                    )

                    Image.fromarray(
                        final_primary
                    ).save(
                        debug_dir
                        / f"{final_prefix}_front.png"
                    )

                    Image.fromarray(
                        final_wrist
                    ).save(
                        debug_dir
                        / f"{final_prefix}_wrist.png"
                    )

                except Exception as exc:
                    print(
                        "[DEBUG IMAGE] "
                        "Failed to save final frame: "
                        f"{exc}"
                    )

                if args.keep_failures and frames:
                    path = save_episode(
                        failure_dir,
                        episode_id=attempt_id,
                        attempt_id=attempt_id,
                        success=False,
                        cfg=cfg,
                        target_object=target_object,
                        instruction=instruction,
                        frames=frames,
                    )

                print(
                    f"[ATTEMPT {attempt_id:04d}] "
                    f"FAILURE "
                    f"phase={last_phase} "
                    f"steps={len(frames)} "
                    f"target={target_object} "
                    f"final_target_lift="
                    f"{target_lift(obs, target_object):.6f}"
                )

            # Save one full MP4 per camera only when explicitly requested.
            # PNG debug frames above are still saved exactly as before.
            if args.save_debug_video:
                try:
                    save_debug_videos(
                        debug_dir,
                        attempt_id=attempt_id,
                        frames=frames,
                        fps=args.debug_video_fps,
                        final_primary=terminal_primary,
                        final_wrist=terminal_wrist,
                    )
                except Exception as exc:
                    print(
                        "[DEBUG VIDEO] "
                        f"Failed to save attempt video: {exc}"
                    )

            append_manifest(
                manifest_path,
                {
                    "attempt_id": attempt_id,
                    "episode_seed": episode_seed,
                    "target_object": target_object,
                    "instruction": instruction,
                    "status": status,
                    "saved_path": (
                        str(path)
                        if path is not None
                        else None
                    ),
                    "num_steps": len(frames),
                    "final_phase": last_phase,
                    "final_target_lift": target_lift(obs, target_object),
                    "timestamp_unix": time.time(),
                },
            )

        try:
            send_message(
                sock,
                {"cmd": "close"},
            )

            response = recv_message(sock)

            print(
                "[COLLECTOR] "
                f"Server close response: {response}"
            )

        except Exception as exc:
            print(
                "[COLLECTOR] "
                f"Close command warning: {exc}"
            )

    print(
        f"[COLLECTOR] Finished: "
        f"{success_count}/"
        f"{args.num_successes} "
        "successful paired-task episodes."
    )

    return (
        0
        if success_count >= args.num_successes
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(run())