#!/usr/bin/env python3
"""
VLA-JEPA zero-shot evaluator for MATTERIX.

Run this script in the `vla-jepa` conda environment.

Responsibilities:
    - connect to matterix_server.py over TCP
    - receive MATTERIX RGB observations
    - run VLA-JEPA inference through M1Inference
    - send a 7D delta action:

        [dx, dy, dz, dRx, dRy, dRz, gripper]

    - MATTERIX-side conversion into the 8D absolute IK action happens
      inside matterix_server.py:

        [x, y, z, qw, qx, qy, qz, gripper]

Recommended location:
    ~/VLA-JEPA/examples/MATTERIX/eval_matterix.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pathlib
import pickle
import socket
import struct
import traceback
from typing import Any

import imageio.v2 as imageio
import numpy as np


# ===========================================================================
# IMPORTANT:
#
# Replace this import with the exact import used by your working
# eval_libero.py.
#
# Example only:
#
# from xxx.yyy import M1Inference
#
# ===========================================================================
try:
    from examples.LIBERO.model2libero_interface import M1Inference  # type: ignore
except ImportError as exc:
    raise ImportError(
        "Could not import M1Inference.\n"
        "Replace:\n"
        "    from m1_inference import M1Inference\n"
        "with the exact import line used by your working LIBERO evaluator."
    ) from exc


LOGGER = logging.getLogger("eval_matterix")

DEBUG_LOGGING = False


def debug_print(*values, **kwargs) -> None:
    """Print verbose client diagnostics only when --debug is enabled."""
    if DEBUG_LOGGING:
        kwargs.setdefault("flush", True)
        print(*values, **kwargs)


def print_progress(step: int, max_steps: int, width: int = 30) -> None:
    """Show a compact one-line rollout progress bar in normal mode."""
    if DEBUG_LOGGING:
        return

    current = min(step + 1, max_steps)
    ratio = current / max_steps if max_steps > 0 else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)

    print(
        f"\r[PROGRESS] [{bar}] {current:>4}/{max_steps:<4} "
        f"({ratio * 100:5.1f}%)",
        end="",
        flush=True,
    )


def finish_progress() -> None:
    """Terminate the current progress-bar line in normal mode."""
    if not DEBUG_LOGGING:
        print(flush=True)



# ===========================================================================
# TCP protocol
# ===========================================================================


def send_message(
    sock: socket.socket,
    obj: Any,
) -> None:
    """
    Serialize a Python object with pickle and send it with an 8-byte
    big-endian payload-size header.
    """
    payload = pickle.dumps(
        obj,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    header = struct.pack(
        "!Q",
        len(payload),
    )

    sock.sendall(header + payload)


def recv_exact(
    sock: socket.socket,
    n: int,
) -> bytes:
    """
    Receive exactly n bytes from the socket.
    """
    chunks: list[bytes] = []
    remaining = n

    while remaining > 0:
        chunk = sock.recv(remaining)

        if not chunk:
            raise ConnectionError(
                "Socket closed while receiving data."
            )

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def recv_message(
    sock: socket.socket,
) -> Any:
    """
    Receive one length-prefixed pickle message.
    """
    header = recv_exact(
        sock,
        8,
    )

    (length,) = struct.unpack(
        "!Q",
        header,
    )

    payload = recv_exact(
        sock,
        length,
    )

    return pickle.loads(payload)


# ===========================================================================
# MATTERIX response validation
# ===========================================================================


def check_server_error(
    response: Any,
) -> None:
    """
    Raise an exception if matterix_server.py returned a server-side error.
    """
    if not isinstance(response, dict):
        return

    if not response.get(
        "server_error",
        False,
    ):
        return

    error_type = response.get(
        "error_type",
        "UnknownError",
    )

    error_message = response.get(
        "error_message",
        "Unknown MATTERIX server error.",
    )

    raise RuntimeError(
        "MATTERIX server returned an error:\n"
        f"  type: {error_type}\n"
        f"  message: {error_message}"
    )


def validate_observation(
    obs: dict[str, Any],
) -> None:
    """
    Validate the response received from matterix_server.py.
    """
    if not isinstance(obs, dict):
        raise TypeError(
            "MATTERIX response must be a dict, "
            f"got {type(obs)}"
        )

    check_server_error(obs)

    required = {
        "primary_image",
        "wrist_image",
        "success",
        "done",
    }

    missing = required - set(obs)

    if missing:
        raise KeyError(
            "MATTERIX observation is missing required keys: "
            f"{sorted(missing)}\n"
            f"Available keys: {sorted(obs.keys())}"
        )


# ===========================================================================
# Action utilities
# ===========================================================================

def binarize_gripper_open(
    open_gripper: np.ndarray,
) -> np.ndarray:
    """
    Convert the semantic VLA-JEPA open_gripper output
    to the MATTERIX physical convention.

    raw > 0.5  -> +1: open
    raw <= 0.5 -> -1: close
    """
    array = np.asarray(
        open_gripper,
        dtype=np.float32,
    ).reshape(-1)

    if array.size != 1:
        raise ValueError(
            "Expected exactly one gripper value, "
            f"got shape {array.shape}"
        )

    value = float(array[0])

    command = (
        1.0
        if value > 0.5
        else -1.0
    )

    return np.asarray(
        [command],
        dtype=np.float32,
    )


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot VLA-JEPA evaluation on "
            "MATTERIX FrankaBeakerLift."
        )
    )

    # -----------------------------------------------------------------------
    # MATTERIX TCP server
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="MATTERIX TCP server host.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="MATTERIX TCP server port.",
    )

    # -----------------------------------------------------------------------
    # VLA-JEPA / M1Inference model server
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--pretrained-path",
        type=str,
        required=True,
        help="Path to the VLA-JEPA checkpoint.",
    )

    parser.add_argument(
        "--model-host",
        type=str,
        default="127.0.0.1",
        help="VLA-JEPA model server host used by M1Inference.",
    )

    parser.add_argument(
        "--model-port",
        type=int,
        required=True,
        help="VLA-JEPA model server port used by M1Inference.",
    )

    parser.add_argument(
        "--resize-size",
        type=int,
        default=224,
        help="Image size passed to M1Inference.",
    )

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--task-description",
        type=str,
        default="Pick up the beaker",
        help="Language instruction passed to VLA-JEPA.",
    )

    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum number of VLA action steps per episode.",
    )

    parser.add_argument(
        "--with-state",
        action="store_true",
        help=(
            "Pass the MATTERIX robot state to VLA-JEPA. "
            "Start without this option for initial integration testing."
        ),
    )

    # -----------------------------------------------------------------------
    # Image preprocessing
    # -----------------------------------------------------------------------

    # parser.add_argument(
    #     "--flip-images",
    #     action="store_true",
    #     help=(
    #         "Apply [::-1, ::-1] to both primary and wrist images, "
    #         "matching the existing LIBERO evaluator behavior."
    #     ),
    # )

    parser.add_argument(
        "--flip-primary-image",
        action="store_true",
        help="Rotate the primary/front image by 180 degrees.",
    )

    parser.add_argument(
        "--flip-wrist-image",
        action="store_true",
        help="Rotate the wrist image by 180 degrees.",
    )

    # -----------------------------------------------------------------------
    # Debug camera frames
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--debug-image-out-path",
        type=str,
        default="./matterix_debug_images",
        help=(
            "Directory for reset and selected step images from both cameras."
        ),
    )

    # -----------------------------------------------------------------------
    # Video
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--video-out-path",
        type=str,
        default="./matterix_videos",
        help="Directory for optional evaluation videos.",
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save primary-camera videos.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose client-side debug logging.",
    )

    return parser


# ===========================================================================
# Image helpers
# ===========================================================================


def save_received_raw_image(
    image: Any,
    output_path: pathlib.Path,
    name: str,
    expected_sha256: str | None = None,
) -> str:
    """
    Save the exact image array received from MATTERIX before flip/preprocessing.

    Returns the SHA-256 digest of the received array bytes.
    """
    array = np.asarray(image)

    if array.ndim != 3:
        raise ValueError(
            f"{name}: expected HWC image, got {array.shape}"
        )

    contiguous = np.ascontiguousarray(array.copy())
    digest = hashlib.sha256(contiguous.tobytes()).hexdigest()

    debug_print(
        f"[CLIENT][RAW] {name}",
        f"shape={contiguous.shape}",
        f"dtype={contiguous.dtype}",
        f"min={float(np.nanmin(contiguous)):.6f}",
        f"max={float(np.nanmax(contiguous)):.6f}",
        f"mean={float(np.nanmean(contiguous)):.6f}",
        f"sha256={digest}",
        f"server_sha256={expected_sha256}",
        f"hash_match={expected_sha256 == digest if expected_sha256 else 'unknown'}",
        flush=True,
    )

    save_array = contiguous

    if save_array.shape[-1] == 4:
        save_array = save_array[..., :3]

    if save_array.shape[-1] != 3:
        raise ValueError(
            f"{name}: expected RGB/RGBA image, got {save_array.shape}"
        )

    if np.issubdtype(save_array.dtype, np.floating):
        max_value = float(np.nanmax(save_array))
        if max_value <= 1.0 + 1e-6:
            save_array = save_array * 255.0

    save_array = np.nan_to_num(
        save_array,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )
    save_array = np.clip(save_array, 0, 255).astype(np.uint8)
    save_array = np.ascontiguousarray(save_array)

    # imageio.imwrite(output_path, save_array)

    debug_print(
        f"[CLIENT][RAW] saved {output_path}",
        flush=True,
    )

    return digest


def prepare_image(
    image: Any,
    flip: bool,
    name: str,
) -> np.ndarray:
    """
    Convert a MATTERIX RGB image into contiguous uint8 HWC format.
    """
    array = np.asarray(image)

    debug_print(
        f"[CLIENT][IMAGE] {name} raw shape: {array.shape}",
        flush=True,
    )

    debug_print(
        f"[CLIENT][IMAGE] {name} raw dtype: {array.dtype}",
        flush=True,
    )

    if array.ndim != 3:
        raise ValueError(
            f"{name} must have 3 dimensions HWC, "
            f"got shape {array.shape}"
        )

    if array.shape[-1] == 4:
        array = array[..., :3]

    if array.shape[-1] != 3:
        raise ValueError(
            f"{name} must have 3 RGB channels, "
            f"got shape {array.shape}"
        )

    if np.issubdtype(
        array.dtype,
        np.floating,
    ):
        max_value = (
            float(array.max())
            if array.size > 0
            else 0.0
        )

        if max_value <= 1.0 + 1e-6:
            array = array * 255.0

    array = np.clip(
        array,
        0,
        255,
    ).astype(np.uint8)

    if flip:
        array = array[::-1, ::-1]

    array = np.ascontiguousarray(array)

    debug_print(
        f"[CLIENT][IMAGE] {name} final shape: {array.shape}",
        flush=True,
    )

    debug_print(
        f"[CLIENT][IMAGE] {name} final dtype: {array.dtype}",
        flush=True,
    )

    return array


# ===========================================================================
# VLA-JEPA action parsing
# ===========================================================================


def parse_vla_response(
    response: Any,
) -> np.ndarray:
    """
    Parse M1Inference response and return a 7D action:

        [dx, dy, dz, dRx, dRy, dRz, gripper]
    """
    debug_print(
        f"[CLIENT][MODEL] response type: {type(response)}",
        flush=True,
    )

    if not isinstance(response, dict):
        raise TypeError(
            "M1Inference response must be a dict, "
            f"got {type(response)}"
        )

    debug_print(
        f"[CLIENT][MODEL] response keys: {list(response.keys())}",
        flush=True,
    )

    if "raw_action" not in response:
        raise KeyError(
            "M1Inference response has no 'raw_action' field. "
            f"Available keys: {list(response.keys())}"
        )

    raw_action = response["raw_action"]

    if not isinstance(raw_action, dict):
        raise TypeError(
            "'raw_action' must be a dict, "
            f"got {type(raw_action)}"
        )

    debug_print(
        f"[CLIENT][MODEL] raw_action keys: {list(raw_action.keys())}",
        flush=True,
    )

    world_vector_delta = np.asarray(
        raw_action.get(
            "world_vector"
        ),
        dtype=np.float32,
    ).reshape(-1)

    rotation_delta = np.asarray(
        raw_action.get(
            "rotation_delta"
        ),
        dtype=np.float32,
    ).reshape(-1)

    open_gripper = np.asarray(
        raw_action.get(
            "open_gripper"
        ),
        dtype=np.float32,
    ).reshape(-1)

    debug_print(
        "[CLIENT][ACTION] world_vector_delta:",
        world_vector_delta,
        flush=True,
    )

    debug_print(
        "[CLIENT][ACTION] rotation_delta:",
        rotation_delta,
        flush=True,
    )

    debug_print(
        "[CLIENT][ACTION] open_gripper:",
        open_gripper,
        flush=True,
    )

    if world_vector_delta.size != 3:
        raise ValueError(
            "Unexpected world_vector shape: "
            f"{world_vector_delta.shape}"
        )

    if rotation_delta.size != 3:
        raise ValueError(
            "Unexpected rotation_delta shape: "
            f"{rotation_delta.shape}"
        )

    if open_gripper.size != 1:
        raise ValueError(
            "Unexpected open_gripper shape: "
            f"{open_gripper.shape}"
        )

    gripper = binarize_gripper_open(
        open_gripper
    )

    action_7d = np.concatenate(
        [
            world_vector_delta,
            rotation_delta,
            gripper,
        ],
        axis=0,
    ).astype(np.float32)

    if action_7d.shape != (7,):
        raise RuntimeError(
            f"Unexpected VLA action shape: {action_7d.shape}"
        )

    debug_print(
        "[CLIENT][ACTION] final 7D action:",
        action_7d,
        flush=True,
    )

    return action_7d


# ===========================================================================
# Main evaluation
# ===========================================================================


def main() -> None:
    global DEBUG_LOGGING

    args = build_parser().parse_args()
    DEBUG_LOGGING = bool(args.debug)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    LOGGER.debug(
        "Arguments:\n%s",
        json.dumps(
            vars(args),
            indent=2,
        ),
    )

    video_dir = pathlib.Path(
        args.video_out_path
    )

    debug_image_dir = pathlib.Path(
        args.debug_image_out_path
    )
    debug_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.save_video:
        video_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -----------------------------------------------------------------------
    # Initialize VLA-JEPA inference client.
    # -----------------------------------------------------------------------

    debug_print(
        "[CLIENT] Initializing M1Inference...",
        flush=True,
    )
        
    model = M1Inference(
        policy_ckpt_path=args.pretrained_path,
        host=args.model_host,
        port=args.model_port,
        image_size=(args.resize_size, args.resize_size),
    )

    debug_print(
        "[CLIENT] M1Inference initialized successfully",
        flush=True,
    )

    total_successes = 0

    sock: socket.socket | None = None

    try:
        # -------------------------------------------------------------------
        # Connect to MATTERIX server.
        # -------------------------------------------------------------------

        debug_print(
            f"[CLIENT] Connecting to MATTERIX server "
            f"{args.host}:{args.port}...",
            flush=True,
        )

        sock = socket.create_connection(
            (
                args.host,
                args.port,
            )
        )

        LOGGER.debug(
            "Connected to MATTERIX server at %s:%d",
            args.host,
            args.port,
        )

        # -------------------------------------------------------------------
        # Episode loop
        # -------------------------------------------------------------------

        for episode_idx in range(
            args.num_trials
        ):
            LOGGER.info(
                "Episode %d/%d | %s",
                episode_idx + 1,
                args.num_trials,
                args.task_description,
            )

            debug_print(
                f"[CLIENT][EPISODE] Starting episode {episode_idx}",
                flush=True,
            )

            # ---------------------------------------------------------------
            # Reset VLA-JEPA inference state.
            # ---------------------------------------------------------------

            debug_print(
                "[CLIENT][EPISODE] Calling model.reset()...",
                flush=True,
            )

            model.reset(
                task_description=args.task_description,
            )

            debug_print(
                "[CLIENT][EPISODE] model.reset() completed",
                flush=True,
            )

            # ---------------------------------------------------------------
            # Reset MATTERIX.
            # ---------------------------------------------------------------

            reset_request = {
                "cmd": "reset",
                "episode_idx": episode_idx,
            }

            debug_print(
                f"[CLIENT][RESET] Sending request: {reset_request}",
                flush=True,
            )

            send_message(
                sock,
                reset_request,
            )

            debug_print(
                "[CLIENT][RESET] Request sent",
                flush=True,
            )

            debug_print(
                "[CLIENT][RESET] Waiting for response...",
                flush=True,
            )

            obs = recv_message(sock)

            debug_print(
                "[CLIENT][RESET] Response received",
                flush=True,
            )

            debug_print(
                f"[CLIENT][RESET] Response type: {type(obs)}",
                flush=True,
            )

            if isinstance(obs, dict):
                debug_print(
                    f"[CLIENT][RESET] Response keys: {list(obs.keys())}",
                    flush=True,
                )

            check_server_error(obs)

            validate_observation(obs)

            debug_print(
                "[CLIENT][RESET] Observation validated successfully",
                flush=True,
            )

            reset_response_id = int(
                obs.get("response_id", -1)
            )

            reset_primary_raw_path = (
                debug_image_dir
                / (
                    f"episode_{episode_idx:03d}_"
                    f"response_{reset_response_id:06d}_"
                    "front_reset_received_raw.png"
                )
            )

            reset_wrist_raw_path = (
                debug_image_dir
                / (
                    f"episode_{episode_idx:03d}_"
                    f"response_{reset_response_id:06d}_"
                    "wrist_reset_received_raw.png"
                )
            )

            reset_primary_img = prepare_image(
                image=obs["primary_image"],
                flip=args.flip_primary_image,
                name="reset_primary_image",
            )
            reset_wrist_img = prepare_image(
                image=obs["wrist_image"],
                flip=args.flip_wrist_image,
                name="reset_wrist_image",
            )
            fixed_primary_img = reset_primary_img.copy()
            fixed_wrist_img = reset_wrist_img.copy()

            reset_primary_path = (
                debug_image_dir
                / (
                    f"episode_{episode_idx:03d}_"
                    f"response_{reset_response_id:06d}_"
                    "front_reset.png"
                )
            )
            reset_wrist_path = (
                debug_image_dir
                / (
                    f"episode_{episode_idx:03d}_"
                    f"response_{reset_response_id:06d}_"
                    "wrist_reset.png"
                )
            )

            debug_print(
                "[CLIENT][RESET] primary_image:",
                np.asarray(
                    obs["primary_image"]
                ).shape,
                np.asarray(
                    obs["primary_image"]
                ).dtype,
                flush=True,
            )

            debug_print(
                "[CLIENT][RESET] wrist_image:",
                np.asarray(
                    obs["wrist_image"]
                ).shape,
                np.asarray(
                    obs["wrist_image"]
                ).dtype,
                flush=True,
            )

            if "state" in obs:
                state_value = obs["state"]

                if state_value is None:
                    debug_print(
                        "[CLIENT][RESET] state: None",
                        flush=True,
                    )
                else:
                    debug_print(
                        "[CLIENT][RESET] state:",
                        np.asarray(
                            state_value
                        ).shape,
                        np.asarray(
                            state_value
                        ).dtype,
                        flush=True,
                    )

                    debug_print(
                        "[CLIENT][RESET] state values:",
                        np.asarray(
                            state_value
                        ),
                        flush=True,
                    )

            replay_images: list[np.ndarray] = []
            replay_wrist_images: list[np.ndarray] = []


            success = False

            # ---------------------------------------------------------------
            # Rollout loop
            # ---------------------------------------------------------------

            for step in range(
                args.max_steps
            ):
                print_progress(
                    step=step,
                    max_steps=args.max_steps,
                )

                debug_print(
                    "------------------------------------------------------------",
                    flush=True,
                )

                debug_print(
                    f"[CLIENT][STEP] step={step}",
                    flush=True,
                )

                primary_img = prepare_image(
                    image=obs["primary_image"],
                    flip=args.flip_primary_image,
                    name="primary_image",
                )

                wrist_img = prepare_image(
                    image=obs["wrist_image"],
                    flip=args.flip_wrist_image,
                    name="wrist_image",
                )

                # primary_img = np.zeros_like(primary_img)
                # wrist_img = np.zeros_like(wrist_img)
                # primary_img = fixed_primary_img.copy()
                # wrist_img = fixed_wrist_img.copy()


                if args.save_video:
                    replay_images.append(
                        primary_img.copy()
                    )
                    replay_wrist_images.append(
                        wrist_img.copy()
                    )

                # -----------------------------------------------------------
                # Build the same general VLA-JEPA input structure used by
                # the LIBERO evaluator.
                # -----------------------------------------------------------

                obs_input: dict[str, Any] = {
                    "images": [
                        primary_img,
                        wrist_img,
                    ],
                    "task_description": (
                        args.task_description
                    ),
                    "step": step,
                }

                if args.with_state:
                    if "state" not in obs:
                        raise KeyError(
                            "--with-state was requested, but MATTERIX "
                            "did not return a 'state' field."
                        )

                    if obs["state"] is None:
                        raise ValueError(
                            "--with-state was requested, but the MATTERIX "
                            "server returned state=None."
                        )

                    state = np.asarray(
                        obs["state"],
                        dtype=np.float32,
                    ).reshape(-1)

                    # Match the existing LIBERO pattern where state has a
                    # batch dimension.
                    obs_input["state"] = np.expand_dims(
                        state,
                        axis=0,
                    )

                    debug_print(
                        "[CLIENT][MODEL] state input shape:",
                        obs_input["state"].shape,
                        flush=True,
                    )

                debug_print(
                    "[CLIENT][MODEL] Calling model.step()...",
                    flush=True,
                )

                debug_print(
                    "[CLIENT][MODEL] primary image shape:",
                    primary_img.shape,
                    flush=True,
                )

                debug_print(
                    "[CLIENT][MODEL] wrist image shape:",
                    wrist_img.shape,
                    flush=True,
                )

                debug_print(
                    "[CLIENT][MODEL] task description:",
                    args.task_description,
                    flush=True,
                )

                debug_print(
                    "[CLIENT][MODEL] step:",
                    step,
                    flush=True,
                )

                # -----------------------------------------------------------
                # VLA-JEPA inference
                # -----------------------------------------------------------

                response = model.step(
                    **obs_input
                )

                debug_print(
                    "[CLIENT][MODEL] model.step() completed",
                    flush=True,
                )

                raw_actions = response.get("raw_actions")

                if raw_actions is not None:
                    raw_actions_array = np.asarray(raw_actions)

                    debug_print(
                        "[CLIENT][CHUNK] raw_actions shape:",
                        raw_actions_array.shape,
                        flush=True,
                    )

                    debug_print(
                        "[CLIENT][CHUNK] raw_actions:",
                        raw_actions_array,
                        flush=True,
                    )
                
                debug_print(
                    "[CLIENT][CHUNK] raw_action:",
                    response.get("raw_action"),
                    flush=True,
                )

                # -----------------------------------------------------------
                # Parse 7D VLA action
                # -----------------------------------------------------------

                delta_action = parse_vla_response(
                    response
                )


                if "beaker_position" in obs and obs.get("state") is not None:
                    ee_position = np.asarray(
                        obs["state"],
                        dtype=np.float32,
                    ).reshape(-1)[:3]

                    beaker_position = np.asarray(
                        obs["beaker_position"],
                        dtype=np.float32,
                    ).reshape(3)

                    target_vector = beaker_position - ee_position

                    target_norm = np.linalg.norm(target_vector)
                    position_axis_sign = np.asarray(
                        [1.0, 1.0, 1.0],
                        dtype=np.float32,
                    )

                    mapped_translation = (
                        delta_action[:3]
                        * position_axis_sign
                    )                
                    action_norm = np.linalg.norm(mapped_translation)


                    if target_norm > 1e-8:
                        target_direction = target_vector / target_norm
                    else:
                        target_direction = np.zeros(3, dtype=np.float32)

                    if action_norm > 1e-8:
                        action_direction = mapped_translation / action_norm
                    else:
                        action_direction = np.zeros(3, dtype=np.float32)

                    alignment = float(
                        np.dot(target_direction, action_direction)
                    )

                    debug_print(
                        "[CLIENT][GEOMETRY] EE position:",
                        ee_position,
                        flush=True,
                    )
                    debug_print(
                        "[CLIENT][GEOMETRY] beaker position:",
                        beaker_position,
                        flush=True,
                    )
                    debug_print(
                        "[CLIENT][GEOMETRY] target vector:",
                        target_vector,
                        flush=True,
                    )

                    debug_print(
                        "[CLIENT][GEOMETRY] raw VLA translation:",
                        delta_action[:3],
                        flush=True,
                    )

                    debug_print(
                        "[CLIENT][GEOMETRY] expected server-mapped translation:",
                        mapped_translation,
                        flush=True,
                    )

                    debug_print(
                        "[CLIENT][GEOMETRY] direction alignment:",
                        alignment,
                        flush=True,
                    )

                # -----------------------------------------------------------
                # Send action to MATTERIX
                # -----------------------------------------------------------

                step_request = {
                    "cmd": "step",
                    "action": delta_action,
                }

                debug_print(
                    "[CLIENT][STEP] Sending raw VLA 7D action:",
                    delta_action,
                    flush=True,
                )

                send_message(
                    sock,
                    step_request,
                )

                debug_print(
                    "[CLIENT][STEP] Action sent",
                    flush=True,
                )

                debug_print(
                    "[CLIENT][STEP] Waiting for MATTERIX response...",
                    flush=True,
                )

                obs = recv_message(sock)

                debug_print(
                    "[CLIENT][STEP] MATTERIX response received",
                    flush=True,
                )

                check_server_error(obs)

                validate_observation(obs)

                debug_print(
                    "[CLIENT][STEP] Response validated",
                    flush=True,
                )

                beaker_lift = float(
                    obs.get(
                        "beaker_lift",
                        float("nan"),
                    )
                )

                beaker_z = float(
                    obs.get(
                        "beaker_z",
                        float("nan"),
                    )
                )

                current_success = bool(
                    obs["success"]
                )

                current_done = bool(
                    obs["done"]
                )

                debug_print(
                    f"[CLIENT][STEP] beaker_z={beaker_z:.6f}",
                    flush=True,
                )

                debug_print(
                    f"[CLIENT][STEP] beaker_lift={beaker_lift:.6f}",
                    flush=True,
                )

                debug_print(
                    f"[CLIENT][STEP] success={current_success}",
                    flush=True,
                )

                debug_print(
                    f"[CLIENT][STEP] done={current_done}",
                    flush=True,
                )

                if step % 10 == 0:
                    LOGGER.debug(
                        (
                            "step=%d | "
                            "beaker_lift=%.4f m | "
                            "success=%s"
                        ),
                        step,
                        beaker_lift,
                        current_success,
                    )

                # -----------------------------------------------------------
                # Success
                # -----------------------------------------------------------

                if current_success:
                    success = True
                    total_successes += 1
                    finish_progress()

                    LOGGER.info(
                        "SUCCESS at step %d",
                        step,
                    )

                    break

                # -----------------------------------------------------------
                # Episode termination
                # -----------------------------------------------------------

                if current_done:
                    finish_progress()

                    LOGGER.info(
                        "Episode terminated at step %d",
                        step,
                    )

                    break

            # ---------------------------------------------------------------
            # Episode result
            # ---------------------------------------------------------------

            if not success and not current_done:
                finish_progress()

            suffix = (
                "success"
                if success
                else "failure"
            )

            if (
                args.save_video
                and replay_images
            ):
                output_path = (
                    video_dir
                    / (
                        "franka_beaker_lift_"
                        f"episode{episode_idx}_"
                        f"{suffix}.mp4"
                    )
                )

                ###MP4動画保存用に左右上下反転。
                fixed_images = [
                    np.rot90(image, 0)
                    for image in replay_images
                ]
                imageio.mimwrite(
                    output_path,
                    fixed_images,
                    fps=10,
                    codec="libx264",
                )

                fixed_images = [
                    np.rot90(image, 0)
                    for image in replay_wrist_images
                ]
                imageio.mimwrite(
                    output_path.with_name(f"{output_path.stem}_wrist{output_path.suffix}"),
                    fixed_images,
                    fps=10,
                    codec="libx264",
                )

                LOGGER.debug(
                    "Saved video: %s",
                    output_path,
                )

            completed = (
                episode_idx + 1
            )

            success_rate = (
                100.0
                * total_successes
                / completed
            )

            LOGGER.info(
                (
                    "Episode result: %s | "
                    "total success rate = "
                    "%d/%d (%.1f%%)"
                ),
                suffix,
                total_successes,
                completed,
                success_rate,
            )

        # -------------------------------------------------------------------
        # Graceful close
        # -------------------------------------------------------------------

        debug_print(
            "[CLIENT] Sending close command to MATTERIX...",
            flush=True,
        )

        try:
            send_message(
                sock,
                {
                    "cmd": "close",
                },
            )

            close_response = recv_message(
                sock
            )

            debug_print(
                "[CLIENT] MATTERIX close response:",
                close_response,
                flush=True,
            )

        except Exception as exc:
            debug_print(
                "[CLIENT] Warning: graceful close failed:",
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    finally:
        if sock is not None:
            debug_print(
                "[CLIENT] Closing MATTERIX socket",
                flush=True,
            )

            try:
                sock.close()
            except Exception:
                traceback.print_exc()

    final_success_rate = (
        100.0
        * total_successes
        / args.num_trials
        if args.num_trials > 0
        else 0.0
    )

    LOGGER.info(
        (
            "Final success rate: "
            "%d/%d (%.1f%%)"
        ),
        total_successes,
        args.num_trials,
        final_success_rate,
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\n[CLIENT] KeyboardInterrupt received",
            flush=True,
        )

        raise

    except Exception as exc:
        print(
            "============================================================",
            flush=True,
        )

        print(
            f"[CLIENT ERROR] {type(exc).__name__}: {exc}",
            flush=True,
        )

        print(
            "============================================================",
            flush=True,
        )

        traceback.print_exc()

        raise