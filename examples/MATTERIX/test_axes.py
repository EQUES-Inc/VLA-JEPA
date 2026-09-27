#!/usr/bin/env python3

from __future__ import annotations

import pickle
import socket
import struct
import time
from typing import Any

import numpy as np


HOST = "127.0.0.1"
PORT = 5555


def send_message(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(
        obj,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    sock.sendall(
        struct.pack("!Q", len(payload)) + payload
    )


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size

    while remaining > 0:
        chunk = sock.recv(remaining)

        if not chunk:
            raise ConnectionError(
                "Socket closed while receiving data."
            )

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Any:
    header = recv_exact(sock, 8)
    (payload_size,) = struct.unpack("!Q", header)
    payload = recv_exact(sock, payload_size)
    return pickle.loads(payload)


def read_ee_position(obs: dict[str, Any]) -> np.ndarray:
    state = np.asarray(
        obs["state"],
        dtype=np.float32,
    ).reshape(-1)

    if state.size < 3:
        raise ValueError(
            f"Unexpected state shape: {state.shape}"
        )

    return state[:3].copy()


def run_test(
    sock: socket.socket,
    name: str,
    action: np.ndarray,
) -> None:
    print()
    print("=" * 60)
    print(f"Test: {name}")
    print(f"7D input action: {action}")
    print("=" * 60)

    # 各テストを同じ初期状態から始める
    send_message(
        sock,
        {
            "cmd": "reset",
            "episode_idx": 0,
        },
    )

    obs_before = recv_message(sock)

    if obs_before.get("server_error", False):
        raise RuntimeError(obs_before)

    pos_before = read_ee_position(obs_before)

    print("EE before:", pos_before)

    send_message(
        sock,
        {
            "cmd": "step",
            "action": action.astype(np.float32),
        },
    )

    obs_after = recv_message(sock)

    if obs_after.get("server_error", False):
        raise RuntimeError(obs_after)

    pos_after = read_ee_position(obs_after)
    displacement = pos_after - pos_before

    print("EE after :", pos_after)
    print("Measured displacement:", displacement)

    time.sleep(0.5)


def main() -> None:
    tests = [
        (
            "VLA +X",
            np.array(
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        ),
        (
            "VLA +Y",
            np.array(
                [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        ),
        (
            "VLA +Z",
            np.array(
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        ),
    ]

    with socket.create_connection((HOST, PORT)) as sock:
        print(f"Connected to MATTERIX server: {HOST}:{PORT}")

        for name, action in tests:
            run_test(
                sock=sock,
                name=name,
                action=action,
            )

        send_message(
            sock,
            {
                "cmd": "close",
            },
        )

        try:
            print("Close response:", recv_message(sock))
        except ConnectionError:
            pass


if __name__ == "__main__":
    main()