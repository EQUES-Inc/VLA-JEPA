import pickle
import socket
import struct


HOST = "127.0.0.1"
PORT = 5555


def send_message(sock, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data


def recv_message(sock):
    length = struct.unpack("!Q", recv_exact(sock, 8))[0]
    return pickle.loads(recv_exact(sock, length))


def step(sock, action):
    send_message(
        sock,
        {
            "cmd": "step",
            "action": action,
        },
    )
    return recv_message(sock)


with socket.create_connection((HOST, PORT)) as sock:

    # RESET
    send_message(
        sock,
        {
            "cmd": "reset",
            "episode_idx": 0,
        },
    )
    response = recv_message(sock)

    print("initial state:")
    print(response["state"])

    # --------------------------------
    # +X only
    # --------------------------------
    print("\n===== TEST +X =====")

    response = step(
        sock,
        [
            0.2, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0,
        ],
    )

    print(response["state"])

    # CLOSE
    send_message(sock, {"cmd": "close"})
    print(recv_message(sock))