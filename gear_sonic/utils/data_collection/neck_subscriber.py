"""ZMQ subscriber for 2-DOF neck ``[yaw, pitch]`` streams.

Used for both neck streams:
  - command: pico_manager_thread_server.py PUB (port 5570)
  - state:   zed_server.py neck present-position PUB (port 5560)

Both publish a JSON list ``[yaw_rad, pitch_rad]`` with no topic prefix.
"""

import json

import numpy as np
import zmq


class ZMQNeckSubscriber:
    """Keeps the latest ``[yaw, pitch]`` sample (non-blocking, CONFLATE)."""

    def __init__(self, host: str, port: int):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self._latest = None
        print(f"[ZMQNeckSubscriber] Connected to tcp://{host}:{port}")

    def get_latest(self) -> np.ndarray | None:
        """Return the newest ``[yaw, pitch]`` as float64 array, or None if none yet."""
        try:
            raw = self._socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return self._latest
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._latest
        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
            self._latest = np.array([float(msg[0]), float(msg[1])], dtype=np.float64)
        return self._latest

    def close(self):
        self._socket.close()
        self._ctx.term()
