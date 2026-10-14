"""
Replay a recorded episode (run_data_exporter LeRobot format) on the robot.

Streams the recorded motion tokens + Dex3 hand joints to the C++ deploy as
protocol v4 on the ``pose`` topic, and the recorded neck command to
zed_server.py (same wire format as pico_manager_thread_server's NeckPublisher).

Deploy must run with ``--input-type zmq`` (do NOT run pico_manager at the same
time; it binds the same ports):

    ./deploy.sh --input-type zmq real

Flow:
  1. Start this script, then press ']' in the deploy terminal to start the
     policy. The script snapshots the robot's standing token from the deploy
     state stream and keeps publishing it (hands open, neck 0, 0).
  2. Deploy terminal: Enter to enable ZMQ streaming. The robot keeps standing
     because the held token is the standing token.
  3. This terminal: 'p' blends to the first recorded frame over
     --blend-seconds, plays the episode, then holds the last frame.
     'p' again replays from the start.
  4. Deploy terminal: Enter disables ZMQ streaming (smooth return to
     standing), 'o' stops the policy.
  5. This terminal: 'q' quits.

Usage (from repo root):
    source .venv_data_collection/bin/activate
    python gear_sonic/scripts/replay_episode.py --dataset <dataset_dir> --episode 0
"""

import argparse
import atexit
import json
import os
from pathlib import Path
import queue
import sys
import termios
import threading
import time
import tty

import numpy as np
import pandas as pd
import zmq

from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

NECK_PORT = 5570


def load_episode(dataset_dir: Path, episode: int, with_neck: bool):
    """Return (fps, tokens [T,64], left_hand [T,7], right_hand [T,7], neck [T,2] or None)."""
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    matches = sorted(dataset_dir.rglob(f"episode_{episode:06d}.parquet"))
    if not matches:
        raise SystemExit(f"[Replay] episode {episode} not found under {dataset_dir}")
    df = pd.read_parquet(matches[0])

    def column(key):
        if key not in df.columns:
            raise SystemExit(f"[Replay] column '{key}' missing in {matches[0]}")
        return np.stack(df[key].to_numpy()).astype(np.float32)

    tokens = column("action.motion_token")
    left_hand = column("teleop.left_hand_joints")
    right_hand = column("teleop.right_hand_joints")
    neck = column("action.neck") if with_neck else None
    print(f"[Replay] loaded {matches[0]}: {len(tokens)} frames @ {fps:g} fps")
    return fps, tokens, left_hand, right_hand, neck


def read_standing_token(host: str, port: int) -> np.ndarray:
    """Snapshot the robot's current encoder token from the deploy state stream.
    Deploy only publishes state once the policy is started (']' in its terminal),
    so this waits until then."""
    sub = ZMQStateSubscriber(host=host, port=port)
    print("[Replay] waiting for deploy state: press ']' in the deploy terminal ...")
    try:
        while True:
            msg = sub.get_msg()
            if msg is not None and len(msg.get("token_state", [])) > 0:
                return np.asarray(msg["token_state"], dtype=np.float32)
            time.sleep(0.02)
    finally:
        sub.close()


def start_key_listener() -> queue.Queue:
    """Single-key terminal input (no Enter needed) on a daemon thread."""
    keys = queue.Queue()
    fd = sys.stdin.fileno()
    if os.isatty(fd):
        old_attrs = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old_attrs)

    def _loop():
        while True:
            chunk = os.read(fd, 1024)
            if not chunk:
                return
            for ch in chunk.decode("utf-8", "replace"):
                keys.put(ch.lower())

    threading.Thread(target=_loop, daemon=True, name="replay-keys").start()
    return keys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="LeRobot dataset dir (contains meta/ and data/)")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--blend-seconds", type=float, default=2.0,
                   help="blend from the current pose to the first recorded frame")
    p.add_argument("--pose-port", type=int, default=5556, help="deploy --zmq-port")
    p.add_argument("--state-host", default="localhost", help="deploy state stream host")
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--no-neck", action="store_true", help="do not publish the neck command")
    args = p.parse_args()

    fps, tokens, left_hand, right_hand, neck = load_episode(
        Path(args.dataset), args.episode, with_neck=not args.no_neck
    )
    standing_token = read_standing_token(args.state_host, args.state_port)
    if standing_token.shape != tokens.shape[1:]:
        raise SystemExit(
            f"[Replay] token size mismatch: deploy {standing_token.shape} vs recorded {tokens.shape[1:]}"
        )

    context = zmq.Context()
    pose_socket = context.socket(zmq.PUB)
    pose_socket.bind(f"tcp://*:{args.pose_port}")
    neck_socket = None
    if neck is not None:
        neck_socket = context.socket(zmq.PUB)
        neck_socket.setsockopt(zmq.SNDHWM, 1)
        neck_socket.bind(f"tcp://*:{NECK_PORT}")
    time.sleep(0.1)

    # Current output; starts as "stand still": standing token, hands open, neck centered.
    cur_token = standing_token.copy()
    cur_left = np.zeros(7, dtype=np.float32)
    cur_right = np.zeros(7, dtype=np.float32)
    cur_neck = np.zeros(2, dtype=np.float32)

    frame_index = 0

    def send(token, left, right, neck_cmd):
        nonlocal frame_index
        msg = pack_pose_message(
            {
                "token_state": token.reshape(1, -1).astype(np.float32),
                "frame_index": np.array([frame_index], dtype=np.int64),
                "left_hand_joints": left.reshape(1, 7).astype(np.float32),
                "right_hand_joints": right.reshape(1, 7).astype(np.float32),
            },
            topic="pose",
            version=4,
        )
        pose_socket.send(msg)
        if neck_socket is not None:
            neck_socket.send(json.dumps([float(neck_cmd[0]), float(neck_cmd[1])]).encode("utf-8"))
        frame_index += 1

    # A playback is a list of per-tick frames: blend-in followed by the episode.
    playback = []
    n_blend = max(1, round(args.blend_seconds * fps))
    n_frames = len(tokens)

    def build_playback():
        out = []
        neck0 = neck[0] if neck is not None else cur_neck
        for i in range(1, n_blend + 1):
            a = i / n_blend
            out.append((
                (1 - a) * cur_token + a * tokens[0],
                (1 - a) * cur_left + a * left_hand[0],
                (1 - a) * cur_right + a * right_hand[0],
                (1 - a) * cur_neck + a * neck0,
            ))
        for t in range(n_frames):
            out.append((tokens[t], left_hand[t], right_hand[t], neck[t] if neck is not None else cur_neck))
        return out

    keys = start_key_listener()
    print(
        f"[Replay] holding standing token on port {args.pose_port}"
        f"{'' if neck_socket is None else f', neck on {NECK_PORT}'}.\n"
        "[Replay] Deploy terminal: Enter enables ZMQ streaming.\n"
        "[Replay] This terminal: 'p' play episode, 'q' quit."
    )

    period = 1.0 / fps
    next_tick = time.monotonic()
    try:
        while True:
            while not keys.empty():
                k = keys.get_nowait()
                if k == "q":
                    return
                if k == "p":
                    playback = build_playback()
                    print(f"[Replay] playing: {n_blend} blend frames + {n_frames} episode frames")

            if playback:
                cur_token, cur_left, cur_right, cur_neck = playback.pop(0)
                if not playback:
                    print("[Replay] episode done, holding last frame ('p' to replay)")
            send(cur_token, cur_left, cur_right, cur_neck)

            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()
    finally:
        pose_socket.close(linger=0)
        if neck_socket is not None:
            neck_socket.close(linger=0)
        context.term()
        print("[Replay] exit")


if __name__ == "__main__":
    main()
