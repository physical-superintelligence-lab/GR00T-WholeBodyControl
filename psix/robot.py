"""Robot-side I/O for the psix client: ZED camera, deploy state, token/neck publishers,
the SONIC body-token encoder, and the VLA /info contract."""

import json
import os
import threading
import time
from base64 import b64decode, b64encode

import cv2
import msgpack
import numpy as np
import requests
import zmq
from numpy.lib.format import descr_to_dtype, dtype_to_descr

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, pack_pose_message

GROOT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CTRL_DT = 1.0 / 30.0
HAND_DIM, TOKEN_DIM, NECK_DIM = 14, 64, 2
# state [hand 14 | arm 14 | leg 15 (| neck 2)], action [hand 14 | token 64 (| neck 2)]
WIRE_DIMS = {True: (45, 80), False: (43, 78)}
GOAL_KEY, EGO_KEY = "subgoal.egocentric", "video.egocentric"

# body-token column the checkpoint was trained on -> SONIC encoder that produces it
TOKEN_ENCODERS = {
    "action.body_token_v1_1": ("v1_1", "gear_sonic_deploy/policy/sonic_v1_1/model_encoder.onnx"),
    "action.body_token": ("v1", "gear_sonic_deploy/policy/release/model_encoder.onnx"),
}


# ---------------------------------------------------------------- wire helpers

def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {"__numpy__": b64encode(data).decode(), "dtype": dtype_to_descr(o.dtype), "shape": o.shape}
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        arr = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return arr.reshape(dct["shape"]) if dct["shape"] else arr[0]
    return dct


def convert_numpy(data, func):
    if isinstance(data, dict):
        return func(data) if "__numpy__" in data else {k: convert_numpy(v, func) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_numpy(v, func) for v in data]
    if isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    return data


def encode_jpeg_b64(rgb, quality):
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return b64encode(encoded.tobytes()).decode("ascii")


def decode_jpeg_b64(value):
    bgr = cv2.imdecode(np.frombuffer(b64decode(value, validate=True), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def get_json(url, timeout=5.0):
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()


def read_vla_info(host, port, neck):
    """Check the served checkpoint against the robot's wire layout (with / without the
    neck); returns (info, encoder version, encoder path, goal (H, W))."""
    info = get_json(f"http://{host}:{port}/info")
    wire = info["wire"]
    if (wire["state_dim"], wire["action_dim"]) != WIRE_DIMS[neck]:
        raise RuntimeError(f"VLA wire dims {wire['state_dim']}/{wire['action_dim']}, robot "
                           f"{'with' if neck else 'without'} neck needs {WIRE_DIMS[neck][0]}/{WIRE_DIMS[neck][1]}")
    if info.get("goal_key") != GOAL_KEY:
        raise RuntimeError(f"VLA reads its goal under {info.get('goal_key')!r}, client sends {GOAL_KEY!r}")
    token_keys = [k for k in wire["action_keys"] if k in TOKEN_ENCODERS]
    if len(token_keys) != 1:
        raise RuntimeError(f"cannot pick a SONIC encoder from action keys {wire['action_keys']}")
    version, path = TOKEN_ENCODERS[token_keys[0]]
    resize = next((t["size"] for t in info.get("transforms", []) if t.get("name") == "resize"), (270, 480))
    return info, version, os.path.join(GROOT_ROOT, path), tuple(resize)


def resize_goal(rgb, hw):
    h, w = hw
    if rgb.shape[:2] == (h, w):
        return rgb
    return np.ascontiguousarray(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA))


# ---------------------------------------------------------------- body-token encoder

_MUJOCO_TO_ISAACLAB = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24,
                                18, 25, 19, 26, 20, 27, 21, 28])


def _quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class BodyTokenEncoder:
    """SONIC encoder (g1 mode) on a frozen pose: the current joints and base quat held
    for all 10 frames with zero velocity. Gives the first-frame RTC prefix token and
    the hold token for the measured pose."""

    # obs dim, offset of the 10-frame anchor orientation, heading-normalized anchor (v1_1)
    LAYOUTS = {"v1": (1762, 601, False), "v1_1": (1751, 584, True)}

    def __init__(self, path, version):
        import onnxruntime as ort
        self.obs_dim, self.anchor_offset, self.heading = self.LAYOUTS[version]
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        model_dim = self.session.get_inputs()[0].shape[-1]
        if isinstance(model_dim, int) and model_dim != self.obs_dim:
            raise ValueError(f"{path} takes {model_dim}-D obs, encoder {version} builds {self.obs_dim}")
        print(f"[encoder] {version}: {path}", flush=True)

    def _anchor_6d(self, quat):
        q = np.asarray(quat, np.float64)
        q = q / np.linalg.norm(q)
        if self.heading:  # relative to the yaw-only heading of the same quat
            r = _quat_matrix(q)
            h = np.arctan2(r[1, 0], r[0, 0])
            ref = np.array([np.cos(h / 2), 0.0, 0.0, -np.sin(h / 2)])  # inverse heading
            w1, x1, y1, z1 = ref
            w2, x2, y2, z2 = q
            q = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
            r = _quat_matrix(q)
        else:  # relative to itself: identity
            r = np.eye(3)
        return r[:, :2].reshape(-1).astype(np.float32)

    def encode(self, state):
        qpos = np.asarray(state["body_q_measured"], np.float32).reshape(29)[_MUJOCO_TO_ISAACLAB]
        quat = np.asarray(state["base_quat_measured"], np.float32).reshape(4)  # wxyz
        obs = np.zeros(self.obs_dim, np.float32)
        obs[4:294] = np.tile(qpos, 10)  # obs[0] = mode 0 (g1); velocities stay zero
        obs[self.anchor_offset:self.anchor_offset + 60] = np.tile(self._anchor_6d(quat), 10)
        token = np.asarray(self.session.run(None, {self.input_name: obs[None]})[0], np.float32).reshape(-1)
        if token.shape != (TOKEN_DIM,) or not np.isfinite(token).all():
            raise RuntimeError("encoder returned an invalid token")
        return token


# ---------------------------------------------------------------- ZMQ endpoints

class _Freshness:
    def __init__(self):
        self._at = None

    def mark(self):
        self._at = time.monotonic()

    def age(self):
        return float("inf") if self._at is None else time.monotonic() - self._at


class Camera(_Freshness):
    """REQ for one ego frame (5558); part 0 of the reply is the RGB JPEG. Served by
    zed_server.py (neck ZED, 672x376) or the onboard RealSense server (640x480)."""

    def __init__(self, address, timeout_ms=1000):
        super().__init__()
        self.address, self.timeout_ms = address, timeout_ms
        self.context = zmq.Context()
        self._connect()

    def _connect(self):
        self.socket = self.context.socket(zmq.REQ)
        for opt in (zmq.SNDTIMEO, zmq.RCVTIMEO):
            self.socket.setsockopt(opt, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.address)

    def get_frame(self):
        try:
            self.socket.send(b"get_frame")
            parts = self.socket.recv_multipart()
        except zmq.ZMQError as exc:
            # a timed-out REQ socket cannot send again before a reply: rebuild it
            self.socket.close(linger=0)
            self._connect()
            raise TimeoutError(f"camera request to {self.address} failed: {exc}") from exc
        bgr = cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR) if parts and parts[0] else None
        if bgr is None:
            raise ValueError("camera returned no decodable ego frame")
        self.mark()
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def stop(self):
        self.socket.close(linger=0)
        self.context.term()


class RobotStateSubscriber(_Freshness):
    """SUB to the deploy's msgpack state stream (5557, topic g1_debug)."""

    def __init__(self, host, port=5557, topic="g1_debug"):
        super().__init__()
        self.topic = topic.encode()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        self.socket.setsockopt(zmq.RCVTIMEO, 100)
        self._lock = threading.Lock()
        self._state = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="state-sub", daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                msg = self.socket.recv()
            except zmq.Again:
                continue
            state = msgpack.unpackb(msg[len(self.topic):] if msg.startswith(self.topic) else msg, raw=False)
            with self._lock:
                self._state = state
                self.mark()

    def get(self):
        with self._lock:
            return self._state

    def stop(self):
        self._running = False
        self._thread.join(timeout=0.5)
        self.socket.close(linger=0)
        self.context.term()


class TokenPublisher:
    """PUB (5556) of Protocol v4 pose messages to the SONIC deploy."""

    def __init__(self, port=5556, topic="pose"):
        self.topic = topic
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")

    def command(self, start=False, stop=False):
        self.socket.send(build_command_message(start=start, stop=stop, planner=True))
        print(f"[wbc] command start={start} stop={stop}", flush=True)

    def publish(self, token, hands):
        pose = {"token_state": token[None].astype(np.float32),
                "left_hand_joints": hands[None, :7].astype(np.float32),
                "right_hand_joints": hands[None, 7:].astype(np.float32)}
        self.socket.send(pack_pose_message(pose, topic=self.topic, version=4))

    def stop(self):
        self.socket.close(linger=0)
        self.context.term()


class NeckStateReader(_Freshness):
    """SUB to zed_server.py's measured neck stream (5560, JSON [yaw, pitch] rad)."""

    def __init__(self, address):
        super().__init__()
        self.socket = zmq.Context.instance().socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(address)
        self._latest = None

    def get(self):
        try:
            msg = json.loads(self.socket.recv(flags=zmq.NOBLOCK))
            self._latest = np.asarray(msg[:2], np.float32)
            self.mark()
        except zmq.Again:
            pass
        return self._latest

    def stop(self):
        self.socket.close(linger=0)


class NeckPublisher:
    """PUB (5570) of the neck command [yaw, pitch] rad that zed_server.py drives."""

    def __init__(self, port=5570):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")

    def publish(self, yaw, pitch):
        self.socket.send(json.dumps([float(yaw), float(pitch)]).encode())

    def stop(self):
        self.socket.close(linger=0)
        self.context.term()


def home_neck(publisher, reader, hold_s=3.0, tol=0.02, stable_s=0.4):
    """Command [0, 0] at 100 Hz until the readback stays within tol for stable_s."""
    t0, in_band_since, latest = time.monotonic(), None, None
    while time.monotonic() - t0 < hold_s:
        publisher.publish(0.0, 0.0)
        latest = reader.get()
        if latest is not None and np.all(np.abs(latest) <= tol):
            in_band_since = in_band_since or time.monotonic()
            if time.monotonic() - in_band_since >= stable_s:
                return True, latest
        else:
            in_band_since = None
        time.sleep(0.01)
    return False, latest
