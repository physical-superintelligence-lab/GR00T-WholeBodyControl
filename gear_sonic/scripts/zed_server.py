"""
Unified camera + neck-motor server (robot-side):
  - ZED egocentric camera  (left RGB + stereo L|R)
  - 2-DOF neck (Dynamixel IDs 0=yaw, 1=pitch on /dev/ttyUSB0), driven by the
    `[yaw, pitch]` stream from pico_manager_thread_server.py.

Camera PUB bind: tcp://*:5555, in the composed_camera format consumed by
gear_sonic/scripts/run_data_exporter.py (msgpack, one message per frame):

  {"timestamps": {"ego_view": <capture time>}, "images": {"ego_view": <ZED left JPEG>}}

Camera REP bind: tcp://*:5558, for inference clients (ZedNeckCamera). Each
request gets the next new frame as a 2-part reply [ego_rgb_jpeg, ego_stereo_jpeg].
Serve a single REQ client: several clients on one REP socket split the frames.

Optional outputs:
  - H.264 TCP stream of ZED stereo to Pico VR (--enable-pico).
  - ZMQ PUB on tcp://*:5560 of neck motor present-position
    `[yaw_rad, pitch_rad]`, recorded by run_data_exporter.py --record-neck.
    Toggle via --neck-state-pub (default on, '' to disable).

Usage (full pipeline, on the robot):
    python gear_sonic/scripts/zed_server.py \
        --enable-pico --pico-ip <PICO_IP> \
        --enable-neck-motor \
        --pose-zmq tcp://<DESKTOP_IP>:5570

Environment overrides: CAMERA_PUB, ZMQ_BIND, ENABLE_PICO, PICO_IP, VIDEO_PORT,
                      ZED_RESOLUTION, ZED_FPS, VR_WIDTH, VR_HEIGHT,
                      POSE_ZMQ, NECK_MOTOR, NECK_STATE_PUB
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import signal
import socket
import struct
import threading
import time
from typing import Any, Optional

import cv2
import msgpack
import numpy as np
import pyzed.sl as sl
import zmq

gi = None
Gst = None

# ──────────────────────────────────────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────────────────────────────────────

latest_ego_rgb_bytes: Optional[bytes] = None
latest_ego_stereo_bytes: Optional[bytes] = None
latest_ego_stereo_frame: Optional[np.ndarray] = None
latest_ego_timestamp = 0.0
frame_seq = 0
frame_cond = threading.Condition()

# Defaults (overridden by CLI / env)
CAMERA_PUB_DEFAULT = os.environ.get("CAMERA_PUB", "tcp://*:5555")
ZMQ_BIND_DEFAULT = os.environ.get("ZMQ_BIND", "tcp://*:5558")
PICO_IP = os.environ.get("PICO_IP", "192.168.0.128")
VIDEO_PORT = int(os.environ.get("VIDEO_PORT", "12345"))
VR_WIDTH = int(os.environ.get("VR_WIDTH", "0"))   # 0 = use native stereo size
VR_HEIGHT = int(os.environ.get("VR_HEIGHT", "0"))
FPS = 30

RESOLUTION_MAP = {
    "auto": sl.RESOLUTION.AUTO,
    "vga": sl.RESOLUTION.VGA,
    "hd720": sl.RESOLUTION.HD720,
    "hd1080": sl.RESOLUTION.HD1080,
    "hd1200": sl.RESOLUTION.HD1200,
    "hd2k": sl.RESOLUTION.HD2K,
}

# Neck-motor configuration — edit these in place to recalibrate.
NECK_PORT             = "/dev/ttyUSB0"
NECK_BAUD             = 2_000_000
NECK_YAW_ID           = 0
NECK_PITCH_ID         = 1
NECK_YAW_ZERO_TICK    = 1897
# NECK_PITCH_ZERO_TICK  = 2900
NECK_PITCH_ZERO_TICK  = 3275
NECK_YAW_LIMIT_DEG    = 60.0
NECK_PITCH_LIMIT_DEG  = 45.0
NECK_SMOOTH_ALPHA     = 0.3    # 0 = frozen, 1 = no smoothing
NECK_CONTROL_HZ       = 50
# Sign multipliers applied to the SMPL-X-derived (yaw, pitch). Flip if the
# motor moves opposite to the head.
NECK_YAW_SIGN         = 1
NECK_PITCH_SIGN       = -1

ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_POSITION = 132
ADDR_HW_ERROR_STATUS  = 70
COMM_SUCCESS          = 0
# Hardware Error Status (addr 70) bits, Dynamixel X-series.
NECK_HW_ERROR_BITS    = {0: "input voltage", 2: "overheating", 3: "motor encoder",
                         4: "electrical shock", 5: "overload"}
NECK_LOG_INTERVAL_S   = 2.0    # min seconds between repeats of the same neck error log
NECK_STATS_INTERVAL_S = 5.0
TICKS_PER_REV         = 4096


def _bgra_to_bgr(bgra: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bgra[:, :, :3])


def _jpeg_encode_bgr(bgr: np.ndarray, quality: int) -> bytes:
    q = max(1, min(100, int(quality)))
    try:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            return buf.tobytes()
    except (cv2.error, TypeError, ValueError):
        pass
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "JPEG encode failed in OpenCV and Pillow is not installed"
        ) from e
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    im = Image.fromarray(rgb, mode="RGB")
    bio = io.BytesIO()
    im.save(bio, format="JPEG", quality=q)
    return bio.getvalue()


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


# ──────────────────────────────────────────────────────────────────────────────
# ZED egocentric capture
# ──────────────────────────────────────────────────────────────────────────────


def zed_capture_thread(cfg: Any) -> None:
    global latest_ego_rgb_bytes, latest_ego_stereo_bytes
    global latest_ego_stereo_frame, latest_ego_timestamp, frame_seq

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = cfg.resolution
    init.camera_fps = cfg.fps
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.sdk_verbose = cfg.sdk_verbose
    # Make grab() reject torn/corrupted frames (USB payload loss) instead of
    # handing them to the encoder. Needs ZED SDK >= 4.1.
    if hasattr(init, "enable_image_validity_check"):
        init.enable_image_validity_check = True
    else:
        print("[ZED] SDK has no enable_image_validity_check; torn frames pass through.")

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(
            f"\033[91m[ZED][ERROR] Camera open failed: {repr(err)}. "
            "No ego frames will be published.\033[0m"
        )
        return

    left_mat = sl.Mat()
    right_mat = sl.Mat()
    runtime = sl.RuntimeParameters()

    print(f"[ZED] Started: resolution={cfg.resolution_name} fps={cfg.fps}")

    fps_t0 = time.time()
    fps_count = 0
    grab_fails: dict = {}

    while True:
        try:
            now = time.time()
            if now - fps_t0 >= 5.0:
                if fps_count == 0:
                    print(
                        f"\033[91m[ZED][ERROR] No frames captured in the last 5s "
                        f"(grab errors: {grab_fails or 'none'})\033[0m"
                    )
                else:
                    print(f"[ZED] capture {fps_count / (now - fps_t0):.1f} fps")
                    if grab_fails:
                        print(f"\033[93m[ZED] rejected frames (last 5s): {grab_fails}\033[0m")
                grab_fails = {}
                fps_t0 = now
                fps_count = 0

            grab_err = zed.grab(runtime)
            if grab_err != sl.ERROR_CODE.SUCCESS:
                key = repr(grab_err)
                grab_fails[key] = grab_fails.get(key, 0) + 1
                continue

            capture_time = time.time()
            zed.retrieve_image(left_mat, sl.VIEW.LEFT, sl.MEM.CPU)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT, sl.MEM.CPU)

            left_rgba = np.ascontiguousarray(
                np.asarray(left_mat.get_data(), dtype=np.uint8)
            )
            right_rgba = np.ascontiguousarray(
                np.asarray(right_mat.get_data(), dtype=np.uint8)
            )
            if left_rgba.ndim != 3 or right_rgba.ndim != 3:
                continue
            if left_rgba.shape[2] != 4 or right_rgba.shape[2] != 4:
                continue

            left_bgr = _bgra_to_bgr(left_rgba)
            right_bgr = _bgra_to_bgr(right_rgba)
            stereo_bgr = np.ascontiguousarray(np.hstack((left_bgr, right_bgr)))

            enc_rgb = _jpeg_encode_bgr(left_bgr, cfg.jpeg_rgb_quality)
            enc_stereo = _jpeg_encode_bgr(stereo_bgr, cfg.jpeg_stereo_quality)

            with frame_cond:
                latest_ego_rgb_bytes = enc_rgb
                latest_ego_stereo_bytes = enc_stereo
                latest_ego_stereo_frame = stereo_bgr.copy()
                latest_ego_timestamp = capture_time
                frame_seq += 1
                frame_cond.notify_all()

            fps_count += 1

        except Exception as e:
            print(f"[ZED] Capture error: {e}")
            time.sleep(0.01)


def camera_rep_thread(cfg: Any) -> None:
    """Answers inference-client frame requests on the REP socket. Each request
    waits for a frame newer than the last one sent, so a single client gets
    every frame; the other camera outputs track frames independently."""
    if not getattr(cfg, "zmq_bind", ""):
        return
    try:
        sock = zmq.Context.instance().socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(cfg.zmq_bind)
        print(f"[ZMQ] Camera REP bound to {cfg.zmq_bind}")
    except Exception as e:
        print(f"\033[91m[ZMQ] Camera REP bind failed: {e}\033[0m")
        return

    last_seq = 0
    while True:
        sock.recv()
        with frame_cond:
            while frame_seq == last_seq:
                frame_cond.wait(timeout=0.1)
            enc_rgb = latest_ego_rgb_bytes
            enc_stereo = latest_ego_stereo_bytes
            last_seq = frame_seq
        sock.send_multipart([enc_rgb or b"", enc_stereo or b""])


def viewer_pub_thread(cfg: Any) -> None:
    """Sends frames to the live viewer on a separate thread so the PUB work
    never rides the capture loop's critical path (off it, an attached viewer
    can't slow frame production / recording)."""
    if not getattr(cfg, "viewer_pub", ""):
        return
    try:
        sock = zmq.Context.instance().socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, 2)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(cfg.viewer_pub)
        print(f"[ZED] Viewer PUB bound: {cfg.viewer_pub}")
    except Exception as e:
        print(f"\033[91m[ZED] Viewer PUB bind failed: {e}\033[0m")
        return

    last_seq = 0
    while True:
        with frame_cond:
            while frame_seq == last_seq:
                frame_cond.wait(timeout=0.1)
            enc_rgb = latest_ego_rgb_bytes
            enc_stereo = latest_ego_stereo_bytes
            last_seq = frame_seq
        if enc_rgb is None:
            continue
        try:
            sock.send_multipart(
                [enc_rgb, enc_stereo if enc_stereo is not None else b""],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Pico Video Streamer (GStreamer H.264 + TCP) — streams ZED stereo
# ──────────────────────────────────────────────────────────────────────────────


class PicoVideoStreamer:
    """Streams ego stereo BGR frames to Pico via GStreamer x264.

    3 threads:
      _push_loop        — waits for fresh ZED stereo frames, feeds appsrc
      _on_encoded_frame — GStreamer callback, queues H.264 NALUs (never blocks)
      _send_loop        — pops from queue, does TCP sendall (isolated from pipeline)
      _connection_loop  — reconnects TCP on failure
    """

    def __init__(
        self,
        pico_ip: str = PICO_IP,
        port: int = VIDEO_PORT,
        width: int = VR_WIDTH,
        height: int = VR_HEIGHT,
        fps: int = FPS,
    ):
        self.pico_ip = pico_ip
        self.port = port
        self.vr_width = width
        self.vr_height = height
        self.fps = fps
        self._running = False
        self._connected = False
        self._sock: Optional[socket.socket] = None
        self._sock_lock = threading.Lock()
        self._pipeline = None
        self._appsrc = None
        self._frame_id = 0
        self._send_q: collections.deque = collections.deque(maxlen=3)
        self._send_event = threading.Event()

    def start(self) -> None:
        global Gst
        if Gst is None:
            raise RuntimeError("GStreamer (gi) not loaded; cannot start Pico streamer")
        # PyGObject >=3.50 rejects None here; pass an empty argv list.
        Gst.init([])
        self._running = True
        self._pipeline_started = False
        threading.Thread(target=self._connection_loop, daemon=True).start()
        threading.Thread(target=self._push_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()

    def _connection_loop(self) -> None:
        while self._running:
            if not self._connected:
                old = self._sock
                self._sock = None
                if old:
                    try:
                        old.close()
                    except Exception:
                        pass
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    s.settimeout(2)
                    s.connect((self.pico_ip, self.port))
                    s.settimeout(None)
                    with self._sock_lock:
                        self._sock = s
                        self._connected = True
                    print(
                        f"\033[92m[PicoStreamer] Connected to Pico "
                        f"{self.pico_ip}:{self.port}\033[0m"
                    )
                except Exception:
                    pass
            time.sleep(2)

    def _start_pipeline(self, width: int, height: int) -> None:
        self.vr_width = width
        self.vr_height = height
        pipe_str = (
            f"appsrc name=src is-live=True format=time "
            f"do-timestamp=True ! "
            f"video/x-raw,format=BGR,"
            f"width={width},height={height},"
            f"framerate={self.fps}/1 ! "
            f"videoconvert ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast "
            f"bitrate=4000 vbv-buf-capacity=500 "
            f"key-int-max=3 threads=2 ! "
            f"video/x-h264,profile=baseline ! "
            f"h264parse config-interval=-1 ! "
            f"video/x-h264,stream-format=byte-stream,alignment=au ! "
            f"appsink name=sink emit-signals=True sync=False"
        )
        self._pipeline = Gst.parse_launch(pipe_str)
        self._appsrc = self._pipeline.get_by_name("src")
        appsink = self._pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_encoded_frame)
        self._pipeline.set_state(Gst.State.PLAYING)
        print(
            f"[PicoStreamer] GStreamer pipeline: {width}x{height}@{self.fps} "
            f"bitrate=4000k preset=ultrafast keyint=3"
        )

    def _on_encoded_frame(self, sink) -> Any:
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            data = bytes(info.data)
            buf.unmap(info)
            self._send_q.append(data)
            self._send_event.set()
        return Gst.FlowReturn.OK

    def _send_loop(self) -> None:
        while self._running:
            self._send_event.wait(timeout=0.1)
            self._send_event.clear()
            while self._send_q:
                data = self._send_q.popleft()
                with self._sock_lock:
                    if not self._connected or self._sock is None:
                        continue
                    try:
                        header = struct.pack(">I", len(data))
                        self._sock.sendall(header + data)
                    except Exception:
                        self._connected = False
                        print("\033[91m[PicoStreamer] Connection lost. Retrying...\033[0m")

    def _push_loop(self) -> None:
        last_seq = 0
        while self._running:
            frame = None
            with frame_cond:
                while frame_seq == last_seq and self._running:
                    frame_cond.wait(timeout=0.1)
                if latest_ego_stereo_frame is not None:
                    frame = latest_ego_stereo_frame
                    last_seq = frame_seq

            if frame is None:
                continue

            h, w = frame.shape[:2]

            if not self._pipeline_started:
                out_w = self.vr_width if self.vr_width > 0 else w
                out_h = self.vr_height if self.vr_height > 0 else h
                self._start_pipeline(out_w, out_h)
                self._pipeline_started = True

            if w != self.vr_width or h != self.vr_height:
                frame = cv2.resize(frame, (self.vr_width, self.vr_height))

            stereo = np.ascontiguousarray(frame)
            gst_buf = Gst.Buffer.new_wrapped(stereo.tobytes())
            gst_buf.pts = self._frame_id * (Gst.SECOND // self.fps)
            gst_buf.duration = Gst.SECOND // self.fps
            self._appsrc.emit("push-buffer", gst_buf)
            self._frame_id += 1

    def stop(self) -> None:
        self._running = False
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)


# ──────────────────────────────────────────────────────────────────────────────
# Neck motor driver (Pico head pose → 2x Dynamixel)
# ──────────────────────────────────────────────────────────────────────────────


class NeckMotor:
    """Subscribes to SMPL-X-derived neck angles over ZMQ and drives 2 Dynamixels.

    The publisher (pico_manager_thread_server.py on the desktop) extracts neck yaw/pitch
    from SMPL-X body data via `human_head_to_robot_neck`, which decouples
    head rotation from torso lean. This class just consumes the resulting
    `[neck_yaw, neck_pitch]` JSON message and writes the motors.

    Thread model:
      _loop — CONTROL_HZ: read [yaw, pitch] from ZMQ SUB → sign/clamp →
              smooth → rad_to_tick → Dynamixel write. Periodic HW-error
              check + auto-reboot.

    Shutdown via stop(): zero-pose write, sleep 0.5s, torque off, close port.
    """

    def __init__(
        self,
        port: str = NECK_PORT,
        baud: int = NECK_BAUD,
        yaw_id: int = NECK_YAW_ID,
        pitch_id: int = NECK_PITCH_ID,
        yaw_zero_tick: int = NECK_YAW_ZERO_TICK,
        pitch_zero_tick: int = NECK_PITCH_ZERO_TICK,
        yaw_limit_rad: float = math.radians(NECK_YAW_LIMIT_DEG),
        pitch_limit_rad: float = math.radians(NECK_PITCH_LIMIT_DEG),
        smooth_alpha: float = NECK_SMOOTH_ALPHA,
        control_hz: int = NECK_CONTROL_HZ,
        yaw_sign: int = NECK_YAW_SIGN,
        pitch_sign: int = NECK_PITCH_SIGN,
        pose_zmq_addr: str = "",
        state_pub_addr: str = "",
    ):
        self.port_path = port
        self.baud = baud
        self.yaw_id = yaw_id
        self.pitch_id = pitch_id
        self.yaw_zero_tick = yaw_zero_tick
        self.pitch_zero_tick = pitch_zero_tick
        self.yaw_limit_rad = yaw_limit_rad
        self.pitch_limit_rad = pitch_limit_rad
        self.smooth_alpha = smooth_alpha
        self.control_hz = control_hz
        self.yaw_sign = yaw_sign
        self.pitch_sign = pitch_sign
        self.pose_zmq_addr = pose_zmq_addr
        self.state_pub_addr = state_pub_addr

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._port = None
        self._packet = None
        self._pose_sub = None
        self._state_pub = None
        self._zmq_ctx = None
        self._last_log: dict = {}
        self._reboot_count = {yaw_id: 0, pitch_id: 0}

    def _log(self, key: str, msg: str, color: str = "91") -> None:
        """Print msg (red by default) at most once per NECK_LOG_INTERVAL_S per key."""
        now = time.time()
        if now - self._last_log.get(key, 0.0) >= NECK_LOG_INTERVAL_S:
            self._last_log[key] = now
            print(f"\033[{color}m{msg}\033[0m")

    @staticmethod
    def _decode_hw_error(hw: int) -> str:
        names = [name for bit, name in NECK_HW_ERROR_BITS.items() if hw & (1 << bit)]
        return ", ".join(names) if names else "none"

    @staticmethod
    def _rad_to_tick(rad: float, zero_tick: int) -> int:
        return zero_tick + int(rad * TICKS_PER_REV / (2 * math.pi))

    @staticmethod
    def _tick_to_rad(tick: int, zero_tick: int) -> float:
        return (tick - zero_tick) * (2 * math.pi) / TICKS_PER_REV

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _check_and_reboot(self, motor_id: int) -> bool:
        hw, res, err = self._packet.read1ByteTxRx(
            self._port, motor_id, ADDR_HW_ERROR_STATUS
        )
        if res != COMM_SUCCESS:
            self._log(
                f"hwcheck{motor_id}",
                f"[Neck][ERROR] ID {motor_id} hardware-error check failed: "
                f"{self._packet.getTxRxResult(res)}",
            )
            return False
        if err == 128 or hw != 0:
            self._reboot_count[motor_id] = self._reboot_count.get(motor_id, 0) + 1
            print(
                f"\033[91m[Neck][ERROR] ID {motor_id} hw_error=0b{hw:08b} "
                f"({self._decode_hw_error(hw)}) -> rebooting "
                f"(reboot #{self._reboot_count[motor_id]})\033[0m"
            )
            reboot_res, _ = self._packet.reboot(self._port, motor_id)
            if reboot_res != COMM_SUCCESS:
                print(
                    f"\033[91m[Neck][ERROR] ID {motor_id} reboot failed: "
                    f"{self._packet.getTxRxResult(reboot_res)}\033[0m"
                )
            time.sleep(0.5)
            torque_res, torque_err = self._packet.write1ByteTxRx(
                self._port, motor_id, ADDR_TORQUE_ENABLE, 1
            )
            if torque_res != COMM_SUCCESS or torque_err != 0:
                print(
                    f"\033[91m[Neck][ERROR] ID {motor_id} torque re-enable after reboot "
                    f"failed: {self._packet.getTxRxResult(torque_res)} "
                    f"{self._packet.getRxPacketError(torque_err)}\033[0m"
                )
            else:
                hw_after, _, _ = self._packet.read1ByteTxRx(
                    self._port, motor_id, ADDR_HW_ERROR_STATUS
                )
                print(
                    f"[Neck] ID {motor_id} rebooted, torque on, "
                    f"hw_error now 0b{hw_after:08b} ({self._decode_hw_error(hw_after)})"
                )
            return True
        return False

    def start(self) -> bool:
        """Open ZMQ SUB + serial port, enable torque, spawn loop thread.

        Requires pose_zmq_addr — pico_manager_thread_server.py (running on the
        desktop) is the source of neck angles.

        Returns True on success, False on failure (caller should continue
        without motor control).
        """
        try:
            from dynamixel_sdk import PacketHandler, PortHandler
        except ImportError as e:
            print(
                f"\033[91m[Neck] Import failed: {e}. "
                f"pip install dynamixel-sdk.\033[0m"
            )
            return False

        if not self.pose_zmq_addr:
            print(
                "\033[91m[Neck] --pose-zmq is required. Run pico_manager_thread_server.py "
                "on the desktop and pass --pose-zmq tcp://<desktop-ip>:5570.\033[0m"
            )
            return False

        self._zmq_ctx = zmq.Context.instance()
        self._pose_sub = self._zmq_ctx.socket(zmq.SUB)
        self._pose_sub.setsockopt(zmq.CONFLATE, 1)
        self._pose_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._pose_sub.setsockopt(zmq.RCVTIMEO, 100)
        try:
            self._pose_sub.connect(self.pose_zmq_addr)
            print(f"[Neck] ZMQ SUB neck-angle source: {self.pose_zmq_addr}")
        except Exception as e:
            print(f"\033[91m[Neck] ZMQ connect failed: {e}\033[0m")
            return False

        self._port = PortHandler(self.port_path)
        self._packet = PacketHandler(2.0)
        if not self._port.openPort() or not self._port.setBaudRate(self.baud):
            print(
                f"\033[91m[Neck] Could not open {self.port_path} @ {self.baud}\033[0m"
            )
            return False

        for i in (self.yaw_id, self.pitch_id):
            _, res, _ = self._packet.ping(self._port, i)
            if res != 0:
                print(f"\033[91m[Neck] Ping ID {i} failed (res={res})\033[0m")
                self._port.closePort()
                return False
            self._check_and_reboot(i)
            self._packet.write1ByteTxRx(
                self._port, i, ADDR_TORQUE_ENABLE, 1
            )

        # Start at zero pose
        self._packet.write4ByteTxRx(
            self._port, self.yaw_id, ADDR_GOAL_POSITION,
            self._rad_to_tick(0.0, self.yaw_zero_tick),
        )
        self._packet.write4ByteTxRx(
            self._port, self.pitch_id, ADDR_GOAL_POSITION,
            self._rad_to_tick(0.0, self.pitch_zero_tick),
        )
        time.sleep(0.3)

        if self.state_pub_addr:
            self._state_pub = self._zmq_ctx.socket(zmq.PUB)
            self._state_pub.setsockopt(zmq.SNDHWM, 1)
            self._state_pub.setsockopt(zmq.LINGER, 0)
            try:
                self._state_pub.bind(self.state_pub_addr)
                print(f"[Neck] State PUB bound: {self.state_pub_addr}")
            except Exception as e:
                print(f"\033[91m[Neck] State PUB bind failed: {e}\033[0m")
                self._state_pub.close(linger=0)
                self._state_pub = None

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        print(
            f"\033[92m[Neck] Started: {self.port_path}@{self.baud} "
            f"IDs {self.yaw_id}/{self.pitch_id} "
            f"limits yaw±{math.degrees(self.yaw_limit_rad):.0f}° "
            f"pitch±{math.degrees(self.pitch_limit_rad):.0f}° "
            f"@ {self.control_hz}Hz\033[0m"
        )
        return True

    def _read_present_state(self):
        """Return (yaw_rad, pitch_rad) read from motors, or (None, None) on error."""
        if self._packet is None or self._port is None:
            return None, None
        try:
            yaw_tick, yres, yerr = self._packet.read4ByteTxRx(
                self._port, self.yaw_id, ADDR_PRESENT_POSITION
            )
            pitch_tick, pres, perr = self._packet.read4ByteTxRx(
                self._port, self.pitch_id, ADDR_PRESENT_POSITION
            )
            ok = True
            for name, motor_id, res, err in (
                ("yaw", self.yaw_id, yres, yerr), ("pitch", self.pitch_id, pres, perr)
            ):
                if res != COMM_SUCCESS:
                    # On a comm failure the SDK returns tick 0 with error 0; never
                    # publish that as a position.
                    ok = False
                    self._log(
                        f"read{motor_id}",
                        f"[Neck][ERROR] {name} (ID {motor_id}) present-position read failed: "
                        f"{self._packet.getTxRxResult(res)}; neck state not published",
                    )
                elif err != 0:
                    ok = False
                    self._log(
                        f"read{motor_id}",
                        f"[Neck][ERROR] {name} (ID {motor_id}) present-position read returned "
                        f"error 0x{err:02x} ({self._packet.getRxPacketError(err)}); "
                        "neck state not published",
                    )
            if not ok:
                return None, None
            return (
                self._tick_to_rad(yaw_tick, self.yaw_zero_tick),
                self._tick_to_rad(pitch_tick, self.pitch_zero_tick),
            )
        except Exception as e:
            self._log("read_exc", f"[Neck][ERROR] present-position read raised: {e!r}")
            return None, None
        
        

    def _read_neck_angles(self):
        """Return (yaw_rad, pitch_rad) from the ZMQ SUB, or None."""
        if self._pose_sub is None:
            return None
        try:
            raw = self._pose_sub.recv(flags=0)
        except zmq.Again:
            return None
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(msg, (list, tuple)) or len(msg) < 2:
            return None
        return float(msg[0]), float(msg[1])

    def _loop(self) -> None:
        dt = 1.0 / self.control_hz
        yaw_cmd = 0.0
        pitch_cmd = 0.0
        last_hw_check = 0.0
        last_status_print = 0.0
        last_warn_print = 0.0
        pose_valid = False
        pose_ever = False
        next_tick = time.time()
        stats_t0 = time.time()
        stats = {"published": 0, "read_fail": 0, "write_fail": 0, "overrun": 0}

        while self._running:
            next_tick += dt
            loop_start = time.time()

            # Read SMPL-X-derived neck angles from the ZMQ publisher.
            yaw_target = yaw_cmd
            pitch_target = pitch_cmd
            pose_ok = False
            try:
                angles = self._read_neck_angles()
                if angles is not None:
                    yaw_target = angles[0] * self.yaw_sign
                    pitch_target = angles[1] * self.pitch_sign
                    pose_ok = True
            except Exception as e:
                now = time.time()
                if now - last_warn_print > 2.0:
                    print(f"[Neck] neck-angle read error: {e}")
                    last_warn_print = now

            if not pose_ok:
                now = time.time()
                if pose_valid or now - last_warn_print > 5.0:
                    print(
                        "[Neck] no neck data yet "
                        "(pico_manager_thread_server.py running? trackers streaming?)"
                    )
                    last_warn_print = now
                pose_valid = False
                # Hold current cmd — don't snap to zero.
                yaw_target = yaw_cmd
                pitch_target = pitch_cmd
            else:
                if not pose_valid:
                    print(
                        f"\033[92m[Neck] neck data {'resumed' if pose_ever else 'received'} "
                        f"from {self.pose_zmq_addr}\033[0m"
                    )
                pose_valid = True
                pose_ever = True

            if abs(yaw_target) > self.yaw_limit_rad:
                self._log(
                    "clamp_yaw",
                    f"[Neck] yaw command {math.degrees(yaw_target):+.1f}° clamped to "
                    f"±{math.degrees(self.yaw_limit_rad):.0f}°",
                    color="93",
                )
            if abs(pitch_target) > self.pitch_limit_rad:
                self._log(
                    "clamp_pitch",
                    f"[Neck] pitch command {math.degrees(pitch_target):+.1f}° clamped to "
                    f"±{math.degrees(self.pitch_limit_rad):.0f}°",
                    color="93",
                )

            # Clamp
            yaw_target = self._clamp(
                yaw_target, -self.yaw_limit_rad, self.yaw_limit_rad
            )
            pitch_target = self._clamp(
                pitch_target, -self.pitch_limit_rad, self.pitch_limit_rad
            )

            # Low-pass smoothing
            yaw_cmd += self.smooth_alpha * (yaw_target - yaw_cmd)
            pitch_cmd += self.smooth_alpha * (pitch_target - pitch_cmd)

            yaw_tick = self._rad_to_tick(yaw_cmd, self.yaw_zero_tick)
            pitch_tick = self._rad_to_tick(pitch_cmd, self.pitch_zero_tick)
            try:
                for name, motor_id, tick in (
                    ("yaw", self.yaw_id, yaw_tick), ("pitch", self.pitch_id, pitch_tick)
                ):
                    res, err = self._packet.write4ByteTxRx(
                        self._port, motor_id, ADDR_GOAL_POSITION, tick
                    )
                    if res != COMM_SUCCESS or err != 0:
                        stats["write_fail"] += 1
                        self._log(
                            f"write{motor_id}",
                            f"[Neck][ERROR] {name} (ID {motor_id}) goal write: "
                            f"{self._packet.getTxRxResult(res)} "
                            f"{self._packet.getRxPacketError(err) if err else ''}",
                        )
            except Exception as e:
                stats["write_fail"] += 1
                self._log("write_exc", f"[Neck][ERROR] goal write raised: {e!r}")

            if self._state_pub is not None:
                yaw_meas, pitch_meas = self._read_present_state()
                if yaw_meas is not None:
                    try:
                        self._state_pub.send_string(
                            json.dumps([yaw_meas, pitch_meas])
                        )
                        stats["published"] += 1
                    except Exception as e:
                        self._log("pub_exc", f"[Neck][ERROR] state publish raised: {e!r}")
                else:
                    stats["read_fail"] += 1

            now = time.time()

            # Periodic hw-error check every 2s
            if now - last_hw_check > 2.0:
                for i in (self.yaw_id, self.pitch_id):
                    try:
                        self._check_and_reboot(i)
                    except Exception as e:
                        print(f"[Neck] hw check error: {e}")
                last_hw_check = now

            # Status line every 0.5s
            if now - last_status_print > 0.5:
                print(
                    f"[Neck] yaw {math.degrees(yaw_cmd):+6.1f}° "
                    f"(tick {yaw_tick:4d})  pitch "
                    f"{math.degrees(pitch_cmd):+6.1f}° (tick {pitch_tick:4d})"
                )
                last_status_print = now

            elapsed = time.time() - loop_start
            if elapsed > 2 * dt:
                stats["overrun"] += 1
                self._log(
                    "overrun",
                    f"[Neck] control loop iteration took {elapsed * 1000:.0f} ms "
                    f"(budget {dt * 1000:.0f} ms)",
                    color="93",
                )

            # Problems summary every NECK_STATS_INTERVAL_S (silent when healthy).
            if now - stats_t0 >= NECK_STATS_INTERVAL_S:
                if self._state_pub is not None and stats["published"] == 0:
                    print(
                        f"\033[91m[Neck][ERROR] no neck state published in the last "
                        f"{now - stats_t0:.0f}s (read failures: {stats['read_fail']})\033[0m"
                    )
                elif stats["read_fail"] or stats["write_fail"] or stats["overrun"]:
                    print(
                        f"\033[93m[Neck] last {now - stats_t0:.0f}s: state published "
                        f"{stats['published']}, read failures {stats['read_fail']}, "
                        f"write failures {stats['write_fail']}, loop overruns "
                        f"{stats['overrun']}\033[0m"
                    )
                stats_t0 = now
                stats = {k: 0 for k in stats}

            # Sleep
            sleep_s = next_tick - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.time()

    def stop(self) -> None:
        if not self._running and self._port is None:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            if self._port is not None and self._packet is not None:
                self._packet.write4ByteTxRx(
                    self._port, self.yaw_id, ADDR_GOAL_POSITION,
                    self._rad_to_tick(0.0, self.yaw_zero_tick),
                )
                self._packet.write4ByteTxRx(
                    self._port, self.pitch_id, ADDR_GOAL_POSITION,
                    self._rad_to_tick(0.0, self.pitch_zero_tick),
                )
                time.sleep(0.5)
                for i in (self.yaw_id, self.pitch_id):
                    self._packet.write1ByteTxRx(
                        self._port, i, ADDR_TORQUE_ENABLE, 0
                    )
                self._port.closePort()
                print("[Neck] Shutdown: zeroed, torque off, port closed.")
        except Exception as e:
            print(f"[Neck] Shutdown error: {e}")
        finally:
            self._port = None
            self._packet = None
            if self._pose_sub is not None:
                try:
                    self._pose_sub.close(linger=0)
                except Exception:
                    pass
                self._pose_sub = None
            if self._state_pub is not None:
                try:
                    self._state_pub.close(linger=0)
                except Exception:
                    pass
                self._state_pub = None


# ──────────────────────────────────────────────────────────────────────────────
# CLI + server entry
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Camera + neck server: ZED ego (ZMQ + optional Pico) and neck motor."
    )

    # ZED
    p.add_argument(
        "--resolution",
        default=os.environ.get("ZED_RESOLUTION", "vga"),
        choices=sorted(RESOLUTION_MAP.keys()),
        help="ZED resolution preset",
    )
    p.add_argument(
        "--fps", type=int,
        default=int(os.environ.get("ZED_FPS", "30")),
        help="ZED camera FPS",
    )
    p.add_argument(
        "--sdk-verbose", type=int,
        default=int(os.environ.get("ZED_SDK_VERBOSE", "0")),
    )
    p.add_argument(
        "--jpeg-rgb-quality", type=int,
        default=int(os.environ.get("ZED_JPEG_RGB_QUALITY", "80")),
    )
    p.add_argument(
        "--jpeg-stereo-quality", type=int,
        default=int(os.environ.get("ZED_JPEG_STEREO_QUALITY", "60")),
    )

    # Neck motor
    p.add_argument(
        "--enable-neck-motor", action="store_true",
        default=_env_bool("NECK_MOTOR", False),
        help=(
            "Drive the 2-DOF neck (Dynamixel IDs 0=yaw, 1=pitch) from "
            "neck angles published by pico_manager_thread_server.py."
        ),
    )
    p.add_argument(
        "--pose-zmq",
        default=os.environ.get("POSE_ZMQ", ""),
        help=(
            "Required when --enable-neck-motor is set. ZMQ SUB address of "
            "pico_manager_thread_server.py, which publishes [neck_yaw, neck_pitch]. "
            "Example: "
            "'tcp://desktop-ip:5570'."
        ),
    )
    p.add_argument(
        "--neck-state-pub",
        default=os.environ.get("NECK_STATE_PUB", "tcp://*:5560"),
        help=(
            "ZMQ PUB bind address for the neck motor present-position stream "
            "[yaw_rad, pitch_rad]. Read from the Dynamixels at the neck "
            "control rate so a recorder can log it into data.json. "
            "Set to empty string to disable. Default 'tcp://*:5560'."
        ),
    )

    # Network
    p.add_argument(
        "--zmq-bind",
        default=ZMQ_BIND_DEFAULT,
        help="ZMQ REP bind address for inference clients (default tcp://*:5558, '' to disable)",
    )
    p.add_argument(
        "--camera-pub",
        default=CAMERA_PUB_DEFAULT,
        help="ZMQ PUB bind address for the data exporter camera stream (default tcp://*:5555)",
    )
    p.add_argument(
        "--viewer-pub",
        default=os.environ.get("VIEWER_PUB", "tcp://0.0.0.0:5559"),
        help=(
            "ZMQ PUB bind address for the live viewer stream (separate from "
            "the recording camera stream so the viewer never slows recording). "
            "Set to empty string to disable. Default 'tcp://0.0.0.0:5559'."
        ),
    )

    # Pico
    p.set_defaults(enable_pico=_env_bool("ENABLE_PICO", False))
    p.add_argument(
        "--enable-pico", action="store_true", dest="enable_pico",
        help="Enable Pico GStreamer TCP stream of ZED stereo",
    )
    p.add_argument(
        "--no-enable-pico", action="store_false", dest="enable_pico",
        help="Disable Pico stream",
    )
    p.add_argument("--pico-ip", default=PICO_IP)
    p.add_argument("--pico-port", type=int, default=VIDEO_PORT)
    return p.parse_args()


def _build_config(ns: argparse.Namespace) -> Any:
    class C:
        pass

    c = C()
    c.resolution = RESOLUTION_MAP[ns.resolution]
    c.resolution_name = ns.resolution
    c.fps = ns.fps
    c.sdk_verbose = ns.sdk_verbose
    c.jpeg_rgb_quality = ns.jpeg_rgb_quality
    c.jpeg_stereo_quality = ns.jpeg_stereo_quality
    c.camera_pub = ns.camera_pub
    c.zmq_bind = ns.zmq_bind
    c.viewer_pub = ns.viewer_pub
    c.enable_pico = ns.enable_pico
    c.pico_ip = ns.pico_ip
    c.pico_port = ns.pico_port
    c.enable_neck_motor = ns.enable_neck_motor
    c.pose_zmq = ns.pose_zmq
    c.neck_state_pub = ns.neck_state_pub
    return c


def start_server(cfg: Any) -> None:
    global Gst, gi

    threading.Thread(
        target=zed_capture_thread, args=(cfg,), daemon=True
    ).start()
    threading.Thread(
        target=viewer_pub_thread, args=(cfg,), daemon=True
    ).start()
    threading.Thread(
        target=camera_rep_thread, args=(cfg,), daemon=True
    ).start()

    pico_streamer: Optional[PicoVideoStreamer] = None
    if cfg.enable_pico:
        try:
            import gi as _gi
            _gi.require_version("Gst", "1.0")
            from gi.repository import Gst as _Gst
            gi = _gi
            Gst = _Gst
        except ImportError as e:
            print(f"ENABLE_PICO set but GStreamer/PyGObject not available: {e}")
            cfg.enable_pico = False

    if cfg.enable_pico and Gst is not None:
        pico_streamer = PicoVideoStreamer(
            pico_ip=cfg.pico_ip, port=cfg.pico_port
        )
        pico_streamer.start()
        print("\033[92m[PicoStreamer] Started ego stereo streaming to Pico\033[0m")

    neck_motor: Optional[NeckMotor] = None
    if cfg.enable_neck_motor:
        neck_motor = NeckMotor(
            pose_zmq_addr=cfg.pose_zmq,
            state_pub_addr=cfg.neck_state_pub,
        )
        if not neck_motor.start():
            print("\033[91m[Neck] Failed to start; continuing without motor control.\033[0m")
            neck_motor = None

    context = zmq.Context()
    sock = context.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 20)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(cfg.camera_pub)
    print(f"[ZMQ] Camera PUB bound to {cfg.camera_pub}")

    def _force_exit(sig, _frame):
        print(f"\n\033[91m[Server] Signal {sig} received – shutting down.\033[0m")
        if neck_motor is not None:
            neck_motor.stop()
        if pico_streamer is not None:
            pico_streamer.stop()
        os._exit(0)

    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)

    last_sent_seq = 0

    try:
        while True:
            with frame_cond:
                while frame_seq == last_sent_seq:
                    frame_cond.wait(timeout=0.1)
                ego_rgb = latest_ego_rgb_bytes
                ego_timestamp = latest_ego_timestamp
                last_sent_seq = frame_seq

            if ego_rgb is None:
                continue
            msg = {
                "timestamps": {"ego_view": ego_timestamp},
                "images": {"ego_view": ego_rgb},
            }
            try:
                sock.send(msgpack.packb(msg, use_bin_type=True), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
    finally:
        sock.close()
        context.term()
        if pico_streamer is not None:
            pico_streamer.stop()
        if neck_motor is not None:
            neck_motor.stop()


def main() -> None:
    ns = _parse_args()
    cfg = _build_config(ns)
    start_server(cfg)


if __name__ == "__main__":
    main()
