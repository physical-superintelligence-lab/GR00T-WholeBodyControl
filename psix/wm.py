"""World-model goal images: a worker that keeps a last-good future goal for the current
HLP step, refreshed every `period` seconds from the latest ego frame."""

import threading
import time
import uuid

import cv2
import numpy as np
import requests

from .recording import log_event
from .robot import decode_jpeg_b64, encode_jpeg_b64

# Stay-put monitor: on a near-static scene, a goal that barely differs from the
# current frame is resampled with a new seed (up to STAYPUT_RETRIES times) after
# STAYPUT_STREAK such proposals in a row, then accepted as a possible completion frame.
GATE_SIZE = (160, 120)
STAYPUT_OBS_MOTION_MAX = 6.5
STAYPUT_GOAL_VS_OBS_MAX = 25.0
STAYPUT_STREAK = 3
STAYPUT_RETRIES = 3


def _gray(rgb):
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), GATE_SIZE).astype(np.float32)


class WmClient:
    """Requests are serialized and tagged with the prompt epoch; a step change (new
    epoch) drops the last-good goal, so an in-flight answer for the old step can never
    become the current goal. The camera frame is pushed in by the VLA loop."""

    def __init__(self, base_url, task, recorder, period=0.8, seconds=2.4, timeout=15.0, jpeg_quality=90,
                 goal_hard_age=30.0):
        self.base_url = base_url.rstrip("/")
        self.recorder = recorder
        self.period, self.seconds, self.timeout = period, seconds, timeout
        self.jpeg_quality, self.goal_hard_age = jpeg_quality, goal_hard_age
        self.session_id = str(uuid.uuid4())
        self._http = requests.Session()
        self._http.trust_env = False
        self._lock = threading.Lock()
        self._wake, self._stop = threading.Event(), threading.Event()
        self._thread = None
        self._task = task
        self._epoch = 0
        self._enabled = True
        self._ego = self._ego_at = None
        self._goal = self._goal_at = None
        self._generation = 0
        self._req_id = 0
        self._error = None
        self._reset_gate_locked()

    def _reset_gate_locked(self):
        self._prev_obs = None
        self._streak = self._rejects = 0
        self._seed = None

    # ------------------------------------------------------------ state

    def update_ego(self, rgb):
        with self._lock:
            first = self._ego is None
            self._ego, self._ego_at = rgb, time.monotonic()
        if first:
            self._wake.set()

    def latest_ego(self):
        with self._lock:
            return self._ego, self._ego_at

    def snapshot(self):
        with self._lock:
            age = None if self._goal_at is None else time.monotonic() - self._goal_at
            return {"epoch": self._epoch, "task": self._task, "goal": self._goal, "goal_age": age,
                    "expired": age is not None and 0 < self.goal_hard_age < age,
                    "generation": self._generation, "error": self._error}

    def set_task(self, text):
        """New WM prompt under a new epoch; the robot is gated until its goal lands."""
        with self._lock:
            self._task, self._enabled = text, True
            self._invalidate_locked()
        print(f"[wm] task -> {text!r}", flush=True)
        self._wake.set()

    def disable(self):
        """Stop requesting and drop the goal (no current step)."""
        with self._lock:
            self._enabled = False
            self._invalidate_locked()

    def _invalidate_locked(self):
        self._epoch += 1
        self._goal = self._goal_at = None
        self._reset_gate_locked()

    # ------------------------------------------------------------ worker

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="wm", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._http.close()

    def _loop(self):
        print(f"[wm] {self.base_url} period={self.period}s horizon={self.seconds}s", flush=True)
        next_due = time.monotonic()
        while not self._stop.is_set():
            self._wake.wait(max(0.0, next_due - time.monotonic()))
            self._wake.clear()
            if self._stop.is_set():
                break
            started = time.monotonic()
            try:
                self._poll_once()
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                print(f"[wm] request failed ({exc}); keeping last-good goal", flush=True)
            age = self.snapshot()["goal_age"]
            if age is not None and age > 3 * self.period:
                print(f"[wm] WARNING: goal is {age:.1f}s old (hard age {self.goal_hard_age}s)", flush=True)
            next_due = started + self.period

    def _poll_once(self):
        with self._lock:
            if self._ego is None or not self._enabled:
                return
            self._req_id += 1
            req_id, epoch, task, ego, seed = self._req_id, self._epoch, self._task, self._ego, self._seed
        body = {"transport": "jpeg", "jpeg": True, "ego_jpeg": encode_jpeg_b64(ego, self.jpeg_quality),
                "task": task, "subtask": task, "req_id": req_id, "robot_episode_session_id": self.session_id,
                "prompt_gen": epoch, "mode": "future", "seconds": self.seconds}
        if seed is not None:
            body["seed"] = seed  # forces a resampled future for the same frame
        record = self.recorder.begin("wm", body, context={"epoch": epoch})
        t0 = time.monotonic()
        response = self._http.post(f"{self.base_url}/wm", json=body, timeout=self.timeout)
        reply = response.json()
        if response.status_code != 200 or "subgoal_jpeg" not in reply or int(reply.get("req_id", -1)) != req_id:
            self.recorder.finish(record, "invalid_response", reply=reply, http_status=response.status_code)
            raise ValueError(f"bad /wm reply (http {response.status_code}): {str(reply)[:200]}")
        goal = decode_jpeg_b64(reply["subgoal_jpeg"])
        wm_ms = (time.monotonic() - t0) * 1000.0
        gray_obs, gray_goal = _gray(ego), _gray(goal)

        with self._lock:
            if epoch != self._epoch:
                decision = "stale"
            else:
                decision = self._gate_locked(gray_obs, gray_goal)
                if decision == "accept":
                    self._goal, self._goal_at = goal, time.monotonic()
                    self._generation += 1
                    self._error = None
            generation, seed = self._generation, self._seed
        self.recorder.finish(record, decision, reply=reply, wm_ms=wm_ms, generation=generation)
        log_event("wm_response", decision=decision, req_id=req_id, epoch=epoch, wm_ms=wm_ms, generation=generation)
        if decision == "stayput_retry":
            print(f"[wm] stay-put goal held, resampling with seed={seed}", flush=True)
            self._wake.set()
        elif decision == "accept":
            print(f"[wm] goal gen={generation} epoch={epoch} {wm_ms:.0f}ms", flush=True)

    def _gate_locked(self, gray_obs, gray_goal):
        goal_vs_obs = float(np.abs(gray_goal - gray_obs).mean())
        obs_motion = None if self._prev_obs is None else float(np.abs(gray_obs - self._prev_obs).mean())
        if obs_motion is not None and obs_motion < STAYPUT_OBS_MOTION_MAX and goal_vs_obs < STAYPUT_GOAL_VS_OBS_MAX:
            self._streak += 1
        else:
            self._streak = self._rejects = 0
        if self._streak >= STAYPUT_STREAK:
            if self._rejects < STAYPUT_RETRIES:
                self._rejects += 1
                self._seed = 1 if self._seed is None else self._seed + 1
                return "stayput_retry"
            self._streak = self._rejects = 0
        self._seed = None
        self._prev_obs = gray_obs
        return "accept"
