"""VLA transports and robot publication.

RTCWebSocketClient streams observations to serve_psix /ws at 30 Hz and publishes every
action row the server pushes back; HttpChunkClient (rtc off) posts to /act, plays the
chunk at 30 Hz and holds the measured pose until the next one lands.

Safety, shared by both:
  * an action executes only if it was computed for the current HLP step: each step
    change bumps the WM epoch, holds the robot and blocks everything in flight until an
    observation for the new step has gone out (RTC: version floor; HTTP: exact epoch);
  * state / camera / neck / last observation older than observation_stale_timeout,
    or no WM goal for the current step, holds the robot;
  * holding = publishing the encoder token of the measured pose with the last hands and
    neck; no fresh action for action_stale_timeout stops the WBC.
"""

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass

import cv2
import numpy as np
import requests
from websocket import WebSocketApp

from .recording import log_event
from .robot import (CTRL_DT, EGO_KEY, GOAL_KEY, HAND_DIM, NECK_DIM, TOKEN_DIM, WIRE_DIMS, convert_numpy,
                    numpy_deserialize, numpy_serialize, resize_goal)


@dataclass
class Observation:
    frame: np.ndarray   # RGB ego
    state: dict         # raw g1_debug message
    states: np.ndarray  # (45,) / (43,) model state
    goal: np.ndarray    # RGB goal at the VLA canvas size
    epoch: int


class VlaClientBase:
    def __init__(self, server_url, task, camera, state_sub, wm, encoder, goal_hw, running, neck=True,
                 neck_reader=None, token_pub=None, neck_pub=None, rollout=None, dry_run=True,
                 obs_stale_timeout=0.5, action_stale_timeout=0.5, show_goal=False):
        self.server_url = server_url
        self.neck = neck
        self.state_dim, self.action_dim = WIRE_DIMS[neck]
        self.instruction = f"Task: {task.strip().lower()}"  # training prompt: task text only, lower case
        self.camera, self.state_sub, self.neck_reader, self.wm = camera, state_sub, neck_reader, wm
        self.encoder, self.goal_hw, self.running = encoder, goal_hw, running
        self.token_pub, self.neck_pub, self.rollout, self.dry_run = token_pub, neck_pub, rollout, dry_run
        self.obs_stale_timeout, self.action_stale_timeout = obs_stale_timeout, action_stale_timeout
        self.show_goal = show_goal and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if show_goal and not self.show_goal:
            print("[goal-window] no DISPLAY; goal window disabled", flush=True)
        self._alive = True
        self._connected = threading.Event()
        # lock order: _send_lock -> _publish_lock -> _action_lock
        self._send_lock = threading.Lock()     # observation send vs step transition
        self._publish_lock = threading.Lock()  # robot publication vs hold / transition
        self._action_lock = threading.Lock()   # version bookkeeping
        self._step = None
        self._instruction_hold = None
        self._execution_started_at = None
        self._wbc_started = False
        self._holding = False
        self._hold_action = None
        self._last_published = None
        self._last_obs_at = None
        self._last_version = -1
        self._action_floor = None  # newest version seen when the first obs of this step went out
        self._last_accepted_at = None
        self._last_log_at = {}
        self._last_goal_generation = None

    def _throttled(self, key, message, period=2.0):
        now = time.monotonic()
        if now - self._last_log_at.get(key, -float("inf")) >= period:
            self._last_log_at[key] = now
            print(message, flush=True)

    # ------------------------------------------------------------ hold / stop

    def _hold_locked(self, reason):
        if not self._holding:
            self._hold_action = None
            log_event("hold", reason=reason)
            self._throttled("hold", f"[client] hold: {reason}")
        self._holding = True

    def _hold(self, reason):
        with self._publish_lock:
            self._hold_locked(reason)

    def _block_locked(self):
        with self._action_lock:
            self._action_floor = None
            self._last_accepted_at = None

    def _hold_and_block(self, reason):
        with self._publish_lock:
            self._hold_locked(reason)
            self._block_locked()

    def _terminate_locked(self, reason):
        self._alive = False
        if self._wbc_started and not self.dry_run:
            self.token_pub.command(stop=True)
        self._wbc_started = False
        self._block_locked()
        log_event("terminate", reason=reason)

    def hold_loop(self):
        """30 Hz while holding: re-encode the measured pose, keep the last hands/neck."""
        while self._alive and self.running.is_set():
            started = time.monotonic()
            with self._publish_lock:
                if self._holding and self._wbc_started and not self.dry_run:
                    try:
                        if self.state_sub.age() > self.obs_stale_timeout:
                            raise RuntimeError("robot state is stale")
                        if self._hold_action is None:
                            self._hold_action = self._freeze(self._last_published)
                        self._execute(self._hold_action)
                        if self.rollout is not None:
                            self.rollout.record_action(time.monotonic(), -1, -1, -1, True, self._hold_action)
                    except Exception as exc:
                        print(f"[client] ERROR: hold failed ({exc}); stopping WBC", flush=True)
                        self._terminate_locked(f"hold failure: {exc}")
            time.sleep(max(0.0, CTRL_DT - (time.monotonic() - started)))

    def _freeze(self, action):
        frozen = np.array(action, np.float32, copy=True)
        frozen[HAND_DIM:HAND_DIM + TOKEN_DIM] = self.encoder.encode(self.state_sub.get())
        return frozen

    # ------------------------------------------------------------ step transitions

    def _transition(self, label, apply):
        """Change the condition atomically with respect to observation send and
        publication: nothing computed for the previous epoch can execute afterwards."""
        with self._send_lock, self._publish_lock:
            before = self.wm.snapshot()["epoch"]
            apply()
            after = self.wm.snapshot()["epoch"]
            if after != before:
                self._execution_started_at = None
                self._hold_locked(f"{label}: epoch {before} -> {after}")
                self._block_locked()
                log_event("transition", label=label, epoch_before=before, epoch_after=after, step=self._step)

    def set_step(self, step, reason):
        def apply():
            self.wm.set_task(step)
            self._step, self._instruction_hold = step, None
        self._transition(f"step ({reason})", apply)

    def hold_instructions(self, reason):
        def apply():
            self._instruction_hold = reason
            self.wm.disable()
        self._transition(reason, apply)

    def condition_snapshot(self):
        with self._publish_lock, self._action_lock:
            return {"executing": self._action_floor is not None and not self._holding,
                    "execution_started_at": self._execution_started_at}

    # ------------------------------------------------------------ observation

    def _build_state(self, state):
        body = np.asarray(state["body_q_measured"], np.float32).reshape(29)
        hands = np.concatenate([np.asarray(state["left_hand_q"], np.float32).reshape(7),
                                np.asarray(state["right_hand_q"], np.float32).reshape(7)])
        parts = [hands, body[15:29], body[:15]]  # hand | arm | leg (| neck)
        if self.neck:
            neck = self.neck_reader.get()
            if neck is None or self.neck_reader.age() > self.obs_stale_timeout:
                raise ValueError(f"neck state missing or stale ({self.neck_reader.age():.3f}s)")
            parts.append(neck)
        states = np.concatenate(parts)
        if states.shape != (self.state_dim,) or not np.isfinite(states).all():
            raise ValueError(f"bad model state {states.shape}")
        return states

    def _observe(self):
        """One tick's inputs, or None when gated (the robot is then already held)."""
        frame = self.camera.get_frame()  # capture first, then sample state (as in collection)
        state = self.state_sub.get()
        if state is None or self.state_sub.age() > self.obs_stale_timeout:
            self._hold("robot state missing or stale")
            return None
        states = self._build_state(state)
        self.wm.update_ego(frame)
        if self.rollout is not None:
            now = time.monotonic()
            self.rollout.record_state(now, states)
            self.rollout.record_frame(now, frame)
        wm = self.wm.snapshot()
        if wm["expired"]:
            self._hold_and_block(f"WM goal expired ({wm['goal_age']:.1f}s)")
            return None
        if wm["goal"] is None:
            self._hold("waiting for the WM goal of the current step")
            self._throttled("gate", f"[client] gated: waiting for WM goal (epoch={wm['epoch']} error={wm['error']!r})")
            return None
        if self._instruction_hold:
            return None
        goal = resize_goal(wm["goal"], self.goal_hw)
        if self.show_goal and wm["generation"] != self._last_goal_generation:
            self._last_goal_generation = wm["generation"]
            cv2.imshow("VLA goal", cv2.cvtColor(goal, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return Observation(frame=frame, state=state, states=states, goal=goal, epoch=wm["epoch"])

    def _payload(self, obs, extra_state=None):
        return {"image": {EGO_KEY: obs.frame, GOAL_KEY: obs.goal},
                "state": {"states": obs.states, **(extra_state or {})},
                "instruction": self.instruction, "history": None, "condition": None,
                "gt_action": None, "dataset_name": None, "timestamp": None}

    def _observation_sent(self):
        with self._action_lock:
            if self._action_floor is None:
                self._action_floor = self._last_version
        self._last_obs_at = time.monotonic()

    def _fresh(self):
        ages = {"obs": float("inf") if self._last_obs_at is None else time.monotonic() - self._last_obs_at,
                "state": self.state_sub.age(), "camera": self.camera.age()}
        if self.neck:
            ages["neck"] = self.neck_reader.age()
        return max(ages.values()) <= self.obs_stale_timeout, ages

    # ------------------------------------------------------------ action

    def _accept(self, version, epoch):
        """(ok, reason). HTTP knows the epoch of the observation a chunk answers; RTC
        does not (epoch=None) and relies on the server numbering replies in order."""
        with self._action_lock:
            if version <= self._last_version:
                raise ValueError(f"non-monotonic action version {version} <= {self._last_version}")
            self._last_version = version
            floor = self._action_floor
        if epoch is not None:
            current = self.wm.snapshot()["epoch"]
            return (True, None) if epoch == current else (False, f"computed for epoch {epoch}, now {current}")
        if floor is None:
            return False, "no observation sent for the current step yet"
        if version <= floor:
            return False, f"version {version} <= floor {floor}: computed before the step changed"
        return True, None

    def _check_liveness(self):
        if self.dry_run:
            return
        with self._publish_lock:
            if not self._wbc_started or self._holding:
                return
            with self._action_lock:
                last = self._last_accepted_at
            age = float("inf") if last is None else time.monotonic() - last
            if age > self.action_stale_timeout:
                self.token_pub.command(stop=True)
                self._wbc_started = False
                log_event("wbc_stop", reason=f"action stream stale {age:.3f}s")
                print(f"[safety] WBC stopped: no fresh action for {age:.3f}s", flush=True)

    def _execute(self, action):
        """action = [hand 14 | token 64 (| neck 2)]. The policy token goes out raw
        (no FSQ quantization): the SONIC decoder takes continuous values."""
        if self.dry_run:
            return
        self.token_pub.publish(action[HAND_DIM:HAND_DIM + TOKEN_DIM], action[:HAND_DIM])
        if self.neck:
            self.neck_pub.publish(*action[HAND_DIM + TOKEN_DIM:HAND_DIM + TOKEN_DIM + NECK_DIM])
        self._last_published = action.copy()

    def _publish(self, action, version, epoch, meta, freeze_repeat=False):
        """Validate, gate and publish one action row. Returns ok / held / dropped."""
        action = np.asarray(action, np.float32).reshape(-1)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            self._hold(f"invalid VLA action {action.shape}")
            return "dropped"
        wm = self.wm.snapshot()
        if wm["expired"]:
            self._hold_and_block("WM goal expired")
            return "held"
        if wm["goal"] is None:
            self._hold("no WM goal for the current step")
            return "held"
        fresh, ages = self._fresh()
        if not fresh:
            self._hold("stale observation " + " ".join(f"{k}={v:.3f}s" for k, v in ages.items()))
            return "held"
        ok, reason = self._accept(int(version), epoch)
        if not ok:
            self._hold(f"action not for the current step: {reason}")
            self._throttled("drop", f"[client] dropped action version={version}: {reason}", 1.0)
            return "dropped"
        now = time.monotonic()
        with self._publish_lock:
            if self.wm.snapshot()["epoch"] != wm["epoch"]:
                self._hold_locked("step changed before publication")
                return "dropped"
            self._holding, self._hold_action = False, None
            if not self.dry_run and not self._wbc_started:
                self.token_pub.command(start=True)
                self._wbc_started = True
                log_event("wbc_start")
            if freeze_repeat and meta.get("rtc_repeat_last"):
                action = self._freeze(action)  # server ran out of chunk: hold the measured pose
            self._execute(action)
            with self._action_lock:
                self._last_accepted_at = now
            if self._execution_started_at is None:
                self._execution_started_at = now
                log_event("execution_started", epoch=wm["epoch"], version=int(version))
        if self.rollout is not None:
            self.rollout.record_action(now, int(version), meta.get("rtc_chunk_id"), meta.get("rtc_chunk_tick"),
                                       bool(meta.get("rtc_repeat_last")), action)
        return "ok"

    def stop(self):
        self._alive = False
        with self._publish_lock:
            self._terminate_locked("client shutdown")
        self._connected.set()


class RTCWebSocketClient(VlaClientBase):
    """Observations up and one action row per server control tick down, over /ws."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ws = None
        self._sender = None
        self._sent_init_prev = False

    def _on_open(self, ws):
        with self._action_lock:
            self._action_floor = self._last_accepted_at = None
        print("[client] connected to the VLA server", flush=True)
        self._connected.set()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get("action") is not None:
                action = convert_numpy(data["action"], numpy_deserialize)
                self._publish(action, data.get("version", -1), None, data, freeze_repeat=True)
        except Exception as exc:
            self._hold(f"action processing error: {exc}")
            print(f"[client] action processing error: {exc}", flush=True)

    def _on_close(self, ws, code, reason):
        print(f"[client] VLA connection closed: {code} {reason}", flush=True)
        self._hold("VLA WebSocket closed")
        self._alive = False
        self.running.clear()

    def _send_loop(self):
        while self._alive and self.running.is_set() and not self._connected.wait(0.1):
            pass
        while self._alive and self.running.is_set():
            started = time.monotonic()
            self._check_liveness()
            try:
                obs = self._observe()
                if obs is not None:
                    extra = {}
                    if not self._sent_init_prev:
                        # first frame: current hands + encoder token of the measured pose (+ neck),
                        # the one-row prefix the server's first RTC chunk continues from
                        prev = [obs.states[:HAND_DIM], self.encoder.encode(obs.state)]
                        if self.neck:
                            prev.append(obs.states[-NECK_DIM:])
                        extra["init_prev_action"] = np.concatenate(prev)
                    message = json.dumps(convert_numpy(self._payload(obs, extra), numpy_serialize))
                    with self._send_lock:
                        wm = self.wm.snapshot()
                        if wm["epoch"] == obs.epoch and not wm["expired"]:
                            if not (self._ws.sock and self._ws.sock.connected):
                                self._hold("VLA WebSocket disconnected")
                                break
                            self._ws.send(message)
                            self._observation_sent()
                            self._sent_init_prev = True
            except Exception as exc:
                self._hold(f"observation error: {exc}")
                self._throttled("obs", f"[client] observation rejected: {exc}")
            time.sleep(max(0.0, CTRL_DT - (time.monotonic() - started)))

    def run(self):
        print(f"[client] RTC WebSocket -> {self.server_url}", flush=True)
        self._ws = WebSocketApp(self.server_url, on_open=self._on_open, on_message=self._on_message,
                                on_error=lambda ws, e: print(f"[client] WebSocket error: {e}", flush=True),
                                on_close=self._on_close)
        self._sender = threading.Thread(target=self._send_loop, name="vla-send", daemon=True)
        self._sender.start()
        self._ws.run_forever()
        self._alive = False
        self._connected.set()
        self._sender.join(timeout=2.0)

    def stop(self):
        super().stop()
        if self._ws is not None:
            self._ws.close()


class HttpChunkClient(VlaClientBase):
    """rtc off: capture at 30 Hz, POST the newest observation to /act when the played
    chunk runs out, hold the measured pose until the next chunk lands."""

    def __init__(self, *args, http_timeout=30.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.http_timeout = http_timeout
        self.overdue_timeout = 2.0
        self._http = requests.Session()
        self._http.trust_env = False
        self._obs_lock, self._chunk_lock = threading.Lock(), threading.Lock()
        self._latest = None          # newest observation record
        self._pending = None         # chunk landed, not yet played
        self._request = threading.Event()
        self._request_seq = self._chunk_seq = self._version = 0
        self._threads = []

    def _capture_loop(self):
        while self._alive and self.running.is_set():
            started = time.monotonic()
            self._check_liveness()
            try:
                obs = self._observe()
                record = None if obs is None else {"payload": self._payload(obs), "epoch": obs.epoch,
                                                   "captured_at": time.monotonic()}
                with self._obs_lock:
                    self._latest = record
                if record is not None:
                    self._observation_sent()
            except Exception as exc:
                self._hold(f"observation error: {exc}")
                self._throttled("obs", f"[client] observation rejected: {exc}")
                with self._obs_lock:
                    self._latest = None
            time.sleep(max(0.0, CTRL_DT - (time.monotonic() - started)))

    def _infer_loop(self):
        while self._alive and self.running.is_set():
            if not self._request.wait(0.05):
                continue
            seq = self._request_seq
            with self._obs_lock:
                record = self._latest
            if record is None or time.monotonic() - record["captured_at"] > self.obs_stale_timeout:
                time.sleep(CTRL_DT)
                continue
            with self._send_lock:
                if self.wm.snapshot()["epoch"] != record["epoch"]:
                    time.sleep(CTRL_DT)
                    continue
            t0 = time.monotonic()
            try:
                response = self._http.post(self.server_url, timeout=self.http_timeout,
                                           data=json.dumps(convert_numpy(record["payload"], numpy_serialize)),
                                           headers={"Content-Type": "application/json"})
                response.raise_for_status()
                chunk = np.asarray(convert_numpy(response.json()["action"], numpy_deserialize), np.float32)
                chunk = chunk.reshape(-1, self.action_dim)
            except Exception as exc:
                self._hold(f"VLA /act failed: {exc}")
                self._throttled("act", f"[http] /act failed: {exc}", 1.0)
                time.sleep(0.2)
                continue
            self._chunk_seq += 1
            with self._chunk_lock:
                self._pending = {"seq": self._chunk_seq, "chunk": chunk, "epoch": record["epoch"],
                                 "infer_ms": (time.monotonic() - t0) * 1000.0}
            if self._request_seq == seq:
                self._request.clear()
            log_event("http_chunk", seq=self._chunk_seq, rows=len(chunk), epoch=record["epoch"],
                      infer_ms=(time.monotonic() - t0) * 1000.0)

    def _request_chunk(self):
        self._request_seq += 1
        self._request.set()
        return time.monotonic()

    def _play_loop(self):
        chunk = meta = frozen = last = None
        idx = 0
        requested_at = self._request_chunk()
        next_tick = time.monotonic()
        while self._alive and self.running.is_set():
            with self._chunk_lock:
                pending, self._pending = self._pending, None
            if pending is not None:
                if pending["epoch"] != self.wm.snapshot()["epoch"]:
                    requested_at = self._request_chunk()  # computed for an old step
                else:
                    chunk, meta, idx, frozen, requested_at = pending["chunk"], pending, 0, None, None
            if chunk is not None and idx < len(chunk):
                action, tick, repeat = chunk[idx], idx, False
                idx += 1
            else:
                requested_at = requested_at or self._request_chunk()
                if last is None or meta is None:
                    action = None
                elif time.monotonic() - requested_at > self.overdue_timeout:
                    self._hold(f"VLA chunk overdue ({time.monotonic() - requested_at:.1f}s)")
                    action = None
                else:
                    frozen = self._freeze(last) if frozen is None else frozen
                    action, tick, repeat = frozen, len(chunk), True
            if action is not None:
                self._version += 1
                outcome = self._publish(action, self._version, meta["epoch"],
                                        {"rtc_chunk_id": meta["seq"], "rtc_chunk_tick": tick, "rtc_repeat_last": repeat})
                if outcome == "ok":
                    last = np.array(action, np.float32, copy=True)
                elif outcome == "dropped":
                    chunk = meta = frozen = last = None
                    idx = 0
                    requested_at = self._request_chunk()
            next_tick += CTRL_DT
            now = time.monotonic()
            if next_tick > now:
                time.sleep(next_tick - now)
            else:
                next_tick = now  # a missed tick restarts the schedule instead of bursting

    def run(self):
        print(f"[client] HTTP chunk mode -> {self.server_url}", flush=True)
        self._connected.set()
        self._threads = [threading.Thread(target=self._capture_loop, name="vla-capture", daemon=True),
                         threading.Thread(target=self._infer_loop, name="vla-act", daemon=True)]
        for thread in self._threads:
            thread.start()
        self._play_loop()

    def stop(self):
        super().stop()
        self._request.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._http.close()
