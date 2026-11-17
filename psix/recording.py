"""Run records: events.jsonl, run_manifest.json, HLP/WM request records, and the
continuous rollout (states, executed actions, ego video). Disk I/O runs on writer
threads so the 30 Hz paths only enqueue."""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from base64 import b64decode
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .robot import GROOT_ROOT

_EVENTS = None


class EventLog:
    def __init__(self, path):
        self._queue = queue.Queue(maxsize=4096)
        self._thread = threading.Thread(target=self._writer, args=(path,), name="event-log", daemon=True)
        self._thread.start()

    def emit(self, kind, **fields):
        record = {"kind": kind, "t_wall": datetime.now().isoformat(timespec="milliseconds"),
                  "t_mono": time.monotonic(), **fields}
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass

    def stop(self):
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    def _writer(self, path):
        with open(path, "a", encoding="utf-8") as f:
            for item in iter(self._queue.get, None):
                f.write(json.dumps(item, default=str) + "\n")
                if self._queue.empty():
                    f.flush()


def set_event_log(log):
    global _EVENTS
    _EVENTS = log


def log_event(kind, **fields):
    if _EVENTS is not None:
        _EVENTS.emit(kind, **fields)


def write_manifest(run_dir, config, vla_info):
    sha = subprocess.run(["git", "-C", GROOT_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", GROOT_ROOT, "status", "--porcelain"], capture_output=True,
                                text=True).stdout.strip())
    manifest = {"written_at": datetime.now().isoformat(timespec="seconds"), "argv": sys.argv, "config": config,
                "groot_repo": {"sha": sha, "dirty": dirty}, "vla_info": vla_info}
    Path(run_dir, "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))


class RequestRecorder:
    """One JSON per HLP/WM request (+ the sent ego and returned goal as JPEG), written
    before the POST and updated with the reply and the client's decision."""

    def __init__(self, run_dir, enabled=True):
        self.root = Path(run_dir) / "requests"
        self.enabled = enabled

    @staticmethod
    def _save(record):
        path, data = record
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        os.replace(tmp, path)

    def begin(self, service, payload, context=None):
        if not self.enabled:
            return None
        directory = self.root / service
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f"{time.time_ns()}_req-{payload.get('req_id', 0)}"
        request = deepcopy(payload)
        for holder, key in ((request, "ego_jpeg"), (request.get("ego_image") or {}, "jpeg_b64")):
            if isinstance(holder.get(key), str):
                Path(f"{stem}.obs.jpg").write_bytes(b64decode(holder[key]))
                holder[key] = f"{stem.name}.obs.jpg"
        record = (Path(f"{stem}.json"), {"service": service, "started_mono": time.monotonic(), "request": request,
                                         "context": context or {}, "outcome": "pending"})
        self._save(record)
        return record

    def finish(self, record, outcome, reply=None, **details):
        if record is None:
            return
        path, data = record
        if isinstance(reply, dict):
            reply = dict(reply)
            if isinstance(reply.get("subgoal_jpeg"), str):
                path.with_suffix(".goal.jpg").write_bytes(b64decode(reply["subgoal_jpeg"]))
                reply["subgoal_jpeg"] = path.with_suffix(".goal.jpg").name
            data["reply"] = reply
        data.update(outcome=outcome, finished_mono=time.monotonic(), **details)
        self._save(record)


class RolloutRecorder:
    """states_*.npz / actions_*.npz shards, MJPG ego.avi at video_fps, rollout_meta.json.
    Buffers land every 10 s so a killed run keeps all but the last few seconds."""

    SHARD, FLUSH_S = 3000, 10.0

    def __init__(self, out_dir, video_fps=10.0):
        self.dir, self.video_fps = out_dir, float(video_fps)
        os.makedirs(out_dir, exist_ok=True)
        self._queue = queue.Queue(maxsize=4096)
        self._last_frame_at = -float("inf")
        self._t0 = time.monotonic()
        self.dropped = 0
        self._thread = threading.Thread(target=self._writer, name="rollout-recorder", daemon=True)
        self._thread.start()

    def _put(self, item):
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def record_state(self, mono, states):
        self._put(("state", (mono, np.asarray(states, np.float32).copy())))

    def record_action(self, mono, version, chunk_id, chunk_tick, repeat_last, action):
        self._put(("action", (mono, int(version), -1 if chunk_id is None else int(chunk_id),
                              -1 if chunk_tick is None else int(chunk_tick), bool(repeat_last),
                              np.asarray(action, np.float32).copy())))

    def record_frame(self, mono, rgb):
        if mono - self._last_frame_at >= 1.0 / self.video_fps:
            self._last_frame_at = mono
            self._put(("frame", (mono, rgb)))

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=15.0)

    def _flush(self, kind, rows, index):
        if not rows:
            return index
        path = os.path.join(self.dir, f"{kind}s_{index:04d}.npz")
        if kind == "state":
            np.savez_compressed(path, mono=np.array([r[0] for r in rows]), states=np.stack([r[1] for r in rows]))
        else:
            cols = list(zip(*rows))
            np.savez_compressed(path, mono=np.array(cols[0]), version=np.array(cols[1]),
                                chunk_id=np.array(cols[2]), chunk_tick=np.array(cols[3]),
                                repeat_last=np.array(cols[4]), action=np.stack(cols[5]))
        rows.clear()
        return index + 1

    def _write_meta(self, shards, frame_mono, partial):
        meta = {"partial": partial, "duration_s": time.monotonic() - self._t0, "shards": shards,
                "video": {"file": "ego.avi", "fps": self.video_fps, "frames": len(frame_mono)},
                "video_mono": frame_mono, "dropped": self.dropped}
        Path(self.dir, "rollout_meta.json").write_text(json.dumps(meta))

    def _writer(self):
        rows = {"state": [], "action": []}
        shards = {"state": 0, "action": 0}
        video, frame_mono, last_flush = None, [], time.monotonic()
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                item = ()
            if item is None:
                break
            if time.monotonic() - last_flush >= self.FLUSH_S:
                for kind in rows:
                    shards[kind] = self._flush(kind, rows[kind], shards[kind])
                self._write_meta(shards, frame_mono, partial=True)
                last_flush = time.monotonic()
            if not item:
                continue
            kind, payload = item
            if kind == "frame":
                mono, rgb = payload
                if video is None:
                    h, w = rgb.shape[:2]
                    video = cv2.VideoWriter(os.path.join(self.dir, "ego.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                            self.video_fps, (w, h))
                video.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                frame_mono.append(mono)
            else:
                rows[kind].append(payload)
                if len(rows[kind]) >= self.SHARD:
                    shards[kind] = self._flush(kind, rows[kind], shards[kind])
        for kind in rows:
            shards[kind] = self._flush(kind, rows[kind], shards[kind])
        if video is not None:
            video.release()
        self._write_meta(shards, frame_mono, partial=False)
