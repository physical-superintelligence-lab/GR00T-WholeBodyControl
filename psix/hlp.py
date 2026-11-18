"""Single-task HLP: the planner gets the task sentence and proposes one
step at a time; the current step becomes the WM prompt (the VLA keeps the task).
When the planner reports the task done, the robot keeps running the current step.
Operator keys: Enter resumes, p undoes the last step, i types a step, q quits."""

import os
import queue
import select
import sys
import termios
import threading
import time
import tty

import requests

from .recording import log_event
from .robot import encode_jpeg_b64


class HlpClient:
    def __init__(self, base_url, task, recorder, timeout=20.0, jpeg_quality=90):
        self.base_url, self.task, self.recorder = base_url.rstrip("/"), task, recorder
        self.timeout, self.jpeg_quality = timeout, jpeg_quality
        self._http = requests.Session()
        self._http.trust_env = False

    def request(self, path, body):
        record = self.recorder.begin("hlp", body, context={"path": path})
        try:
            response = self._http.post(self.base_url + path, json=body, timeout=self.timeout)
            reply = response.json()
        except (requests.RequestException, ValueError) as exc:
            self.recorder.finish(record, "transport_error", error=str(exc))
            raise RuntimeError(f"HLP {path}: {exc}") from exc
        ok = response.status_code == 200 and "error" not in reply
        self.recorder.finish(record, "ok" if ok else "server_error", reply=reply)
        if not ok:
            raise RuntimeError(f"HLP {path}: HTTP {response.status_code} {reply.get('error', reply)}")
        return reply

    def reset(self):
        return self.request("/reset", {"task": self.task})

    def poll(self, rgb, executed_revision):
        return self.request("/hlp", {"task": self.task, "executed_revision": executed_revision,
                                     "ego_image": {"jpeg_b64": encode_jpeg_b64(rgb, self.jpeg_quality),
                                                   "quality": self.jpeg_quality}})

    def close(self):
        self._http.close()


class StepManager:
    """Polls the HLP on its own thread and applies its current step to the VLA client.
    Operator commands are queued onto that thread, so no HTTP runs on the keyboard."""

    def __init__(self, client, wm, hlp, period=1.0):
        self.client, self.wm, self.hlp, self.period = client, wm, hlp, period
        self._planner = {}   # the HLP's last reply
        self._key = None     # (revision, step) last applied
        self._revision = None
        self._paused = self._done = False
        self._lock = threading.RLock()
        self._stop, self._wake = threading.Event(), threading.Event()
        self._commands = queue.Queue()
        self._thread = None
        client.hold_instructions("waiting for the first HLP step")

    def command(self, name, text=None):
        """name: next (Enter), prev (p), override (typed step)."""
        with self._lock:
            if name == "prev" and not self._planner.get("can_undo"):
                print("[hlp] nothing to undo", flush=True)
                return
            if name == "next" and not self._done:
                self._paused = False  # only polling was paused; the robot runs the current step
                self._wake.set()
                return
            self.client.hold_instructions(f"operator {name}: synchronizing HLP")
            self._commands.put((name, text))
            self._wake.set()

    def _run_command(self, name, text):
        path, body = {"next": ("/resume", {}), "prev": ("/undo", {}),
                      "override": ("/override", {"subtask": text})}[name]
        try:
            reply = self.hlp.request(path, body)
        except RuntimeError as exc:
            self._fail(exc)
            return
        with self._lock:
            self._planner = reply
            self._apply(reply, source="operator", force=True)
            self._paused = name != "next"
            if self._paused:
                print("[hlp] paused; Enter resumes", flush=True)

    def poll_once(self):
        with self._lock:
            if self._paused:
                return
        rgb, captured = self.wm.latest_ego()
        if rgb is None:
            return
        try:
            reply = self.hlp.poll(rgb, self._executed_revision(captured))
        except RuntimeError as exc:
            self._fail(exc)
            return
        with self._lock:
            self._planner = reply
            self._apply(reply)

    def _executed_revision(self, captured):
        """The revision the robot has started running, reported only from a frame
        captured after that start; the planner needs both to advance."""
        status = self.client.condition_snapshot()
        started = status["execution_started_at"]
        seen = started is not None and captured is not None and captured >= started
        return self._revision if status["executing"] and seen else None

    def _apply(self, planner, source="hlp", force=False):
        key = (planner["revision"], planner["instruction"])
        if key == self._key and not force:
            return
        step = planner["instruction"]
        if step is not None:
            self.client.set_step(step, reason=source)
        elif not (planner["done"] and source == "hlp" and self._key and self._key[1]):
            self.client.hold_instructions(planner.get("reason") or "HLP planning")
        self._key, self._revision, self._done = key, planner["revision"], bool(planner["done"])
        print(f"[hlp] step ({source}) rev={planner['revision']}: {step!r}", flush=True)
        log_event("hlp_step", source=source, step=step, revision=planner["revision"], done=self._done,
                  reason=planner.get("reason"))
        if self._done:
            self._paused = True
            print("[hlp] task done; Enter resumes the planner, p reopens the last step", flush=True)

    def _fail(self, exc):
        with self._lock:
            if self._key is not None:
                self.client.hold_instructions("HLP unreachable")
            self._key = None
        print(f"[hlp] failed: {exc}", flush=True)

    def _loop(self):
        while not self._stop.is_set():
            started = time.monotonic()
            self._wake.clear()
            if not self._planner:
                try:
                    self._planner = self.hlp.reset()
                    print(f"[hlp] episode started: {self.hlp.task!r}", flush=True)
                    log_event("hlp_episode_started", task=self.hlp.task, reply=self._planner)
                except RuntimeError as exc:
                    self._fail(exc)
                    self._stop.wait(2.0)
                continue
            try:
                name, text = self._commands.get_nowait()
            except queue.Empty:
                self.poll_once()
                self._wake.wait(max(0.0, self.period - (time.monotonic() - started)))
            else:
                self._run_command(name, text)

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="hlp", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.hlp.close()


class Keyboard:
    """Single-key terminal input: Enter, p, q act at once; i starts typing a step."""

    def __init__(self, steps, running):
        self.steps, self.running = steps, running
        self._stop = threading.Event()
        self._saved = None
        self._thread = None

    def start(self):
        if not sys.stdin.isatty():
            print("[keyboard] stdin is not a terminal; keys disabled", flush=True)
            return
        self._saved = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._read, name="keyboard", daemon=True)
        self._thread.start()

    def _read(self):
        fd, buffer, typing = sys.stdin.fileno(), bytearray(), False
        while self.running.is_set() and not self._stop.is_set():
            if not select.select([fd], [], [], 0.1)[0]:
                continue
            key = os.read(fd, 1)
            if typing:
                if key in (b"\r", b"\n"):
                    text = buffer.decode(errors="replace").strip()
                    buffer.clear()
                    typing = False
                    print(flush=True)
                    if text:
                        self.steps.command("override", text)
                elif key == b"\x1b":
                    buffer.clear()
                    typing = False
                    print("\n[keyboard] cancelled", flush=True)
                else:
                    if key in (b"\x7f", b"\x08"):
                        buffer[:] = buffer.decode(errors="ignore")[:-1].encode()
                    else:
                        buffer.extend(key)
                    sys.stdout.write("\r[step] > " + buffer.decode(errors="replace") + "\033[K")
                    sys.stdout.flush()
            elif key in (b"\r", b"\n"):
                self.steps.command("next")
            elif key == b"p":
                self.steps.command("prev")
            elif key == b"i":
                typing = True
                print("[keyboard] type a step, Enter applies, Esc cancels", flush=True)
            elif key == b"q":
                self.running.clear()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._saved is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
