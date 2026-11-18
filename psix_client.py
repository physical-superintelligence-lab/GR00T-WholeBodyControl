"""Run one task on a G1 with the psix VLA (see README_PSIX.md).

    python psix_client.py --prompt "Pick up banana and place it into the white basket."          # dry run
    python psix_client.py --real --prompt "Pick up banana and place it into the white basket."
    python psix_client.py --real --no-hlp --prompt "..."     # without the HLP server
    python psix_client.py --real --no-neck --prompt "..."    # G1 without the neck
"""

import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from psix import recording, robot
from psix.hlp import HlpClient, Keyboard, StepManager
from psix.vla import HttpChunkClient, RTCWebSocketClient
from psix.wm import WmClient


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", required=True, help="task sentence, spelled as in the training data")
    p.add_argument("--real", action="store_true", help="publish robot commands (default: dry run)")
    p.add_argument("--no-neck", action="store_true", help="robot without neck: RealSense head camera, 43/78 dims")
    p.add_argument("--no-hlp", action="store_true", help="no HLP server: the WM is prompted with the task sentence")
    p.add_argument("--host", default="127.0.0.1", help="psix VLA server")
    p.add_argument("--port", type=int, default=8014)
    p.add_argument("--hlp-host", default="127.0.0.1")
    p.add_argument("--hlp-port", type=int, default=8015)
    p.add_argument("--hlp-period", type=float, default=1.0)
    p.add_argument("--hlp-timeout", type=float, default=20.0)
    p.add_argument("--wm-host", default="192.168.123.240")
    p.add_argument("--wm-port", type=int, default=8016)
    p.add_argument("--wm-period", type=float, default=0.8, help="WM refresh period (s)")
    p.add_argument("--wm-seconds", type=float, default=2.4, help="WM future horizon (s); 2.4 s = the training goal offset")
    p.add_argument("--wm-timeout", type=float, default=15.0)
    p.add_argument("--wm-goal-hard-age", type=float, default=30.0, help="hold once the last good goal is older")
    p.add_argument("--robot-host", default="127.0.0.1", help="host of the deploy's g1_debug state stream")
    p.add_argument("--camera", default="tcp://192.168.123.164:5558", help="zed_server.py or RealSense server REP")
    p.add_argument("--neck-state", default="tcp://192.168.123.164:5560")
    p.add_argument("--token-port", type=int, default=5556)
    p.add_argument("--neck-port", type=int, default=5570)
    p.add_argument("--obs-stale-timeout", type=float, default=0.5)
    p.add_argument("--action-stale-timeout", type=float, default=0.5)
    p.add_argument("--http-timeout", type=float, default=30.0)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--no-neck-home", action="store_true", help="skip driving the neck to [0, 0] before start")
    p.add_argument("--show-goal", action="store_true", help="cv2 window with the goal image sent to the VLA")
    p.add_argument("--output-dir", default=None, help="run directory (default ~/psix_rollouts/<time>)")
    p.add_argument("--no-record", action="store_true", help="no request / rollout recording")
    return p.parse_args(argv)


def preflight(args):
    neck = not args.no_neck
    info, enc_version, enc_path, goal_hw = robot.read_vla_info(args.host, args.port, neck)
    if robot.get_json(f"http://{args.wm_host}:{args.wm_port}/ready").get("ready") is not True:
        raise RuntimeError("WM /ready did not report ready=true")
    if not args.no_hlp:
        health = robot.get_json(f"http://{args.hlp_host}:{args.hlp_port}/health")
        if health.get("status") != "ok":
            raise RuntimeError(f"HLP /health did not report status ok: {health}")
    print(f"[client] VLA {info['embodiment_tag']} ckpt_{info.get('ckpt_step')} rtc={info['rtc_mode']} "
          f"encoder={enc_version} goal={goal_hw} neck={neck} hlp={not args.no_hlp}", flush=True)
    return info, enc_version, enc_path, goal_hw


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def main(argv=None):
    args = parse_args(argv)
    neck = not args.no_neck
    info, enc_version, enc_path, goal_hw = preflight(args)
    run_dir = Path(args.output_dir or Path.home() / "psix_rollouts" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "client.log", "a", buffering=1)
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log_file)
    print(f"[client] task: {args.prompt!r}  output: {run_dir}  {'REAL' if args.real else 'dry run'}", flush=True)

    events = recording.EventLog(run_dir / "events.jsonl")
    recording.set_event_log(events)
    recorder = recording.RequestRecorder(run_dir, enabled=not args.no_record)
    rollout = None if args.no_record else recording.RolloutRecorder(run_dir / "rollout")
    recording.write_manifest(run_dir, vars(args), info)

    running = threading.Event()
    running.set()
    camera = robot.Camera(args.camera)
    state_sub = robot.RobotStateSubscriber(args.robot_host)
    token_pub = robot.TokenPublisher(args.token_port) if args.real else None
    neck_reader = robot.NeckStateReader(args.neck_state) if neck else None
    neck_pub = robot.NeckPublisher(args.neck_port) if neck and args.real else None
    encoder = robot.BodyTokenEncoder(enc_path, enc_version)
    if neck_pub is not None and not args.no_neck_home:
        time.sleep(0.5)  # let the subscriber connect before homing
        ok, angles = robot.home_neck(neck_pub, neck_reader)
        print(f"[client] neck home {'converged' if ok else 'NOT converged'} at {angles}", flush=True)
        recording.log_event("neck_home", converged=ok, neck=angles)
        time.sleep(5.0)  # zed_server smooths the command; let the head settle before observing

    wm = WmClient(f"http://{args.wm_host}:{args.wm_port}", args.prompt, recorder, period=args.wm_period,
                  seconds=args.wm_seconds, timeout=args.wm_timeout, jpeg_quality=args.jpeg_quality,
                  goal_hard_age=args.wm_goal_hard_age)
    common = dict(task=args.prompt, camera=camera, state_sub=state_sub, wm=wm, encoder=encoder, goal_hw=goal_hw,
                  running=running, neck=neck, neck_reader=neck_reader, token_pub=token_pub, neck_pub=neck_pub,
                  rollout=rollout, dry_run=not args.real, obs_stale_timeout=args.obs_stale_timeout,
                  action_stale_timeout=args.action_stale_timeout, show_goal=args.show_goal)
    if info["rtc_mode"] == "off":
        client = HttpChunkClient(f"http://{args.host}:{args.port}/act", http_timeout=args.http_timeout, **common)
    else:
        client = RTCWebSocketClient(f"ws://{args.host}:{args.port}/ws", **common)
    steps = keyboard = None
    if not args.no_hlp:
        hlp = HlpClient(f"http://{args.hlp_host}:{args.hlp_port}", args.prompt, recorder,
                        timeout=args.hlp_timeout, jpeg_quality=args.jpeg_quality)
        steps = StepManager(client, wm, hlp, period=args.hlp_period)
        keyboard = Keyboard(steps, running)

    for _ in range(30):
        if state_sub.get() is not None:
            break
        time.sleep(0.5)
    else:
        print("[client] WARNING: no robot state after 15 s", flush=True)

    signal.signal(signal.SIGINT, lambda *_: running.clear())
    signal.signal(signal.SIGTERM, lambda *_: running.clear())
    wm.start()
    if steps is not None:
        steps.start()
        keyboard.start()
    transport = threading.Thread(target=client.run, name="vla", daemon=True)
    transport.start()
    threading.Thread(target=client.hold_loop, name="hold", daemon=True).start()
    recording.log_event("run_start", real=args.real, rtc_mode=info["rtc_mode"], neck=neck, hlp=not args.no_hlp)
    print("[client] running; " + ("Ctrl-C quits" if args.no_hlp else
                                  "Enter: resume | p: undo step | i: type step | q: quit"), flush=True)
    try:
        while running.is_set() and transport.is_alive():
            time.sleep(0.2)
    finally:
        running.clear()
        recording.log_event("shutdown")
        client.stop()  # sends the WBC stop first
        transport.join(timeout=3.0)
        for component in (keyboard, steps, wm, camera, state_sub, neck_reader, token_pub, neck_pub):
            if component is not None:
                component.stop()
        if rollout is not None:
            rollout.close()
        events.stop()
        print("[client] shutdown complete", flush=True)


if __name__ == "__main__":
    main()
