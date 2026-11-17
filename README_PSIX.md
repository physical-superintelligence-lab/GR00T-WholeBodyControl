# psix on the real G1

`psix_client.py` runs one task on a G1 with the psix policy. A high-level planner (HLP)
splits the task into steps, a world model (WM) turns the current step and the camera
frame into a goal image, and the psix VLA turns the task, camera frame, robot state and
goal image into hand joints, a SONIC body token and neck angles. The client sends the
token and hands to the SONIC deploy and the neck angles to the camera/neck server.

| Option | Use when | Camera | State / action |
|---|---|---|---|
| default | G1 with the 2-DoF neck | ZED Mini on the neck (`zed_server.py`) | 45 / 80 |
| `--no-neck` | G1 without the neck | onboard RealSense | 43 / 78 |
| `--no-hlp` | no HLP server; the WM gets the whole task sentence | | |

The VLA checkpoint must match the robot; the client checks this against the server's
`/info` at start.

## Services and addresses

| Service | Default address | Flag |
|---|---|---|
| psix VLA server (`serve_psix` in the psi repo) | `127.0.0.1:8014` | `--host`, `--port` |
| HLP server | `127.0.0.1:8015` | `--hlp-host`, `--hlp-port` |
| WM server | `192.168.123.240:8016` | `--wm-host`, `--wm-port` |
| SONIC deploy (`gear_sonic_deploy/deploy.sh --input-type zmq`), robot state | `127.0.0.1:5557` | `--robot-host` |
| Camera server on the G1 | `tcp://192.168.123.164:5558` | `--camera` |
| Neck state on the G1 (neck only) | `tcp://192.168.123.164:5560` | `--neck-state` |

The client binds 5556 (token + hands to the deploy) and 5570 (neck command).
`192.168.123.164` is the G1 onboard computer on its default network; machines that talk
to the G1 need a static address in `192.168.123.0/24`. Set the flags for wherever your
VLA, HLP and WM servers run.

The camera server replies to `get_frame` with a multipart message whose first part is
the RGB JPEG: `gear_sonic/scripts/zed_server.py` with the neck, any RealSense server with
the same reply without it. Camera, neck and deploy setup: [README_PSI_SONIC.md](README_PSI_SONIC.md).

## Setup

```bash
pip install numpy opencv-python pyzmq msgpack requests websocket-client onnxruntime
python download_from_hf.py   # SONIC models, including the encoder the client uses
```

The client encodes the measured pose with the SONIC encoder (to start the first action
chunk and to hold the robot in place). The encoder follows the checkpoint's body token:
`action.body_token_v1_1` uses `policy/sonic_v1_1`, `action.body_token` uses
`policy/release`. Run the deploy with the same SONIC model.

## Run

1. G1: start the camera server (with the neck:
   `python zed_server.py --enable-neck-motor --pose-zmq tcp://<client-ip>:5570`).
2. Start the VLA server (`CHECKPOINT_DIR=<run dir> bash scripts/deploy/serve_psix_rtc.sh`
   in the psi repo), the HLP server and the WM server.
3. Start the deploy with `--input-type zmq` and confirm.
4. Start the client. Without `--real` it runs everything but publishes no command:
   ```bash
   ./run_psix_client.sh --prompt "Pick up banana and place it into the white basket."
   ./run_psix_client.sh --real --prompt "Pick up banana and place it into the white basket."
   ```
5. In the deploy terminal press `]` (the controller starts), then `Enter` (follow the client).

Write the task exactly as in the training data. The client uses RTC streaming over `/ws`,
or HTTP chunks over `/act` when the VLA server runs with `RTC_MODE=off`.

Keys in the client terminal (HLP only): `Enter` lets the HLP continue after a pause or
after it reports the task done, `p` goes back one step, `i` types a step for the HLP,
`q` quits. Without the HLP, quit with Ctrl-C.

To stop: `Enter` in the deploy terminal, quit the client, `o` in the deploy terminal.

## Safety

- An action is executed only if it was computed for the current step. On a step change
  the robot holds until the WM goal for the new step arrives.
- Holding means sending the encoder token of the measured pose with the last hand
  (and neck) command.
- The robot holds when the state, camera, neck or last observation is older than
  `--obs-stale-timeout` (0.5 s), when there is no WM goal, or when the WM goal is older
  than `--wm-goal-hard-age` (30 s). The deploy is stopped when no new action arrives for
  `--action-stale-timeout` (0.5 s).

## Output

Each run writes to `~/psix_rollouts/<time>/` (or `--output-dir`): `client.log`,
`events.jsonl`, `run_manifest.json` (arguments, VLA `/info`, git commit),
`requests/{hlp,wm}/` (every request with the camera frame sent and the goal returned)
and `rollout/` (state and action arrays, `ego.avi` at 10 fps).
