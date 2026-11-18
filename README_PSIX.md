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

The client checks at start that the VLA checkpoint matches the robot (server `/info`).

## Services and addresses

| Service | Default address | Flag |
|---|---|---|
| psix VLA server (`serve_psix` in the psi repo) | `127.0.0.1:8014` | `--host`, `--port` |
| HLP server | `127.0.0.1:8015` | `--hlp-host`, `--hlp-port` |
| WM server | `192.168.123.240:8016` | `--wm-host`, `--wm-port` |
| SONIC deploy, robot state | `127.0.0.1:5557` | `--robot-host` |
| Camera server on the G1 | `tcp://192.168.123.164:5558` | `--camera` |
| Neck state on the G1 (neck only) | `tcp://192.168.123.164:5560` | `--neck-state` |

The client binds 5556 (token + hands to the deploy) and 5570 (neck command).
`192.168.123.164` is the G1 onboard computer on its default network; machines that talk
to the G1 need a static address in `192.168.123.0/24`. The camera server must reply to
`get_frame` with the RGB JPEG as the first part: `zed_server.py` with the neck, any RealSense
server without it. Port details: [README_PSI_SONIC.md](README_PSI_SONIC.md).

## Setup

```bash
pip install numpy opencv-python pyzmq msgpack requests websocket-client onnxruntime
python download_from_hf.py               # SONIC v1.0 (release) models + planner
python download_from_hf.py --sonic-v1-1  # SONIC v1.1 models
```

The client runs the SONIC encoder of the checkpoint's body token (`action.body_token` ->
v1.0, `action.body_token_v1_1` -> v1.1) to start the first action chunk and to hold the robot.

## Run

1. G1: start the camera and neck server (copy `gear_sonic/scripts/zed_server.py` to the G1;
   environment in `docs/source/getting_started/installation_g1_camera_neck.md`):
   ```bash
   conda activate zed
   python zed_server.py --enable-neck-motor --pose-zmq tcp://<client-ip>:5570
   ```
   Without the neck, start a RealSense server that replies on 5558 instead.
2. Desktop, optional: camera viewer (`zed_server.py` streams it on 5559):
   ```bash
   python gear_sonic/scripts/test_viewer.py --server 192.168.123.164
   ```
3. Start the VLA server (`CHECKPOINT_DIR=<run dir> bash scripts/deploy/serve_psix_rtc.sh`
   in the psi repo), the HLP server and the WM server.
4. Desktop: start the deploy with the SONIC model of the checkpoint, and answer `Y` at
   `Proceed with deployment? [Y/n]`:
   ```bash
   cd gear_sonic_deploy
   source scripts/setup_env.sh
   # SONIC v1.0 (action.body_token)
   ./deploy.sh --input-type zmq real
   # SONIC v1.1 (action.body_token_v1_1)
   ./deploy.sh --input-type zmq \
     --cp policy/sonic_v1_1/model \
     --obs-config policy/sonic_v1_1/observation_config.yaml \
     real
   ```
5. Desktop: start the client. Without `--real` it runs everything but publishes no command:
   ```bash
   ./run_psix_client.sh --prompt "Pick up banana and place it into the white basket."
   ./run_psix_client.sh --real --prompt "Pick up banana and place it into the white basket."
   ```
6. In the deploy terminal press `]` (the controller starts), then `Enter` (follow the client).

To stop: `Enter` in the deploy terminal, quit the client, `o` in the deploy terminal.

Write the task exactly as in the training data. The client uses RTC streaming over `/ws`,
or HTTP chunks over `/act` when the VLA server runs with `RTC_MODE=off`.

Keys in the client terminal (HLP only): `Enter` lets the HLP continue after a pause or
after it reports the task done, `p` goes back one step, `i` types a step for the HLP,
`q` quits. Without the HLP, quit with Ctrl-C. When the HLP reports the task done, the
robot keeps running the current step.

## Safety

- At an HLP step change the robot keeps the previous step's goal until the new one lands;
  after 2 s without it, the robot holds and only actions computed with the new goal run.
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
