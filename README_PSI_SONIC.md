# PSI-SONIC

## Policy inference on the real G1

A token policy predicts Dex3 hand joints, neck angles and a 64-D SONIC motion
token. A policy client on the desktop sends the token and hand joints to the
SONIC deploy, which decodes the token into whole-body joint targets, and sends
the neck angles to the G1.

| Component | Runs on | Program |
|---|---|---|
| Camera and neck service | G1 onboard computer | `gear_sonic/scripts/zed_server.py` |
| Whole-body controller | Desktop | `gear_sonic_deploy/deploy.sh --input-type zmq` |
| Policy client | Desktop | Your own client (interface below) |

Desktop `192.168.123.222`, G1 `192.168.123.164`, wired Ethernet.

| Port | Bound by | Used by | Content |
|---|---|---|---|
| 5556 | policy client (PUB) | deploy | Protocol v4, topic `pose`: `token_state` (64) + `left_hand_joints` / `right_hand_joints` (7 + 7) |
| 5557 | deploy (PUB) | policy client | Robot state, topic `g1_debug` (msgpack: `body_q_measured`, `base_quat_measured`, `left_hand_q_measured`, `right_hand_q_measured`, ...) |
| 5558 | `zed_server.py` (REP) | policy client | Request `get_frame`, reply `[rgb_jpeg, stereo_jpeg]`; ZED left image 672x376, BGR JPEG |
| 5560 | `zed_server.py` (PUB) | policy client | Measured neck `[yaw, pitch]` in radians, JSON, motor frame |
| 5570 | policy client (PUB) | `zed_server.py` | Neck command `[yaw, pitch]` in radians, JSON |

Protocol v4 messages are built with `pack_pose_message(..., version=4)` from
`gear_sonic/utils/teleop/zmq/zmq_planner_sender.py`.

### 1. G1 onboard: camera and neck

Copy `gear_sonic/scripts/zed_server.py` to the G1 (it does not import
`gear_sonic`); the environment setup is in
`docs/source/getting_started/installation_g1_camera_neck.md`.

```bash
sudo chmod 777 /dev/ttyUSB0
conda activate zed
python zed_server.py --enable-neck-motor --pose-zmq tcp://192.168.123.222:5570
```

`--pose-zmq` connects to the policy client's neck publisher on the desktop.
The server clamps the neck command to yaw ±60° and pitch ±45°.

### 2. Desktop: whole-body controller

Release SONIC model:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq real
```

SONIC v1.1:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq \
  --cp policy/sonic_v1_1/model \
  --obs-config policy/sonic_v1_1/observation_config.yaml \
  real
```

Answer `Y` (or Enter) at `Proceed with deployment? [Y/n]`. The deploy ignores
`Ctrl-C`; it is stopped with `o` (see below). Use the SONIC model the policy
was trained with: the motion token is only valid for that model. If the client
runs the SONIC encoder itself, use the matching encoder
(`policy/release/model_encoder.onnx` or `policy/sonic_v1_1/model_encoder.onnx`).

### 3. Desktop: policy client

Start your policy server (if separate) and then the client. The client
connects to 5557 on `localhost`, 5558 and 5560 on the G1, and binds 5556 and
5570. Stop any teleop server (`pico_manager_thread_server.py`) first: it binds
the same ports.

### 4. Start and stop

All keys go to the deploy terminal and act immediately.

1. G1: start `zed_server.py`.
2. Desktop: start `deploy.sh` and confirm.
3. Desktop: start the policy client.
4. Press `]`: the controller starts and the robot stands.
5. Press `Enter`: the deploy switches to the ZMQ stream and the robot follows
   the policy.

The deploy ignores start/stop command messages in `zmq` mode, so `]` and
`Enter` are both required. Before `Enter` the robot holds frame 0 of the
loaded reference motion; do not press `t`, which plays that motion.

To stop:

1. Press `Enter`: the deploy leaves the ZMQ stream and returns to the
   standing reference motion.
2. Stop the policy client.
3. Press `o` to stop the controller.
4. Stop `zed_server.py` with `Ctrl-C`.
