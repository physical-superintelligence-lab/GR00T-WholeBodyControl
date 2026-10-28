# 遥操作 + 脖子（ZED）操作手册

G1 + Dex3 + 2 自由度脖子（ZED 头部相机）的 Pico 遥操作与数据录制。

- 机器人 IP：`192.168.123.164`；桌面 IP：`192.168.123.222`（按实际调整）。
- 端口：5555 相机（录制用）/ 5556 身体姿态 + 模式 / 5557 WBC 状态 /
  5558 推理取图（REQ/REP）/ 5559 实时画面 / 5560 脖子电机状态 / 5570 脖子命令。
- 录制用 SONIC 原生的 `run_data_exporter.py`（LeRobot 格式），多录脖子两项。

---

## 1. G1 板载

开放电机串口、强制加载系统 libffi（GStreamer 需要），启动相机 + 脖子 server：

```bash
sudo chmod 777 /dev/ttyUSB0
export LD_PRELOAD=/lib/aarch64-linux-gnu/libffi.so.7

conda activate sonic
cd ~/PSI-SONIC     # 按实际路径调整
python zed_server.py \
    --enable-pico --pico-ip 192.168.0.114 \
    --enable-neck-motor \
    --pose-zmq tcp://192.168.123.222:5570
```

依赖：`pyzed`、`msgpack`、`dynamixel-sdk`、GStreamer + PyGObject（`--enable-pico` 时）。

---

## 2. 桌面端

### 2.1 XRoboToolKit 守护进程

```bash
sudo bash /opt/apps/roboticsservice/runService.sh
```

### 2.2 WBC deploy

C++ 改动（进出遥操作平滑、手默认张开）需要先重新 build。

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real
```

v1.1（模型 onnx 已软链接自内部仓库 `GR00T-WholeBodyControl/gear_sonic_deploy/policy/sonic_v1_1/`）：

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --cp policy/sonic_v1_1/model \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --input-type zmq_manager real
```

### 2.3 身体 / 手 / 脖子 server

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

在这个终端里操作（单键，按下即生效，不用回车）：

| 按键 | 作用 |
|---|---|
| `]` | 启动策略，进入 PLANNER（同时做 VR 校准，人先站好） |
| Enter | POSE（全身跟随）和 PLANNER 之间切换（enable / disable） |
| `s` 或 `q` | 录制开始 / 停止并保存（切换，以录制程序终端的提示为准） |
| `d` | 丢弃当前录制 |
| `r` | 脖子保持 (0, 0) 开 / 关 |

停止：在 deploy 终端按 `o`。

- 手：Pico trigger 控制 Dex3 开合。
- 脖子：策略启动后（不是 OFF）跟随头部，之前保持 (0, 0)。
- 不用脖子：加 `--no_neck`（不发脖子数据）。
- 脖子角度 = 人头角度 × 1.5（`NeckPublisher.RETARGET_SCALE`）。

### 2.4 数据录制

第一次先建环境：`bash install_scripts/install_data_collection.sh`。

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the cup" \
    --data-collection-frequency 30 \
    --camera-host 192.168.123.164 \
    --ego-view-height 376 --ego-view-width 672 \
    --record-neck \
    --neck-state-zmq-host 192.168.123.164
```

不用脖子（配合 `--no_neck`）：去掉 `--record-neck --neck-state-zmq-host ...`。

- 录制用 2.3 终端里的 `s`/`q`（切换）和 `d`（丢弃）控制，或用手柄。
- `--data-collection-frequency 30` 跟 ZED 的 30 fps 一致（默认 50 会重复帧）。
- `--ego-view-*` 要和 ZED 分辨率一致（VGA 是 672x376）。
- 输出在 `outputs/<dataset-name>/`，脖子是 `observation.neck_state`（实际角度）和
  `action.neck`（命令），都是 `[yaw, pitch]` 弧度。

### 2.5 实时 ZED 画面（可选）

```bash
python gear_sonic/scripts/test_viewer.py --server 192.168.123.164 --port 5559 --show-stereo
```

### 2.6 回放录制数据

deploy 用 `--input-type zmq`（不要同时开 2.3 的 pico_manager，端口冲突），模型必须和录制时一致
（token 是该模型编码器的输出）。录制用的 v1.1：

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --cp policy/sonic_v1_1/model \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --input-type zmq real
```

另开终端：

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/replay_episode.py --dataset outputs/<dataset-name> --episode 0
```

1. 先启动脚本（会等 deploy 状态），再在 deploy 终端按 `]` 启动策略；脚本打印 holding 后，
   deploy 终端按 Enter 开 ZMQ 流（脚本一直发站立 token，机器人保持站立）。
2. 脚本终端：`p` 开始回放（2 秒平滑过渡到第一帧，播完停在最后一帧；再按 `p` 重放），`q` 退出。
3. deploy 终端：Enter 关流（平滑回站立），`o` 停止。

- 回放 token + 双手 + 脖子；不发脖子加 `--no-neck`。
- 过渡时长 `--blend-seconds`（默认 2）。
