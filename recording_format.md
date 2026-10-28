# 录制数据格式（LeRobot v2.1）

由 `gear_sonic/scripts/run_data_exporter.py` 录制。G1 29 自由度 + Dex3 双手 + 2 自由度脖子，ZED 头部相机。

## 目录结构

```
<dataset>/
├── meta/
│   ├── info.json            # 版本、fps、特征定义（dtype / shape / names）、录制时的 deploy 配置
│   ├── modality.json        # 各特征在向量里的切片（state / action / video / annotation 分组）
│   ├── episodes.jsonl       # 每条 episode：index、任务、帧数
│   ├── episodes_stats.jsonl # 每条 episode 的统计量
│   └── tasks.jsonl          # 任务文本（--task-prompt）
├── data/chunk-000/episode_000000.parquet          # 每帧一行
└── videos/chunk-000/observation.images.ego_view/episode_000000.mp4
```

- `codebase_version`: `v2.1`，每个 chunk 最多 1000 条 episode。
- 帧率 30 fps（`--data-collection-frequency 30`，与 ZED 一致），`timestamp` 间隔 1/30 s。
- 视频：H.264，yuv420p，672x376（ZED VGA 原始分辨率，无 padding），帧数与 parquet 行数一致。

## 每帧字段

### 观测（observation.*）

| 字段 | 类型 | 维度 | 说明 |
|---|---|---|---|
| `observation.images.ego_view` | video | 376x672x3 | ZED 左目 RGB |
| `observation.state` | float64 | 43 | 实测关节角，顺序见下 |
| `observation.eef_state` | float64 | 14 | 左/右手腕位置 (3) + 四元数 wxyz (4)，由 FK 计算 |
| `observation.root_orientation` | float64 | 4 | IMU 基座四元数 wxyz |
| `observation.projected_gravity` | float64 | 3 | 基座坐标系下的重力方向 |
| `observation.cpp_rotation_offset` | float64 | 4 | deploy 的参考根旋转偏移 wxyz |
| `observation.init_base_quat` | float64 | 4 | 朝向初始化时的基座四元数 wxyz |
| `observation.neck_state` | float64 | 2 | 脖子电机实测 [yaw, pitch]，弧度 |

`observation.state` / `action.wbc` 的 43 维顺序（`modality.json` 的分组）：

| 切片 | 分组 | 关节 |
|---|---|---|
| 0:6 | left_leg | hip pitch/roll/yaw, knee, ankle pitch/roll |
| 6:12 | right_leg | 同上 |
| 12:15 | waist | yaw, roll, pitch |
| 15:22 | left_arm | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| 22:29 | left_hand | index 0/1, middle 0/1, thumb 0/1/2 |
| 29:36 | right_arm | 同 left_arm |
| 36:43 | right_hand | 同 left_hand |

### 动作（action.*）

| 字段 | 类型 | 维度 | 说明 |
|---|---|---|---|
| `action.wbc` | float64 | 43 | WBC 策略上一步输出的关节目标，顺序同 `observation.state` |
| `action.motion_token` | float64 | 64 | 编码器输出的运动 token（回放 / 推理直接发这个）|
| `action.neck` | float64 | 2 | 脖子命令 [yaw, pitch]，弧度（人头角度 × 1.5）|

### 遥操作输入（teleop.*）

| 字段 | 类型 | 维度 | 说明 |
|---|---|---|---|
| `teleop.stream_mode` | int32 | 1 | 0 OFF，1 POSE，2 PLANNER，3 PLANNER_FROZEN_UPPER_BODY，4 POSE_PAUSE，5 PLANNER_VR_3PT |
| `teleop.smpl_joints` | float32 | 72 | SMPL 24 关节位置 (24x3) |
| `teleop.smpl_pose` | float32 | 63 | SMPL 21 关节轴角 (21x3) |
| `teleop.body_quat_w` | float32 | 4 | 人体根朝向 wxyz |
| `teleop.target_body_orientation` | float32 | 6 | 目标身体朝向（旋转矩阵前两列）|
| `teleop.left_hand_joints` / `right_hand_joints` | float32 | 7 | 发给 Dex3 的手关节命令（Pico 扳机）|
| `teleop.left_wrist_joints` / `right_wrist_joints` | float32 | 3 | 手腕 roll / pitch / yaw |
| `teleop.smpl_frame_index` | int64 | 1 | 遥操作流的帧号 |
| `teleop.delta_heading` | float64 | 1 | 朝向增量 |
| `teleop.planner_mode` | int32 | 1 | planner 运动模式（0 = Idle）|
| `teleop.planner_movement` / `planner_facing` | float32 | 3 | planner 移动方向 / 朝向 |
| `teleop.planner_speed` / `planner_height` | float32 | 1 | -1 表示默认 |
| `teleop.vr_3pt_position` | float32 | 9 | 左腕、右腕、颈 位置 |
| `teleop.vr_3pt_orientation` | float32 | 18 | 左腕、右腕、颈 朝向（各 6 维）|

### 索引

`timestamp`（秒）、`frame_index`（episode 内帧号）、`episode_index`、`index`（全局帧号）、`task_index`。

## 注意

- SMPL 相关字段只在 POSE / POSE_PAUSE 且数据新鲜（< 100 ms）时有值，其余帧为 0（`body_quat_w` 为单位四元数）。
- planner 相关字段只在 PLANNER_VR_3PT 时有值，其余为默认值。
- `meta/info.json` 的 `script_config` 记录了录制时 deploy 的模型路径（如 `policy/sonic_v1_1/...`）；回放 `action.motion_token` 必须用同一个模型。
- `eccbd1d` 之前录制的数据：`action.neck` 在头部相对胸部转过 90° 时会在 ±π 附近翻转（乘 1.5 后可到 ±4.7 rad），训练前需过滤或修正；`observation.neck_state` 不受影响。`eccbd1d` 起已修复。
