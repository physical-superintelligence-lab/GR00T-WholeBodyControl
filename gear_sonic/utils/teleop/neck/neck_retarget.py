"""Pico body tracking -> 2-DOF robot neck angles.

Algorithm adapted from GMR (https://github.com/YanjieZe/GMR,
general_motion_retargeting/neck_retarget.py), MIT License,
Copyright 2025 Yanjieze.

The head rotation is taken relative to the Spine3 joint, so torso lean does
not move the neck; only head rotation does.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# Joint indices in the XRoboToolkit body pose array (SMPL joint order).
SPINE3_INDEX = 9
HEAD_INDEX = 15


def pico_head_to_neck(body_poses_np):
    """Return (neck_yaw, neck_pitch) in radians, or None on invalid input.

    Args:
        body_poses_np: XRoboToolkit body pose array, one row per joint as
            [x, y, z, qx, qy, qz, qw].
    """
    if body_poses_np is None or body_poses_np.shape[0] <= HEAD_INDEX:
        return None
    quats = body_poses_np[[SPINE3_INDEX, HEAD_INDEX], 3:7]
    norms = np.linalg.norm(quats, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms < 1e-6):
        return None
    spine_rotation, head_rotation = R.from_quat(quats)

    # Head rotation expressed in the spine frame.
    relative_rotation = spine_rotation.inv() * head_rotation
    roll, pitch, _yaw = relative_rotation.as_euler("xyz")
    return -pitch, roll
