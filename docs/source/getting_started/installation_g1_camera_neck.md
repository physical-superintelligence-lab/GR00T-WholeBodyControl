# Installation (G1 Camera and Neck)

This guide sets up the Python environment on the G1 onboard computer that runs
`gear_sonic/scripts/zed_server.py`: the head ZED camera and the 2-DOF Dynamixel
neck. The desktop side (C++ deploy, teleop, recording) is covered in
[Installation (Deployment)](installation_deploy.md).

`zed_server.py` serves:

| Port | Direction | Content |
|---|---|---|
| 5555 | PUB | Camera frames for `run_data_exporter.py` (msgpack) |
| 5558 | REP | Camera frames for inference clients, reply `[rgb_jpeg, stereo_jpeg]` |
| 5559 | PUB | Live-view stream |
| 5560 | PUB | Measured neck `[yaw, pitch]` in radians (motor frame) |
| 5570 | SUB (connects to `--pose-zmq`) | Neck command `[yaw, pitch]` in radians, JSON |

```{note}
This guide has not been re-run on a robot. Steps marked **unverified** come
from the previous neck setup (`realsense_server.py`) or were not checked
against the robot's installed environment.
```

## Hardware

- G1 onboard Orin (aarch64), JetPack 6.x / L4T 36.5 (**unverified**).
- ZED Mini on USB 3.
- U2D2 USB adapter on `/dev/ttyUSB0`, two daisy-chained Dynamixel X-series
  motors (Protocol 2.0, 2,000,000 baud):
  - ID 0: yaw, zero tick 1897, limit ±60°.
  - ID 1: pitch, zero tick 3275, limit ±45°.

The motor settings are constants at the top of `zed_server.py`
(`NECK_*`); edit them there to recalibrate.

## 1. Python environment

Create a Python 3.10 environment named `zed` (conda or micromamba):

```bash
conda create -n zed python=3.10 -y
conda activate zed
pip install numpy opencv-python pyzmq msgpack dynamixel-sdk
```

## 2. ZED SDK and pyzed

Download the ZED SDK for your JetPack / L4T version from
[stereolabs.com/developers/release](https://www.stereolabs.com/developers/release)
(the previous setup used ZED SDK 5.2.3 for L4T 36.5, **unverified**). Install
it, then install the Python API into the active environment:

```bash
chmod +x ZED_SDK_Tegra_L4T*.zstd.run
./ZED_SDK_Tegra_L4T*.zstd.run          # follow the interactive prompts
conda activate zed
cd /usr/local/zed
python get_python_api.py
```

## 3. Serial port permission

The neck motors are on `/dev/ttyUSB0`. Either add the user to `dialout` once
(log out and back in to apply):

```bash
sudo usermod -aG dialout $USER
```

or grant access after each boot:

```bash
sudo chmod 777 /dev/ttyUSB0
```

## 4. Verify

Check the imports:

```bash
conda activate zed
python -c "import pyzed.sl, msgpack, dynamixel_sdk, zmq, cv2; print('ok')"
```

Check that both motors answer:

```bash
python -c "
from dynamixel_sdk import PortHandler, PacketHandler
p = PortHandler('/dev/ttyUSB0'); p.openPort(); p.setBaudRate(2_000_000)
pk = PacketHandler(2.0)
for i in (0, 1):
    _, res, _ = pk.ping(p, i)
    print(f'ID {i}: {\"ok\" if res == 0 else \"FAIL\"}')
p.closePort()
"
```

Both IDs should print `ok`.

## 5. Run

Copy `gear_sonic/scripts/zed_server.py` to the G1; it does not import
`gear_sonic`. Lab defaults are G1 `192.168.123.164` and desktop
`192.168.123.222`; adjust them for your network.

```bash
conda activate zed
python zed_server.py --enable-neck-motor --pose-zmq tcp://192.168.123.222:5570
```

`--pose-zmq` points at the desktop process that publishes the neck command
(the teleop server or a policy client). Ports 5555, 5558, 5559 and 5560 are on
by default. On start the server prints `[ZED] Started: resolution=vga fps=30`
and `[Neck] Started: /dev/ttyUSB0@2000000 IDs 0/1 ...`. Without the neck, drop
both neck flags.

Stop with `Ctrl-C`.

## Optional: Pico VR stream (`--enable-pico` only)

The steps below are needed only for `--enable-pico`, which streams the ZED
stereo image to a Pico headset over GStreamer. Policy inference and data
recording do not need them; without GStreamer the server disables the Pico
stream and the rest keeps running.

System packages:

```bash
sudo apt install -y \
    gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    libgirepository1.0-dev libcairo2-dev pkg-config
```

PyGObject in the environment. The previous setup pinned `PyGObject<3.51`
because PyGObject 3.51+ needs GLib 2.80 (`girepository-2.0`); keep the pin if
pip fails with `Dependency 'girepository-2.0' is required but not found`:

```bash
conda activate zed
pip install pycairo "PyGObject<3.51"
```

Conda ships its own `libffi`, which masks the system one that GStreamer's
`libgobject` links against (symptom: `undefined symbol: ffi_type_uint32`).
Preload the system library before starting the server (**unverified**: the
file name depends on the OS release, e.g. `libffi.so.7` or `libffi.so.8`):

```bash
export LD_PRELOAD=/lib/aarch64-linux-gnu/libffi.so.7
```

Verify:

```bash
python -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init([]); print('Gst ok')"
```

Then add `--enable-pico --pico-ip <pico-ip>` to the run command.
