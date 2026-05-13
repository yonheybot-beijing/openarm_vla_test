#!/usr/bin/env python

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


DEFAULT_MODEL_PATH = Path("smolvla_sponge_20k_pretrained_model/pretrained_model")
RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0

OPENARM_STATE_ACTION_NAMES = [
    f"joint_{joint}.{field}"
    for joint in range(1, 8)
    for field in ("pos", "vel", "torque")
] + ["gripper.pos", "gripper.vel", "gripper.torque"]

OPENARM_ROS_JOINT_NAMES = {
    "single": [f"openarm_joint{i}" for i in range(1, 8)],
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}

OPENARM_GRIPPER_JOINT_NAMES = {
    "single": "openarm_finger_joint1",
    "left": "openarm_left_finger_joint1",
    "right": "openarm_right_finger_joint1",
}

OPENARM_TRAJECTORY_ACTIONS = {
    "single": "/joint_trajectory_controller/follow_joint_trajectory",
    "left": "/left_joint_trajectory_controller/follow_joint_trajectory",
    "right": "/right_joint_trajectory_controller/follow_joint_trajectory",
}

OPENARM_GRIPPER_ACTIONS = {
    "single": "/gripper_controller/gripper_cmd",
    "left": "/left_gripper_controller/gripper_cmd",
    "right": "/right_gripper_controller/gripper_cmd",
}


def linear_map(value: float, src_a: float, src_b: float, dst_a: float, dst_b: float) -> float:
    if abs(src_b - src_a) < 1e-8:
        return float(dst_a)
    ratio = (float(value) - src_a) / (src_b - src_a)
    return float(dst_a + ratio * (dst_b - dst_a))


def infer_gripper_open_close(args: argparse.Namespace) -> tuple[float, float]:
    if args.gripper_open_position is not None and args.gripper_close_position is not None:
        return float(args.gripper_open_position), float(args.gripper_close_position)

    low = float(args.gripper_min_position)
    high = float(args.gripper_max_position)
    if low < 0 <= high:
        return low, high
    return high, low


def normalize_proxy_env() -> None:
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.startswith("socks://"):
            os.environ[key] = value.replace("socks://", "socks5://", 1)


def select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def image_msg_to_rgb(msg) -> np.ndarray:
    encoding = msg.encoding.lower()
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
        "8uc4": 4,
    }

    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise RuntimeError(f"Unsupported ROS image encoding: {msg.encoding}")

    row_bytes = msg.width * channels
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = data[:, :row_bytes].reshape(msg.height, msg.width, channels)

    if encoding in {"rgb8", "8uc3"}:
        return image.copy()
    if encoding == "bgr8":
        return image[:, :, ::-1].copy()
    if encoding == "rgba8":
        return image[:, :, :3].copy()
    if encoding == "bgra8":
        return image[:, :, 2::-1].copy()
    if encoding in {"mono8", "8uc1"}:
        return np.repeat(image, 3, axis=2).copy()
    if encoding == "8uc4":
        return image[:, :, :3].copy()

    raise RuntimeError(f"Unsupported ROS image encoding: {msg.encoding}")


def image_payload_to_rgb(payload: dict) -> np.ndarray:
    encoding = payload["encoding"].lower()
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
        "8uc4": 4,
    }

    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise RuntimeError(f"Unsupported ROS image encoding: {payload['encoding']}")

    height = int(payload["height"])
    width = int(payload["width"])
    step = int(payload["step"])
    row_bytes = width * channels
    data = np.frombuffer(base64.b64decode(payload["data_b64"]), dtype=np.uint8).reshape(height, step)
    image = data[:, :row_bytes].reshape(height, width, channels)

    if encoding in {"rgb8", "8uc3"}:
        return image.copy()
    if encoding == "bgr8":
        return image[:, :, ::-1].copy()
    if encoding == "rgba8":
        return image[:, :, :3].copy()
    if encoding == "bgra8":
        return image[:, :, 2::-1].copy()
    if encoding in {"mono8", "8uc1"}:
        return np.repeat(image, 3, axis=2).copy()
    if encoding == "8uc4":
        return image[:, :, :3].copy()

    raise RuntimeError(f"Unsupported ROS image encoding: {payload['encoding']}")


def image_to_tensor(image: Image.Image | np.ndarray | None, fallback_hw: tuple[int, int]) -> torch.Tensor:
    if image is None:
        height, width = fallback_hw
        image = Image.new("RGB", (width, height), color=(0, 0, 0))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class VLAVisualizationServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.status = {
            "mode": "starting",
            "step": 0,
            "max_steps": 0,
            "arm_side": "",
            "task": "",
            "execute": False,
            "image": "waiting",
            "state": [],
            "target": [],
            "gripper": None,
            "raw_gripper": None,
            "latency_ms": None,
            "message": "",
        }
        self.server = ThreadingHTTPServer((host, port), self._make_handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def update_image(self, image: Image.Image | np.ndarray, image_info: str = "") -> None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=82)

        with self.lock:
            self.latest_jpeg = buffer.getvalue()
            if image_info:
                self.status["image"] = image_info

    def update_status(self, **kwargs) -> None:
        with self.lock:
            self.status.update(kwargs)

    def _snapshot(self) -> tuple[bytes | None, dict]:
        with self.lock:
            return self.latest_jpeg, dict(self.status)

    def _make_handler(self):
        visualizer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in {"/", "/index.html"}:
                    body = self._index_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/status.json":
                    _, status = visualizer._snapshot()
                    body = json.dumps(status, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                try:
                    while True:
                        jpeg, status = visualizer._snapshot()
                        if jpeg is None:
                            jpeg = self._placeholder_jpeg(str(status.get("image", "waiting")))

                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def log_message(self, fmt: str, *args) -> None:
                return

            @staticmethod
            def _placeholder_jpeg(text: str) -> bytes:
                image = Image.new("RGB", (640, 360), color=(24, 24, 24))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=70)
                return buffer.getvalue()

            @staticmethod
            def _index_html() -> str:
                return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SmolVLA OpenArm Monitor</title>
  <style>
    body { margin:0; background:#101214; color:#e8ecef; font-family:Arial,sans-serif; }
    header { padding:12px 16px; border-bottom:1px solid #2b3035; display:flex; gap:16px; align-items:center; }
    main { display:grid; grid-template-columns:minmax(360px, 1fr) 420px; gap:16px; padding:16px; }
    img { width:100%; max-height:calc(100vh - 100px); object-fit:contain; background:#050607; }
    .panel { border:1px solid #2b3035; border-radius:6px; padding:12px; background:#171a1d; }
    .row { display:flex; justify-content:space-between; gap:12px; padding:6px 0; border-bottom:1px solid #262b30; }
    .label { color:#9aa4ad; }
    pre { white-space:pre-wrap; overflow-wrap:anywhere; font-size:13px; line-height:1.4; }
    .pill { padding:3px 8px; border-radius:999px; background:#26313a; }
    .exec { background:#57302c; color:#ffd6ce; }
    .dry { background:#243b2c; color:#cff5d8; }
  </style>
</head>
<body>
  <header>
    <strong>SmolVLA OpenArm Monitor</strong>
    <span id="mode" class="pill">starting</span>
    <span id="exec" class="pill">dry-run</span>
  </header>
  <main>
    <section><img src="/stream.mjpg"></section>
    <section class="panel">
      <div class="row"><span class="label">Step</span><span id="step"></span></div>
      <div class="row"><span class="label">Arm</span><span id="arm"></span></div>
      <div class="row"><span class="label">Task</span><span id="task"></span></div>
      <div class="row"><span class="label">Latency</span><span id="latency"></span></div>
      <div class="row"><span class="label">Image</span><span id="image"></span></div>
      <h3>Current State</h3><pre id="state"></pre>
      <h3>Target Action</h3><pre id="target"></pre>
      <h3>Message</h3><pre id="message"></pre>
    </section>
  </main>
  <script>
    function fmt(v) {
      if (Array.isArray(v)) return JSON.stringify(v.map(x => Number(x).toFixed(4)));
      if (v === null || v === undefined) return "";
      return String(v);
    }
    async function refresh() {
      const r = await fetch('/status.json', {cache:'no-store'});
      const s = await r.json();
      document.getElementById('mode').textContent = s.mode || '';
      const ex = document.getElementById('exec');
      ex.textContent = s.execute ? 'execute' : 'dry-run';
      ex.className = 'pill ' + (s.execute ? 'exec' : 'dry');
      document.getElementById('step').textContent = `${s.step || 0}/${s.max_steps || 0}`;
      document.getElementById('arm').textContent = s.arm_side || '';
      document.getElementById('task').textContent = s.task || '';
      document.getElementById('latency').textContent = s.latency_ms == null ? '' : `${s.latency_ms.toFixed(1)} ms`;
      document.getElementById('image').textContent = s.image || '';
      document.getElementById('state').textContent = fmt(s.state);
      const raw = s.raw_gripper == null ? '' : `\\nraw_gripper=${Number(s.raw_gripper).toFixed(4)}`;
      const grip = s.gripper == null ? '' : `\\ngripper=${Number(s.gripper).toFixed(4)}`;
      document.getElementById('target').textContent = fmt(s.target) + raw + grip;
      document.getElementById('message').textContent = s.message || '';
    }
    setInterval(refresh, 500); refresh();
  </script>
</body>
</html>"""

        return Handler


class RosBridgeClient:
    def __init__(self, base_url: str, arm_side: str):
        self.base_url = base_url.rstrip("/")
        self.arm_side = arm_side

    def _get_json(self, path: str) -> dict:
        try:
            with urlopen(f"{self.base_url}{path}", timeout=5.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"Cannot reach ROS bridge at {self.base_url}: {exc}") from exc

        if not payload.get("ok", False):
            raise RuntimeError(payload.get("error", "ROS bridge request failed"))

        return payload

    def _post_json(self, path: str, data: dict) -> dict:
        body = json.dumps(data).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=5.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"Cannot reach ROS bridge at {self.base_url}: {exc}") from exc

        if not payload.get("ok", False):
            raise RuntimeError(payload.get("error", "ROS bridge request failed"))

        return payload

    def wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error = None

        while time.monotonic() < deadline:
            try:
                self._get_json("/health")
                state = self.read_state_vector()
                image = self.read_rgb_image()
                if len(state) == 24 and image is not None:
                    return
            except Exception as exc:
                last_error = exc

            time.sleep(0.2)

        raise RuntimeError(f"ROS bridge inputs not ready: {last_error}")

    def spin_once(self, timeout_sec: float = 0.02) -> None:
        del timeout_sec

    def close(self) -> None:
        pass

    def read_state_vector(self) -> list[float]:
        payload = self._get_json(f"/state?arm_side={self.arm_side}")
        return [float(value) for value in payload["state"]]

    def read_rgb_image(self) -> Image.Image:
        payload = self._get_json("/image")
        return Image.fromarray(image_payload_to_rgb(payload), mode="RGB")

    def send_action(
        self,
        action: np.ndarray,
        *,
        max_joint_delta: float,
        gripper_effort: float,
        trajectory_duration: float,
        gripper_min_position: float,
        gripper_max_position: float,
        execute: bool,
    ) -> dict:
        return self._post_json(
            "/send_action",
            {
                "arm_side": self.arm_side,
                "action": [float(value) for value in action.tolist()],
                "max_joint_delta": max_joint_delta,
                "gripper_effort": gripper_effort,
                "trajectory_duration": trajectory_duration,
                "gripper_min_position": gripper_min_position,
                "gripper_max_position": gripper_max_position,
                "execute": execute,
            },
        )


class OpenArmSmolVLARosNode:
    def __init__(
        self,
        arm_side: str,
        image_topic: str,
        *,
        gripper_open_position: float,
        gripper_close_position: float,
        policy_gripper_open_position: float,
        policy_gripper_close_position: float,
        visualizer: VLAVisualizationServer | None = None,
    ):
        try:
            import rclpy
            from control_msgs.action import FollowJointTrajectory, GripperCommand
            from rclpy.action import ActionClient
            from sensor_msgs.msg import Image as RosImage
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectoryPoint
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 Python packages are not available in this Python environment. "
                "Use the project Python 3.10 environment and source ROS first, for example: "
                "source /opt/ros/humble/setup.bash && uv run python "
                "examples/tutorial/smolvla/smolvla_ros_control.py"
            ) from exc

        if arm_side not in OPENARM_ROS_JOINT_NAMES:
            raise ValueError(f"Unsupported arm side: {arm_side}")

        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        self.rclpy = rclpy
        self.FollowJointTrajectory = FollowJointTrajectory
        self.GripperCommand = GripperCommand
        self.JointTrajectoryPoint = JointTrajectoryPoint

        self.arm_side = arm_side
        self.image_topic = image_topic
        self.visualizer = visualizer
        self.gripper_open_position = float(gripper_open_position)
        self.gripper_close_position = float(gripper_close_position)
        self.policy_gripper_open_position = float(policy_gripper_open_position)
        self.policy_gripper_close_position = float(policy_gripper_close_position)

        self.joint_names = OPENARM_ROS_JOINT_NAMES[arm_side]
        self.gripper_joint_name = OPENARM_GRIPPER_JOINT_NAMES.get(arm_side)

        self.joint_state: dict[str, dict[str, float]] = {}
        self.latest_image = None

        self.node = rclpy.create_node(f"smolvla_{arm_side}_continuous_controller")

        self.joint_sub = self.node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
        )

        self.image_sub = self.node.create_subscription(
            RosImage,
            image_topic,
            self._image_callback,
            10,
        )

        self.trajectory_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            OPENARM_TRAJECTORY_ACTIONS[arm_side],
        )

        self.gripper_client = ActionClient(
            self.node,
            GripperCommand,
            OPENARM_GRIPPER_ACTIONS[arm_side],
        )

    def _joint_state_callback(self, msg) -> None:
        for idx, name in enumerate(msg.name):
            self.joint_state[name] = {
                "pos": msg.position[idx] if idx < len(msg.position) else 0.0,
                "vel": msg.velocity[idx] if idx < len(msg.velocity) else 0.0,
                "torque": msg.effort[idx] if idx < len(msg.effort) else 0.0,
            }

    def _image_callback(self, msg) -> None:
        self.latest_image = msg

        if self.visualizer is not None:
            try:
                self.visualizer.update_image(
                    image_msg_to_rgb(msg),
                    image_info=f"{msg.width}x{msg.height} {msg.encoding} {self.image_topic}",
                )
            except Exception as exc:
                self.node.get_logger().warning(f"Failed to update visualization image: {exc}")

    def spin_once(self, timeout_sec: float = 0.02) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def close(self) -> None:
        self.node.destroy_node()

        if self._owns_rclpy and self.rclpy.ok():
            self.rclpy.shutdown()

    def wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            self.spin_once(0.05)

            has_arm_state = all(name in self.joint_state for name in self.joint_names)
            has_gripper_state = (
                self.gripper_joint_name is None
                or self.gripper_joint_name in self.joint_state
            )

            if has_arm_state and has_gripper_state and self.latest_image is not None:
                return

        missing_arm = [name for name in self.joint_names if name not in self.joint_state]
        missing_gripper = (
            []
            if self.gripper_joint_name is None or self.gripper_joint_name in self.joint_state
            else [self.gripper_joint_name]
        )

        raise RuntimeError(
            f"ROS inputs not ready. missing_joints={missing_arm}, "
            f"missing_gripper={missing_gripper}, "
            f"has_image={self.latest_image is not None}, image_topic={self.image_topic}"
        )

    def read_state_vector(self) -> list[float]:
        values = []

        for name in self.joint_names:
            item = self.joint_state.get(name, {})
            values.extend(
                [
                    float(item.get("pos", 0.0)) * RAD_TO_DEG,
                    float(item.get("vel", 0.0)) * RAD_TO_DEG,
                    float(item.get("torque", 0.0)),
                ]
            )

        gripper = self.joint_state.get(self.gripper_joint_name, {}) if self.gripper_joint_name else {}
        gripper_pos = float(gripper.get("pos", self.gripper_close_position))
        gripper_vel = float(gripper.get("vel", 0.0))
        gripper_scale = (
            (self.policy_gripper_close_position - self.policy_gripper_open_position)
            / (self.gripper_close_position - self.gripper_open_position)
            if abs(self.gripper_close_position - self.gripper_open_position) >= 1e-8
            else 0.0
        )
        values.extend(
            [
                linear_map(
                    gripper_pos,
                    self.gripper_open_position,
                    self.gripper_close_position,
                    self.policy_gripper_open_position,
                    self.policy_gripper_close_position,
                ),
                gripper_vel * gripper_scale,
                float(gripper.get("torque", 0.0)),
            ]
        )

        if len(values) != len(OPENARM_STATE_ACTION_NAMES):
            raise RuntimeError(f"Expected 24-dim state, got {len(values)}")

        return values

    def read_rgb_image(self) -> Image.Image:
        if self.latest_image is None:
            raise RuntimeError(f"No image received from {self.image_topic}")

        return Image.fromarray(image_msg_to_rgb(self.latest_image), mode="RGB")

    def send_action(
        self,
        action: np.ndarray,
        *,
        max_joint_delta: float,
        gripper_effort: float,
        trajectory_duration: float,
        gripper_min_position: float,
        gripper_max_position: float,
        execute: bool,
    ) -> dict:
        if action.shape[0] != len(OPENARM_STATE_ACTION_NAMES):
            raise ValueError(f"Expected 24-dim OpenArm action, got {action.shape[0]}")

        raw_target_joints = [float(action[idx * 3]) for idx in range(7)]
        target_joints = [value * DEG_TO_RAD for value in raw_target_joints]

        raw_gripper_position = float(action[21])
        mapped_gripper_position = linear_map(
            raw_gripper_position,
            self.policy_gripper_open_position,
            self.policy_gripper_close_position,
            self.gripper_open_position,
            self.gripper_close_position,
        )
        gripper_position = float(np.clip(mapped_gripper_position, gripper_min_position, gripper_max_position))

        current = [self.joint_state.get(name, {}).get("pos") for name in self.joint_names]

        if max_joint_delta > 0 and all(value is not None for value in current):
            target_joints = [
                float(np.clip(target, now - max_joint_delta, now + max_joint_delta))
                for target, now in zip(target_joints, current, strict=True)
            ]

        if execute:
            if not self.trajectory_client.wait_for_server(timeout_sec=0.1):
                raise RuntimeError(
                    f"Trajectory action server not available: "
                    f"{OPENARM_TRAJECTORY_ACTIONS[self.arm_side]}"
                )

            trajectory_goal = self.FollowJointTrajectory.Goal()
            trajectory_goal.trajectory.joint_names = self.joint_names

            point = self.JointTrajectoryPoint()
            point.positions = target_joints
            point.time_from_start.sec = int(trajectory_duration)
            point.time_from_start.nanosec = int((trajectory_duration - int(trajectory_duration)) * 1e9)

            trajectory_goal.trajectory.points.append(point)
            self.trajectory_client.send_goal_async(trajectory_goal)

            if self.gripper_client.wait_for_server(timeout_sec=0.05):
                goal = self.GripperCommand.Goal()
                goal.command.position = gripper_position
                goal.command.max_effort = gripper_effort
                self.gripper_client.send_goal_async(goal)

        return {
            "execute": execute,
            "joint_positions": target_joints,
            "raw_joint_positions": raw_target_joints,
            "raw_gripper_position": raw_gripper_position,
            "mapped_gripper_position": mapped_gripper_position,
            "gripper_position": gripper_position,
        }


def make_observation(
    node,
    task: str,
    visualizer: VLAVisualizationServer | None = None,
) -> dict:
    image = node.read_rgb_image()

    if visualizer is not None:
        visualizer.update_image(image, image_info="latest policy camera frame")

    return {
        "observation.state": torch.tensor(node.read_state_vector(), dtype=torch.float32),
        "observation.images.camera1": image_to_tensor(image, (256, 256)),
        "task": task,
    }


def run_control(args: argparse.Namespace) -> None:
    normalize_proxy_env()

    device = select_device(args.device)
    model_path = str(args.model_path.expanduser())
    execute = args.execute
    gripper_open_position, gripper_close_position = infer_gripper_open_close(args)
    gripper_min_position = (
        min(gripper_open_position, gripper_close_position)
        if args.gripper_min_position is None
        else float(args.gripper_min_position)
    )
    gripper_max_position = (
        max(gripper_open_position, gripper_close_position)
        if args.gripper_max_position is None
        else float(args.gripper_max_position)
    )

    visualizer = None if args.no_visualize else VLAVisualizationServer(args.visualize_host, args.visualize_port)

    if visualizer is not None:
        visualizer.update_status(
            mode="starting",
            max_steps=args.max_steps,
            arm_side=args.arm_side,
            task=args.task,
            execute=execute,
            message="initializing ROS interface",
        )
        visualizer.start()
        print(f"VLA visualization: {visualizer.url}")

    if args.ros_interface == "direct":
        node = OpenArmSmolVLARosNode(
            args.arm_side,
            args.image_topic,
            gripper_open_position=gripper_open_position,
            gripper_close_position=gripper_close_position,
            policy_gripper_open_position=args.policy_gripper_open_position,
            policy_gripper_close_position=args.policy_gripper_close_position,
            visualizer=visualizer,
        )
        print(f"Waiting for ROS topics: /joint_states and {args.image_topic} ...")
    else:
        node = RosBridgeClient(args.bridge_url, args.arm_side)
        print(f"Waiting for ROS bridge at {args.bridge_url} ...")

    try:
        if visualizer is not None:
            visualizer.update_status(mode="waiting", message="waiting for ROS state, gripper, and camera")

        node.wait_until_ready(args.startup_timeout)

        print(f"Loading policy from {model_path} on {device} ...")

        if visualizer is not None:
            visualizer.update_status(mode="loading", message=f"loading policy on {device}")

        policy = SmolVLAPolicy.from_pretrained(model_path, cli_overrides=[f"--device={device}"])

        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            model_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        period = 1.0 / args.hz

        print(
            f"Starting {'EXECUTE' if execute else 'DRY-RUN'} loop: interface={args.ros_interface}, "
            f"arm={args.arm_side}, hz={args.hz}, max_steps={args.max_steps}, "
            f"max_joint_delta={args.max_joint_delta}, "
            f"gripper_open={gripper_open_position}, gripper_close={gripper_close_position}, "
            f"gripper_clip=[{gripper_min_position}, {gripper_max_position}], "
            f"policy_gripper_open={args.policy_gripper_open_position}, "
            f"policy_gripper_close={args.policy_gripper_close_position}"
        )

        if not execute:
            print("Dry-run is active. Add --execute only after single-step validation is safe.")

        if visualizer is not None:
            visualizer.update_status(mode="running", message="control loop started")

        for step in range(args.max_steps):
            start = time.monotonic()
            node.spin_once(0.01)

            observation = make_observation(node, args.task, visualizer)

            with torch.inference_mode():
                batch = preprocess(observation)

                if args.fresh_inference_each_step:
                    policy.reset()

                action = policy.select_action(batch)
                action = postprocess(action).squeeze(0).detach().cpu().float().numpy()

            result = node.send_action(
                action,
                max_joint_delta=args.max_joint_delta,
                gripper_effort=args.gripper_effort,
                trajectory_duration=args.trajectory_duration,
                gripper_min_position=gripper_min_position,
                gripper_max_position=gripper_max_position,
                execute=execute,
            )

            print(
                f"[{step + 1:04d}/{args.max_steps}] "
                f"joints={np.round(result['joint_positions'], 4).tolist()} "
                f"raw_gripper={result['raw_gripper_position']:.4f} "
                f"mapped_gripper={result.get('mapped_gripper_position', result['gripper_position']):.4f} "
                f"gripper={result['gripper_position']:.4f}"
            )

            if visualizer is not None:
                visualizer.update_status(
                    mode="running",
                    step=step + 1,
                    state=np.round(observation["observation.state"].detach().cpu().numpy(), 4).tolist(),
                    target=np.round(result["joint_positions"], 4).tolist(),
                    raw_gripper=float(result["raw_gripper_position"]),
                    gripper=float(result["gripper_position"]),
                    latency_ms=(time.monotonic() - start) * 1000.0,
                    message="last action computed" if not execute else "last action sent",
                )

            elapsed = time.monotonic() - start
            time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        print("Interrupted by user.")
        if visualizer is not None:
            visualizer.update_status(mode="stopped", message="interrupted by user")

    except Exception as exc:
        if visualizer is not None:
            visualizer.update_status(mode="error", message=str(exc))
        raise

    finally:
        node.close()

        if visualizer is not None:
            visualizer.update_status(mode="closed", message="control script exited")
            if args.visualize_hold_seconds > 0:
                print(f"Keeping visualization open for {args.visualize_hold_seconds:.1f}s ...")
                time.sleep(args.visualize_hold_seconds)
            visualizer.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous SmolVLA ROS2 control for OpenArm.")

    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task", default="pick up the sponge")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--arm-side", default="single", choices=["single", "left", "right"])
    parser.add_argument("--image-topic", default="/camera/color/image_raw")

    parser.add_argument(
        "--ros-interface",
        default="direct",
        choices=["direct", "bridge"],
        help="Use direct rclpy access from this Python 3.10 environment, or an HTTP ROS bridge.",
    )

    parser.add_argument("--bridge-url", default="http://127.0.0.1:8765")

    parser.add_argument("--visualize-host", default="127.0.0.1")
    parser.add_argument("--visualize-port", type=int, default=8770)
    parser.add_argument("--visualize-hold-seconds", type=float, default=10.0)
    parser.add_argument("--no-visualize", action="store_true")

    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-joint-delta", type=float, default=0.05)

    parser.add_argument("--gripper-effort", type=float, default=10.0)
    parser.add_argument(
        "--gripper-min-position",
        type=float,
        default=0.0,
        help="Safety lower bound for the ROS gripper command.",
    )
    parser.add_argument(
        "--gripper-max-position",
        type=float,
        default=0.044,
        help="Safety upper bound for the ROS gripper command.",
    )
    parser.add_argument(
        "--gripper-open-position",
        type=float,
        default=None,
        help="ROS gripper position for fully open. Defaults to the inferred open end of min/max.",
    )
    parser.add_argument(
        "--gripper-close-position",
        type=float,
        default=None,
        help="ROS gripper position for fully closed. Defaults to the inferred close end of min/max.",
    )
    parser.add_argument(
        "--policy-gripper-open-position",
        type=float,
        default=-67.81115,
        help="Training/action-space gripper value corresponding to fully open.",
    )
    parser.add_argument(
        "--policy-gripper-close-position",
        type=float,
        default=0.0,
        help="Training/action-space gripper value corresponding to fully closed.",
    )

    parser.add_argument(
        "--trajectory-duration",
        type=float,
        default=0.5,
        help="Seconds for each one-point joint trajectory goal sent to the active trajectory controller.",
    )

    parser.add_argument("--startup-timeout", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true", help="Actually publish commands to the robot.")

    parser.add_argument(
        "--fresh-inference-each-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset SmolVLA action queue each loop so every step uses the latest state and image.",
    )

    args = parser.parse_args()
    run_control(args)


if __name__ == "__main__":
    main()
