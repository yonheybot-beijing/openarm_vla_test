#!/usr/bin/env python

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def _normalize_proxy_env() -> None:
    """httpx accepts socks5://, while some shells export socks://."""
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value and value.startswith("socks://"):
            os.environ[key] = value.replace("socks://", "socks5://", 1)


_normalize_proxy_env()

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


DEFAULT_MODEL_PATH = Path("smolvla_sponge_20k_pretrained_model/pretrained_model")
OPENARM_STATE_ACTION_NAMES = [
    f"joint_{joint}.{field}"
    for joint in range(1, 8)
    for field in ("pos", "vel", "torque")
] + ["gripper.pos", "gripper.vel", "gripper.torque"]
OPENARM_ROS_JOINT_NAMES = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}
_ROS_INTERFACES = {}
_ROS_IMAGE_INTERFACES = {}


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_state(state_text: str, state_dim: int) -> torch.Tensor:
    values = [float(part.strip()) for part in state_text.replace("\n", ",").split(",") if part.strip()]
    if len(values) != state_dim:
        raise gr.Error(f"observation.state 需要 {state_dim} 个数值，当前输入了 {len(values)} 个。")
    return torch.tensor(values, dtype=torch.float32)


def _state_dim_from_checkpoint(model_path: str, fallback: int) -> int:
    stats_path = Path(model_path) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    if not stats_path.exists():
        return fallback
    stats = load_file(stats_path)
    state_mean = stats.get("observation.state.mean")
    if state_mean is None:
        return fallback
    return int(state_mean.numel())


def _image_to_tensor(image: Image.Image | np.ndarray | None, fallback_hw: tuple[int, int]) -> torch.Tensor:
    if image is None:
        height, width = fallback_hw
        image = Image.new("RGB", (width, height), color=(0, 0, 0))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _ros_image_msg_to_rgb(msg) -> np.ndarray:
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


def _ros_image_payload_to_rgb(payload: dict) -> np.ndarray:
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


def _bridge_get_json(bridge_url: str, path: str) -> dict:
    url = f"{bridge_url.rstrip('/')}{path}"
    try:
        with urlopen(url, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"无法连接 ROS bridge: {url}: {exc}") from exc
    if not payload.get("ok", False):
        raise RuntimeError(payload.get("error", "ROS bridge request failed"))
    return payload


def _bridge_post_json(bridge_url: str, path: str, data: dict) -> dict:
    url = f"{bridge_url.rstrip('/')}{path}"
    body = json.dumps(data).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"无法连接 ROS bridge: {url}: {exc}") from exc
    if not payload.get("ok", False):
        raise RuntimeError(payload.get("error", "ROS bridge request failed"))
    return payload


class RosImageInterface:
    def __init__(self, topic: str):
        try:
            import rclpy
            from sensor_msgs.msg import Image as RosImage
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 图像包不可用。请先在启动 demo 的终端执行："
                "source /home/yangcw/astra_ws/install/setup.bash"
            ) from exc

        if not rclpy.ok():
            rclpy.init(args=None)

        self.rclpy = rclpy
        self.topic = topic
        self.latest_msg = None
        topic_suffix = topic.strip("/").replace("/", "_") or "image"
        self.node = rclpy.create_node(f"smolvla_gradio_{topic_suffix}_subscriber")
        self.subscription = self.node.create_subscription(RosImage, topic, self._image_callback, 10)

    def _image_callback(self, msg) -> None:
        self.latest_msg = msg

    def read_image(self, timeout: float = 2.0) -> Image.Image:
        deadline = time.monotonic() + timeout
        while self.latest_msg is None and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.latest_msg is None:
            raise RuntimeError(f"没有从 {self.topic} 收到图像。请确认 Astra 相机节点已启动、topic 名正确。")

        self.rclpy.spin_once(self.node, timeout_sec=0.02)
        rgb = _ros_image_msg_to_rgb(self.latest_msg)
        return Image.fromarray(rgb, mode="RGB")


def _get_ros_image_interface(topic: str) -> RosImageInterface:
    if topic not in _ROS_IMAGE_INTERFACES:
        _ROS_IMAGE_INTERFACES[topic] = RosImageInterface(topic)
    return _ROS_IMAGE_INTERFACES[topic]


class OpenArmRosInterface:
    def __init__(self, arm_side: str):
        try:
            import rclpy
            from control_msgs.action import GripperCommand
            from rclpy.action import ActionClient
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64MultiArray
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 Python 包不可用。请先在启动 demo 的终端执行："
                "source /home/yangcw/openarm/openarm_ws/install/setup.bash"
            ) from exc

        if arm_side not in OPENARM_ROS_JOINT_NAMES:
            raise ValueError(f"Unsupported arm side: {arm_side}")

        if not rclpy.ok():
            rclpy.init(args=None)

        self.rclpy = rclpy
        self.Float64MultiArray = Float64MultiArray
        self.GripperCommand = GripperCommand
        self.arm_side = arm_side
        self.joint_names = OPENARM_ROS_JOINT_NAMES[arm_side]
        self.joint_state = {}
        self.node = rclpy.create_node(f"smolvla_gradio_{arm_side}_interface")
        self.command_pub = self.node.create_publisher(
            Float64MultiArray,
            f"/{arm_side}_forward_position_controller/commands",
            10,
        )
        self.joint_sub = self.node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
        )
        self.gripper_client = ActionClient(
            self.node,
            GripperCommand,
            f"/{arm_side}_gripper_controller/gripper_cmd",
        )

    def _joint_state_callback(self, msg):
        for idx, name in enumerate(msg.name):
            self.joint_state[name] = {
                "pos": msg.position[idx] if idx < len(msg.position) else 0.0,
                "vel": msg.velocity[idx] if idx < len(msg.velocity) else 0.0,
                "torque": msg.effort[idx] if idx < len(msg.effort) else 0.0,
            }

    def spin(self, seconds: float = 0.2) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.02)

    def read_state_vector(self) -> list[float]:
        self.spin(0.5)
        values = []
        for name in self.joint_names:
            item = self.joint_state.get(name, {})
            values.extend([float(item.get("pos", 0.0)), float(item.get("vel", 0.0)), float(item.get("torque", 0.0))])
        values.extend([0.0, 0.0, 0.0])
        return values

    def send_action(
        self,
        action: list[float],
        *,
        max_joint_delta: float,
        gripper_effort: float,
        dry_run: bool,
    ) -> dict:
        if len(action) != len(OPENARM_STATE_ACTION_NAMES):
            raise ValueError(f"Expected 24-dim OpenArm action, got {len(action)}")

        self.spin(0.1)
        target_joints = [float(action[idx * 3]) for idx in range(7)]
        gripper_position = float(action[21])

        current = [self.joint_state.get(name, {}).get("pos") for name in self.joint_names]
        if max_joint_delta > 0 and all(value is not None for value in current):
            target_joints = [
                float(np.clip(target, now - max_joint_delta, now + max_joint_delta))
                for target, now in zip(target_joints, current, strict=True)
            ]

        if not dry_run:
            msg = self.Float64MultiArray()
            msg.data = target_joints
            self.command_pub.publish(msg)

            if self.gripper_client.wait_for_server(timeout_sec=0.2):
                goal = self.GripperCommand.Goal()
                goal.command.position = gripper_position
                goal.command.max_effort = gripper_effort
                self.gripper_client.send_goal_async(goal)

        return {
            "dry_run": dry_run,
            "arm_side": self.arm_side,
            "joint_topic": f"/{self.arm_side}_forward_position_controller/commands",
            "gripper_action": f"/{self.arm_side}_gripper_controller/gripper_cmd",
            "joint_positions": target_joints,
            "gripper_position": gripper_position,
            "max_joint_delta": max_joint_delta,
        }


def _get_ros_interface(arm_side: str) -> OpenArmRosInterface:
    if arm_side not in _ROS_INTERFACES:
        _ROS_INTERFACES[arm_side] = OpenArmRosInterface(arm_side)
    return _ROS_INTERFACES[arm_side]


@lru_cache(maxsize=2)
def _load_policy(model_path: str, device: str):
    policy = SmolVLAPolicy.from_pretrained(model_path, cli_overrides=[f"--device={device}"])
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    policy.reset()
    return policy, preprocess, postprocess


def _predict(
    task: str,
    state_text: str,
    camera1: Image.Image | np.ndarray | None,
    camera2: Image.Image | np.ndarray | None,
    camera3: Image.Image | np.ndarray | None,
    use_astra_camera: bool,
    bridge_url: str,
    model_path: str,
    device: str,
):
    model_path = str(Path(model_path).expanduser())
    resolved_device = _select_device(device)
    policy, preprocess, postprocess = _load_policy(model_path, resolved_device)

    state_dim = _state_dim_from_checkpoint(model_path, policy.config.input_features["observation.state"].shape[0])
    camera1_for_model = _read_astra_image_from_bridge(bridge_url)[0] if use_astra_camera else camera1
    observation = {
        "observation.state": _parse_state(state_text, state_dim),
        "observation.images.camera1": _image_to_tensor(camera1_for_model, (256, 256)),
        "observation.images.camera2": _image_to_tensor(camera2, (256, 256)),
        "observation.images.camera3": _image_to_tensor(camera3, (256, 256)),
        "task": task or "",
    }

    with torch.inference_mode():
        batch = preprocess(observation)
        action = policy.select_action(batch)
        action = postprocess(action).squeeze(0).detach().cpu().float().numpy()

    names = OPENARM_STATE_ACTION_NAMES if action.shape[0] == len(OPENARM_STATE_ACTION_NAMES) else None
    rows = pd.DataFrame(
        {
            "index": np.arange(action.shape[0]),
            "name": names or [f"action_{idx}" for idx in range(action.shape[0])],
            "value": action,
        }
    )
    summary = {
        "model_path": model_path,
        "device": resolved_device,
        "camera1_source": bridge_url if use_astra_camera else "manual upload",
        "action_dim": int(action.shape[0]),
        "min": float(action.min()),
        "max": float(action.max()),
        "mean": float(action.mean()),
    }
    return camera1_for_model, rows, json.dumps(summary, ensure_ascii=False, indent=2), action.tolist()


def _read_astra_image_from_bridge(bridge_url: str):
    try:
        payload = _bridge_get_json(bridge_url, "/image")
        image = Image.fromarray(_ros_image_payload_to_rgb(payload), mode="RGB")
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    status = {
        "bridge_url": bridge_url,
        "camera_topic": payload.get("topic"),
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
    }
    return image, json.dumps(status, ensure_ascii=False, indent=2)


def _read_ros_state(arm_side: str, bridge_url: str):
    try:
        payload = _bridge_get_json(bridge_url, f"/state?arm_side={arm_side}")
        values = [float(value) for value in payload["state"]]
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    status = {
        "arm_side": arm_side,
        "source": "/joint_states",
        "state_dim": len(values),
        "state_names": OPENARM_STATE_ACTION_NAMES,
    }
    return ", ".join(f"{value:.6f}" for value in values), json.dumps(status, ensure_ascii=False, indent=2)


def _send_last_action_to_robot(
    action_values: list[float] | None,
    arm_side: str,
    max_joint_delta: float,
    gripper_effort: float,
    dry_run: bool,
    bridge_url: str,
):
    if not action_values:
        raise gr.Error("还没有可发送的 action。请先点击 Predict。")
    try:
        result = _bridge_post_json(
            bridge_url,
            "/send_action",
            {
                "arm_side": arm_side,
                "action": action_values,
                "max_joint_delta": max_joint_delta,
                "gripper_effort": gripper_effort,
                "execute": not dry_run,
            },
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    return json.dumps(result, ensure_ascii=False, indent=2)


def build_app(default_model_path: Path, default_device: str) -> gr.Blocks:
    with gr.Blocks(title="SmolVLA Sponge") as app:
        gr.Markdown("# SmolVLA Sponge")
        last_action = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                model_path = gr.Textbox(label="model path", value=str(default_model_path))
                device = gr.Dropdown(label="device", choices=["auto", "cuda", "cpu", "mps"], value=default_device)
                task = gr.Textbox(label="task", value="pick up the sponge")
                state = gr.Textbox(
                    label="observation.state",
                    value=", ".join(["0"] * len(OPENARM_STATE_ACTION_NAMES)),
                    lines=3,
                )
                gr.Markdown("## Astra camera")
                bridge_url = gr.Textbox(label="ROS bridge URL", value="http://127.0.0.1:8765")
                use_astra_camera = gr.Checkbox(label="use Astra topic for camera1 on Predict", value=True)
                read_astra = gr.Button("Read Astra camera")
                run = gr.Button("Predict", variant="primary")
                gr.Markdown("## ROS2 robot interface")
                arm_side = gr.Radio(label="arm side", choices=["left", "right"], value="left")
                max_joint_delta = gr.Number(label="max joint delta per send", value=0.15, precision=3)
                gripper_effort = gr.Number(label="gripper effort", value=10.0, precision=2)
                dry_run = gr.Checkbox(label="dry run", value=True)
                read_ros = gr.Button("Read /joint_states")
                send_robot = gr.Button("Send last action")
            with gr.Column(scale=2):
                with gr.Row():
                    camera1 = gr.Image(label="camera1", type="pil", image_mode="RGB")
                    camera2 = gr.Image(label="camera2", type="pil", image_mode="RGB")
                    camera3 = gr.Image(label="camera3", type="pil", image_mode="RGB")
                action_table = gr.Dataframe(label="action", interactive=False)
                summary = gr.Code(label="summary", language="json")
                robot_status = gr.Code(label="robot status", language="json")

        run.click(
            _predict,
            inputs=[
                task,
                state,
                camera1,
                camera2,
                camera3,
                use_astra_camera,
                bridge_url,
                model_path,
                device,
            ],
            outputs=[camera1, action_table, summary, last_action],
        )
        read_astra.click(
            _read_astra_image_from_bridge,
            inputs=[bridge_url],
            outputs=[camera1, robot_status],
        )
        read_ros.click(
            _read_ros_state,
            inputs=[arm_side, bridge_url],
            outputs=[state, robot_status],
        )
        send_robot.click(
            _send_last_action_to_robot,
            inputs=[last_action, arm_side, max_joint_delta, gripper_effort, dry_run, bridge_url],
            outputs=[robot_status],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.model_path, args.device)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
