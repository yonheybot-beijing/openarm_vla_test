#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


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
OPENARM_MOTOR_NAMES = [f"joint_{joint}" for joint in range(1, 8)] + ["gripper"]
OPENARM_STATE_ACTION_NAMES = [
    f"{motor}.{field}" for motor in OPENARM_MOTOR_NAMES for field in ("pos", "vel", "torque")
]
SMOLVLA_POSITION_ACTION_INDICES = [idx * 3 for idx in range(8)]
_CAN_INTERFACES: dict[tuple[str, str], "OpenArmCanInterface"] = {}


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


def _available_network_interfaces() -> list[str]:
    net_dir = Path("/sys/class/net")
    if not net_dir.exists():
        return []
    return sorted(path.name for path in net_dir.iterdir())


def _check_socketcan_interface(port: str) -> None:
    interfaces = _available_network_interfaces()
    if port not in interfaces:
        can_interfaces = [name for name in interfaces if name.startswith("can")]
        raise RuntimeError(
            f"SocketCAN interface `{port}` 不存在。当前 CAN 接口: {can_interfaces or '无'}；"
            f"当前全部网络接口: {interfaces or '无'}。请先运行 `ip link show` 和 "
            f"`lerobot-setup-can --mode=test --interfaces={port}`。"
            "如果实际接口是 can1，请把 CAN port 改为 can1。"
        )


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


def _read_usb_camera(camera_index: int | None) -> Image.Image | None:
    if camera_index is None or camera_index < 0:
        return None
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV 不可用，无法读取 USB camera。请安装 opencv-python，或手动上传图像。") from exc

    cap = cv2.VideoCapture(int(camera_index))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 USB camera index {camera_index}。")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"USB camera index {camera_index} 没有返回图像。")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, mode="RGB")
    finally:
        cap.release()


def _json_status(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


class OpenArmCanInterface:
    def __init__(
        self,
        port: str,
        side: str,
        max_joint_delta: float,
        dry_run: bool,
        *,
        robot_id: str = "smolvla_follower",
    ):
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported arm side: {side}")
        self.port = port
        self.side = side
        self.max_joint_delta = float(max_joint_delta)
        self.dry_run = bool(dry_run)
        self.robot_id = robot_id
        self.robot = None

    @property
    def is_connected(self) -> bool:
        return bool(self.robot is not None and self.robot.is_connected)

    def connect(self) -> None:
        _check_socketcan_interface(self.port)
        try:
            from lerobot.robots.openarm_follower import OpenArmFollowerConfig
            from lerobot.robots.utils import make_robot_from_config
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot OpenArm CAN backend 不可用。请确认当前 uv Python 3.10 环境已安装 "
                "LeRobot OpenArm/Damiao CAN 依赖，并已配置 SocketCAN。"
            ) from exc

        if self.is_connected:
            return

        config = OpenArmFollowerConfig(
            id=self.robot_id,
            port=self.port,
            side=self.side,
            cameras={},
        )
        try:
            self.robot = make_robot_from_config(config)
            # Avoid interactive calibration prompts inside the Gradio callback.
            # Calibrate beforehand with the LeRobot CLI if this robot id has no saved calibration.
            self.robot.connect(calibrate=False)
        except ImportError as exc:
            self.robot = None
            raise RuntimeError(
                "LeRobot OpenArm CAN backend 依赖不完整。当前缺少 Damiao/CAN 依赖，"
                "常见原因是未安装 python-can。请在 /home/yangcw/openarm/lerobot-main 下执行："
                "`uv pip install -e '.[damiao]'`，或 `uv pip install -e '.[openarms]'`。"
            ) from exc
        except Exception as exc:
            self.robot = None
            raise RuntimeError(
                f"连接 OpenArm CAN follower 失败: port={self.port}, side={self.side}. "
                "请先检查 SocketCAN 和 OpenArm CAN 支持，例如："
                f"`ip link show {self.port}` 和 "
                f"`lerobot-setup-can --mode=test --interfaces={self.port}`；"
                "如果还没有校准，请先用 LeRobot CLI 完成 OpenArm follower 校准。"
                f"原始错误: {exc}"
            ) from exc

    def read_state_vector(self) -> list[float]:
        self._require_connected()
        observation = self.robot.get_observation()
        return self._state_vector_from_observation(observation)

    def send_action(
        self,
        action_values: list[float],
        max_joint_delta: float,
        dry_run: bool,
    ) -> dict:
        self._require_connected()
        if len(action_values) < len(OPENARM_STATE_ACTION_NAMES):
            raise ValueError(f"Expected at least 24-dim OpenArm action, got {len(action_values)}")

        target_positions = {
            motor: float(action_values[action_idx])
            for motor, action_idx in zip(OPENARM_MOTOR_NAMES, SMOLVLA_POSITION_ACTION_INDICES, strict=True)
        }
        current_positions = self._current_positions()
        clipped_positions = self._clip_positions(target_positions, current_positions, float(max_joint_delta))

        action_payload = self._build_robot_action(clipped_positions)
        sent_action = None
        if not dry_run:
            sent_action = self.robot.send_action(action_payload)

        joint_positions = [float(clipped_positions[f"joint_{idx}"]) for idx in range(1, 8)]
        gripper_position = float(clipped_positions["gripper"])
        return {
            "dry_run": bool(dry_run),
            "port": self.port,
            "side": self.side,
            "joint_positions": joint_positions,
            "gripper_position": gripper_position,
            "max_joint_delta": float(max_joint_delta),
            "backend": "lerobot_openarm_can",
            "action_schema": list(self.robot.action_features.keys()),
            "sent_action": sent_action,
        }

    def status(self) -> dict:
        if not self.is_connected:
            return {
                "connected": False,
                "port": self.port,
                "side": self.side,
                "backend": "lerobot_openarm_can",
            }
        return {
            "connected": True,
            "port": self.port,
            "side": self.side,
            "backend": "lerobot_openarm_can",
            "observation_schema": list(self.robot.observation_features.keys()),
            "action_schema": list(self.robot.action_features.keys()),
            "calibrate_on_connect": False,
        }

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("OpenArm CAN robot 尚未连接。请先点击 Connect CAN robot。")

    def _state_vector_from_observation(self, observation: dict[str, Any]) -> list[float]:
        values: list[float] = []
        for motor in OPENARM_MOTOR_NAMES:
            # OpenArmFollower.get_observation() 当前从 CAN 同步读取 position/velocity/torque，
            # 并暴露为 motor.pos/motor.vel/motor.torque。若某字段在后端缺失，临时补 0。
            values.extend(
                [
                    float(observation.get(f"{motor}.pos", 0.0)),
                    float(observation.get(f"{motor}.vel", 0.0)),
                    float(observation.get(f"{motor}.torque", 0.0)),
                ]
            )
        return values

    def _current_positions(self) -> dict[str, float]:
        observation = self.robot.get_observation()
        return {motor: float(observation.get(f"{motor}.pos", 0.0)) for motor in OPENARM_MOTOR_NAMES}

    def _clip_positions(
        self,
        target_positions: dict[str, float],
        current_positions: dict[str, float],
        max_joint_delta: float,
    ) -> dict[str, float]:
        if max_joint_delta <= 0:
            return dict(target_positions)
        return {
            motor: float(
                np.clip(
                    target,
                    current_positions[motor] - max_joint_delta,
                    current_positions[motor] + max_joint_delta,
                )
            )
            for motor, target in target_positions.items()
        }

    def _build_robot_action(self, positions: dict[str, float]) -> dict[str, float]:
        action_features = self.robot.action_features
        action: dict[str, float] = {}
        for motor, value in positions.items():
            key = f"{motor}.pos"
            if key in action_features:
                action[key] = float(value)
        if not action:
            raise RuntimeError(
                f"OpenArm follower action schema 中没有可用的 .pos 字段: {list(action_features.keys())}"
            )
        return action


def _get_can_interface(port: str, side: str, max_joint_delta: float, dry_run: bool) -> OpenArmCanInterface:
    key = (port, side)
    if key not in _CAN_INTERFACES:
        _CAN_INTERFACES[key] = OpenArmCanInterface(port, side, max_joint_delta, dry_run)
    interface = _CAN_INTERFACES[key]
    interface.max_joint_delta = float(max_joint_delta)
    interface.dry_run = bool(dry_run)
    return interface


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
    usb_camera1_index: int | None,
    model_path: str,
    device: str,
):
    model_path = str(Path(model_path).expanduser())
    resolved_device = _select_device(device)
    policy, preprocess, postprocess = _load_policy(model_path, resolved_device)

    state_dim = _state_dim_from_checkpoint(model_path, policy.config.input_features["observation.state"].shape[0])
    usb_image = _read_usb_camera(usb_camera1_index)
    camera1_for_model = usb_image if usb_image is not None else camera1
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
        "camera1_source": f"usb camera {usb_camera1_index}" if usb_image is not None else "manual upload",
        "action_dim": int(action.shape[0]),
        "smolvla_to_openarm_position_mapping": {
            motor: int(action_idx)
            for motor, action_idx in zip(OPENARM_MOTOR_NAMES, SMOLVLA_POSITION_ACTION_INDICES, strict=True)
        },
        "min": float(action.min()),
        "max": float(action.max()),
        "mean": float(action.mean()),
    }
    return camera1_for_model, rows, _json_status(summary), action.tolist()


def _connect_can_robot(can_port: str, arm_side: str, max_joint_delta: float, dry_run: bool):
    try:
        interface = _get_can_interface(can_port, arm_side, max_joint_delta, dry_run)
        interface.connect()
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    return _json_status(interface.status())


def _read_can_state(can_port: str, arm_side: str, max_joint_delta: float, dry_run: bool):
    try:
        interface = _get_can_interface(can_port, arm_side, max_joint_delta, dry_run)
        values = interface.read_state_vector()
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    status = interface.status()
    status.update(
        {
            "state_dim": len(values),
            "state_names": OPENARM_STATE_ACTION_NAMES,
            "zero_fill_note": "OpenArmFollower 当前提供 pos/vel/torque；若后端缺字段，本 demo 对缺字段补 0。",
        }
    )
    return ", ".join(f"{value:.6f}" for value in values), _json_status(status)


def _send_last_action_to_robot(
    action_values: list[float] | None,
    can_port: str,
    arm_side: str,
    max_joint_delta: float,
    dry_run: bool,
):
    if not action_values:
        raise gr.Error("还没有可发送的 action。请先点击 Predict。")
    try:
        interface = _get_can_interface(can_port, arm_side, max_joint_delta, dry_run)
        result = interface.send_action(action_values, max_joint_delta=max_joint_delta, dry_run=dry_run)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    return _json_status(result)


def build_app(default_model_path: Path, default_device: str, default_can_port: str, default_arm_side: str) -> gr.Blocks:
    with gr.Blocks(title="SmolVLA OpenArm CAN") as app:
        gr.Markdown("# SmolVLA OpenArm CAN")
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
                usb_camera1_index = gr.Number(label="USB camera1 index (-1 disables)", value=-1, precision=0)
                run = gr.Button("Predict", variant="primary")
                gr.Markdown("## OpenArm CAN robot interface")
                can_port = gr.Textbox(label="CAN port", value=default_can_port)
                arm_side = gr.Radio(label="arm side", choices=["left", "right"], value=default_arm_side)
                max_joint_delta = gr.Number(label="max joint delta per send", value=0.15, precision=3)
                dry_run = gr.Checkbox(label="dry run", value=True)
                connect_can = gr.Button("Connect CAN robot")
                read_can = gr.Button("Read CAN robot state")
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
                usb_camera1_index,
                model_path,
                device,
            ],
            outputs=[camera1, action_table, summary, last_action],
        )
        connect_can.click(
            _connect_can_robot,
            inputs=[can_port, arm_side, max_joint_delta, dry_run],
            outputs=[robot_status],
        )
        read_can.click(
            _read_can_state,
            inputs=[can_port, arm_side, max_joint_delta, dry_run],
            outputs=[state, robot_status],
        )
        send_robot.click(
            _send_last_action_to_robot,
            inputs=[last_action, can_port, arm_side, max_joint_delta, dry_run],
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
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--arm-side", default="left", choices=["left", "right"])
    args = parser.parse_args()

    app = build_app(args.model_path, args.device, args.can_port, args.arm_side)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
