#!/usr/bin/env python
#
# SmolVLA OpenArm CAN continuous rollout demo.
#
# Safety test flow:
# 1. First run the CAN smoke test:
#      uv run python examples/tutorial/smolvla/openarm_can_smoke_test.py \
#        --can-port can0 \
#        --arm-side left \
#        --mode read
# 2. Confirm that robot state can be read.
# 3. Start this rollout demo.
# 4. Keep dry run=True.
# 5. Click Connect CAN robot.
# 6. Upload camera1 or set USB camera1 index.
# 7. Click Start rollout and confirm status updates continuously without sending actions.
# 8. Check joint_positions and gripper_position are reasonable.
# 9. Click Stop rollout.
# 10. Set rollout_hz low, for example 2 Hz, max_steps=5, max_duration_sec=3.
# 11. Only after dry-run looks safe, disable dry run carefully.
# 12. Before execution, ensure emergency stop is available and the arm workspace is clear.

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass
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


def _json_status(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


class UsbCameraReader:
    def __init__(self, index: int):
        self.index = int(index)
        self.cap = None
        self.cv2 = None

    def open(self) -> None:
        if self.index < 0:
            return
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV 不可用，无法读取 USB camera。请安装 opencv-python，或手动上传图像。") from exc
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            self.close()
            raise RuntimeError(f"无法打开 USB camera index {self.index}。")

    def read_pil(self) -> Image.Image:
        if self.cap is None or self.cv2 is None:
            raise RuntimeError("USB camera 尚未打开。")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"USB camera index {self.index} 没有返回图像。")
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, mode="RGB")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.cv2 = None


def _read_usb_camera_once(camera_index: int | None) -> Image.Image | None:
    if camera_index is None or int(camera_index) < 0:
        return None
    reader = UsbCameraReader(int(camera_index))
    try:
        reader.open()
        return reader.read_pil()
    finally:
        reader.close()


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
        self.lock = threading.Lock()

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
                "LeRobot OpenArm CAN backend 不可用，请检查 OpenArm/Damiao CAN 依赖和 SocketCAN 配置。"
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
                f"请检查 `ip link show {self.port}` 和 "
                f"`lerobot-setup-can --mode=test --interfaces={self.port}`；"
                "如果还没有校准，请先用 LeRobot CLI 完成 OpenArm follower 校准。"
                f"原始错误: {exc}"
            ) from exc

    def read_observation_dict(self) -> dict[str, float]:
        self._require_connected()
        with self.lock:
            observation = self.robot.get_observation()
        return {
            name: float(observation.get(name, 0.0))
            for name in OPENARM_STATE_ACTION_NAMES
        }

    def read_state_vector(self) -> list[float]:
        observation = self.read_observation_dict()
        return [float(observation[name]) for name in OPENARM_STATE_ACTION_NAMES]

    def current_positions(self) -> dict[str, float]:
        observation = self.read_observation_dict()
        return {motor: float(observation[f"{motor}.pos"]) for motor in OPENARM_MOTOR_NAMES}

    def send_smolvla_action(
        self,
        action_values: list[float],
        *,
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
        current_positions = self.current_positions()
        clipped_positions = self._clip_positions(target_positions, current_positions, float(max_joint_delta))
        action_payload = self._build_robot_action(clipped_positions)

        sent_action = None
        if not dry_run:
            with self.lock:
                sent_action = self.robot.send_action(action_payload)

        joint_positions = [float(clipped_positions[f"joint_{idx}"]) for idx in range(1, 8)]
        gripper_position = float(clipped_positions["gripper"])
        return {
            "dry_run": bool(dry_run),
            "execute": not bool(dry_run),
            "port": self.port,
            "side": self.side,
            "joint_positions": joint_positions,
            "gripper_position": gripper_position,
            "max_joint_delta": float(max_joint_delta),
            "backend": "lerobot_openarm_can",
            "action_schema": list(self.robot.action_features.keys()),
            "sent_action": sent_action,
        }

    def send_action(
        self,
        action_values: list[float],
        max_joint_delta: float,
        dry_run: bool,
    ) -> dict:
        return self.send_smolvla_action(
            action_values,
            max_joint_delta=max_joint_delta,
            dry_run=dry_run,
        )

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


def _predict_action(
    *,
    task: str,
    state_values: list[float],
    camera1: Image.Image | np.ndarray | None,
    camera2: Image.Image | np.ndarray | None,
    camera3: Image.Image | np.ndarray | None,
    model_path: str,
    device: str,
) -> tuple[np.ndarray, int]:
    policy, preprocess, postprocess = _load_policy(model_path, device)
    state_dim = _state_dim_from_checkpoint(model_path, policy.config.input_features["observation.state"].shape[0])
    if len(state_values) != state_dim:
        raise ValueError(f"observation.state 需要 {state_dim} 个数值，当前有 {len(state_values)} 个。")

    observation = {
        "observation.state": torch.tensor(state_values, dtype=torch.float32),
        "observation.images.camera1": _image_to_tensor(camera1, (256, 256)),
        "observation.images.camera2": _image_to_tensor(camera2, (256, 256)),
        "observation.images.camera3": _image_to_tensor(camera3, (256, 256)),
        "task": task or "",
    }
    with torch.inference_mode():
        batch = preprocess(observation)
        action = policy.select_action(batch)
        action = postprocess(action).squeeze(0).detach().cpu().float().numpy()
    return action, state_dim


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
    policy, _, _ = _load_policy(model_path, resolved_device)
    state_dim = _state_dim_from_checkpoint(model_path, policy.config.input_features["observation.state"].shape[0])
    state_values = [float(v) for v in _parse_state(state_text, state_dim).tolist()]
    usb_image = _read_usb_camera_once(usb_camera1_index)
    camera1_for_model = usb_image if usb_image is not None else camera1
    action, _ = _predict_action(
        task=task,
        state_values=state_values,
        camera1=camera1_for_model,
        camera2=camera2,
        camera3=camera3,
        model_path=model_path,
        device=resolved_device,
    )

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


@dataclass(frozen=True)
class RolloutParams:
    task: str
    model_path: str
    device: str
    can_port: str
    arm_side: str
    usb_camera1_index: int
    manual_camera1: Image.Image | np.ndarray | None
    manual_camera2: Image.Image | np.ndarray | None
    manual_camera3: Image.Image | np.ndarray | None
    max_joint_delta: float
    dry_run: bool
    rollout_hz: float
    max_steps: int
    max_duration_sec: float
    require_camera1: bool
    stop_on_error: bool
    update_preview_every_n_steps: int


class RolloutController:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.running = False
        self.latest_step = -1
        self.latest_status: dict[str, Any] = {}
        self.latest_action: list[float] | None = None
        self.latest_frame: Image.Image | None = None
        self.latest_observation_summary: dict[str, Any] = {}
        self.start_time: float | None = None
        self.error: str | None = None

    def start(self, params: RolloutParams) -> dict[str, Any]:
        if params.rollout_hz <= 0:
            raise ValueError("rollout_hz 必须 > 0。")
        if params.max_steps <= 0:
            raise ValueError("max_steps 必须 > 0。")
        if params.max_duration_sec <= 0:
            raise ValueError("max_duration_sec 必须 > 0。")
        if params.update_preview_every_n_steps <= 0:
            raise ValueError("update_preview_every_n_steps 必须 > 0。")
        if params.require_camera1 and params.usb_camera1_index < 0 and params.manual_camera1 is None:
            raise ValueError("require_camera1=True，但 USB camera disabled 且 camera1 未上传。")

        interface = _get_can_interface(
            params.can_port,
            params.arm_side,
            params.max_joint_delta,
            params.dry_run,
        )
        if not interface.is_connected:
            raise RuntimeError("OpenArm CAN robot 尚未连接。请先点击 Connect CAN robot。")

        with self.lock:
            if self.running:
                raise RuntimeError("rollout 已经在运行。")
            self.stop_event.clear()
            self.running = True
            self.latest_step = -1
            self.latest_action = None
            self.latest_frame = None
            self.latest_observation_summary = {}
            self.start_time = time.monotonic()
            self.error = None
            self.latest_status = {
                "running": True,
                "dry_run": params.dry_run,
                "execute": not params.dry_run,
                "port": params.can_port,
                "side": params.arm_side,
                "rollout_hz": params.rollout_hz,
                "max_steps": params.max_steps,
                "max_duration_sec": params.max_duration_sec,
                "started_at_monotonic": self.start_time,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stop_reason": None,
                "error": None,
            }
            self.thread = threading.Thread(target=self._run_loop, args=(params,), daemon=True)
            self.thread.start()
            return dict(self.latest_status)

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        status = self.status()
        status["stop_requested"] = True
        return status

    def status(self) -> dict[str, Any]:
        with self.lock:
            status = dict(self.latest_status)
            status.update(
                {
                    "running": self.running,
                    "latest_step": self.latest_step,
                    "latest_observation_summary": dict(self.latest_observation_summary),
                    "error": self.error,
                }
            )
            if self.latest_action is not None:
                action = np.asarray(self.latest_action, dtype=np.float32)
                status["latest_action_summary"] = {
                    "action_dim": int(action.shape[0]),
                    "min": float(action.min()),
                    "max": float(action.max()),
                    "mean": float(action.mean()),
                }
            return status

    def _run_loop(self, params: RolloutParams) -> None:
        reader: UsbCameraReader | None = None
        stop_reason = "finished"
        try:
            interface = _get_can_interface(
                params.can_port,
                params.arm_side,
                params.max_joint_delta,
                params.dry_run,
            )
            if not interface.is_connected:
                raise RuntimeError("OpenArm CAN robot 尚未连接。请先点击 Connect CAN robot。")

            policy, _, _ = _load_policy(params.model_path, params.device)
            state_dim = _state_dim_from_checkpoint(
                params.model_path,
                policy.config.input_features["observation.state"].shape[0],
            )
            if params.usb_camera1_index >= 0:
                reader = UsbCameraReader(params.usb_camera1_index)
                reader.open()

            period = 1.0 / params.rollout_hz
            loop_start = time.monotonic()
            next_tick = loop_start
            for step in range(params.max_steps):
                if self.stop_event.is_set():
                    stop_reason = "stop_requested"
                    break
                now = time.monotonic()
                elapsed = now - loop_start
                if elapsed >= params.max_duration_sec:
                    stop_reason = "max_duration_sec"
                    break

                step_start = time.monotonic()
                try:
                    state_values = interface.read_state_vector()
                    if len(state_values) != state_dim:
                        raise RuntimeError(f"robot state dim={len(state_values)} 与 policy state_dim={state_dim} 不匹配。")

                    if reader is not None:
                        camera1 = reader.read_pil()
                        camera1_source = f"usb camera {params.usb_camera1_index}"
                    elif params.manual_camera1 is not None:
                        camera1 = params.manual_camera1
                        camera1_source = "manual upload"
                    elif params.require_camera1:
                        raise RuntimeError("camera1 为空且 USB camera disabled。")
                    else:
                        camera1 = None
                        camera1_source = "black image fallback"

                    action, _ = _predict_action(
                        task=params.task,
                        state_values=state_values,
                        camera1=camera1,
                        camera2=params.manual_camera2,
                        camera3=params.manual_camera3,
                        model_path=params.model_path,
                        device=params.device,
                    )
                    if action.shape[0] < len(OPENARM_STATE_ACTION_NAMES):
                        raise RuntimeError(f"action 维度小于 24: got {action.shape[0]}")
                    send_result = interface.send_smolvla_action(
                        action.tolist(),
                        max_joint_delta=params.max_joint_delta,
                        dry_run=params.dry_run,
                    )

                    step_end = time.monotonic()
                    elapsed_after_step = step_end - loop_start
                    actual_hz = float((step + 1) / elapsed_after_step) if elapsed_after_step > 0 else 0.0
                    status = {
                        "running": True,
                        "step": step,
                        "elapsed_sec": elapsed_after_step,
                        "actual_hz": actual_hz,
                        "loop_time_ms": (step_end - step_start) * 1000.0,
                        "dry_run": params.dry_run,
                        "execute": not params.dry_run,
                        "port": params.can_port,
                        "side": params.arm_side,
                        "rollout_hz": params.rollout_hz,
                        "max_steps": params.max_steps,
                        "max_duration_sec": params.max_duration_sec,
                        "camera1_source": camera1_source,
                        "action_dim": int(action.shape[0]),
                        "action_min": float(action.min()),
                        "action_max": float(action.max()),
                        "action_mean": float(action.mean()),
                        "joint_positions": send_result["joint_positions"],
                        "gripper_position": send_result["gripper_position"],
                        "max_joint_delta": params.max_joint_delta,
                        "stop_reason": None,
                        "error": None,
                    }
                    observation_summary = {
                        "state_dim": len(state_values),
                        "state_min": float(np.min(state_values)),
                        "state_max": float(np.max(state_values)),
                        "state_mean": float(np.mean(state_values)),
                    }
                    if step % params.update_preview_every_n_steps == 0:
                        latest_frame = camera1.copy() if isinstance(camera1, Image.Image) else None
                    else:
                        latest_frame = None
                    self._update_status(
                        step=step,
                        status=status,
                        action=action.tolist(),
                        observation_summary=observation_summary,
                        frame=latest_frame,
                    )
                except Exception as exc:
                    self._set_error(str(exc), step=step)
                    if params.stop_on_error:
                        stop_reason = "error"
                        break

                next_tick += period
                remaining = next_tick - time.monotonic()
                while remaining > 0:
                    if self.stop_event.wait(min(remaining, 0.05)):
                        stop_reason = "stop_requested"
                        break
                    remaining = next_tick - time.monotonic()
                if stop_reason == "stop_requested":
                    break
            else:
                stop_reason = "max_steps"
        except Exception as exc:
            stop_reason = "error"
            self._set_error(str(exc), step=self.latest_step)
        finally:
            if reader is not None:
                reader.close()
            with self.lock:
                self.running = False
                final_status = dict(self.latest_status)
                final_status.update(
                    {
                        "running": False,
                        "stop_reason": stop_reason,
                        "error": self.error,
                    }
                )
                self.latest_status = final_status

    def _update_status(
        self,
        *,
        step: int,
        status: dict[str, Any],
        action: list[float],
        observation_summary: dict[str, Any],
        frame: Image.Image | None,
    ) -> None:
        with self.lock:
            self.latest_step = step
            self.latest_status = status
            self.latest_action = action
            self.latest_observation_summary = observation_summary
            if frame is not None:
                self.latest_frame = frame

    def _set_error(self, error: str, *, step: int) -> None:
        with self.lock:
            self.error = error
            self.latest_step = step
            status = dict(self.latest_status)
            status.update({"error": error, "stop_reason": "error"})
            self.latest_status = status


_ROLLOUT_CONTROLLER = RolloutController()


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
        result = interface.send_smolvla_action(
            action_values,
            max_joint_delta=max_joint_delta,
            dry_run=dry_run,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    return _json_status(result)


def _start_rollout(
    task: str,
    camera1: Image.Image | np.ndarray | None,
    camera2: Image.Image | np.ndarray | None,
    camera3: Image.Image | np.ndarray | None,
    usb_camera1_index: int | None,
    model_path: str,
    device: str,
    can_port: str,
    arm_side: str,
    max_joint_delta: float,
    dry_run: bool,
    rollout_hz: float,
    max_steps: int,
    max_duration_sec: float,
    require_camera1: bool,
    stop_on_error: bool,
    update_preview_every_n_steps: int,
):
    try:
        params = RolloutParams(
            task=task or "",
            model_path=str(Path(model_path).expanduser()),
            device=_select_device(device),
            can_port=can_port,
            arm_side=arm_side,
            usb_camera1_index=int(usb_camera1_index if usb_camera1_index is not None else -1),
            manual_camera1=camera1,
            manual_camera2=camera2,
            manual_camera3=camera3,
            max_joint_delta=float(max_joint_delta),
            dry_run=bool(dry_run),
            rollout_hz=float(rollout_hz),
            max_steps=int(max_steps),
            max_duration_sec=float(max_duration_sec),
            require_camera1=bool(require_camera1),
            stop_on_error=bool(stop_on_error),
            update_preview_every_n_steps=int(update_preview_every_n_steps),
        )
        status = _ROLLOUT_CONTROLLER.start(params)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    return _json_status(status)


def _stop_rollout():
    return _json_status(_ROLLOUT_CONTROLLER.stop())


def _get_rollout_status():
    status = _ROLLOUT_CONTROLLER.status()
    return _json_status(status)


def build_app(default_model_path: Path, default_device: str, default_can_port: str, default_arm_side: str) -> gr.Blocks:
    with gr.Blocks(title="SmolVLA OpenArm CAN Rollout") as app:
        gr.Markdown("# SmolVLA OpenArm CAN Rollout")
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

                gr.Markdown("## Continuous rollout")
                rollout_hz = gr.Number(label="rollout_hz", value=5.0, precision=2)
                max_steps = gr.Number(label="max_steps", value=100, precision=0)
                max_duration_sec = gr.Number(label="max_duration_sec", value=20.0, precision=2)
                require_camera1 = gr.Checkbox(label="require_camera1", value=True)
                stop_on_error = gr.Checkbox(label="stop_on_error", value=True)
                update_preview_every_n_steps = gr.Number(
                    label="update_preview_every_n_steps",
                    value=1,
                    precision=0,
                )
                start_rollout = gr.Button("Start rollout", variant="primary")
                stop_rollout = gr.Button("Stop rollout", variant="stop")
                refresh_rollout = gr.Button("Refresh rollout status")
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
        start_rollout.click(
            _start_rollout,
            inputs=[
                task,
                camera1,
                camera2,
                camera3,
                usb_camera1_index,
                model_path,
                device,
                can_port,
                arm_side,
                max_joint_delta,
                dry_run,
                rollout_hz,
                max_steps,
                max_duration_sec,
                require_camera1,
                stop_on_error,
                update_preview_every_n_steps,
            ],
            outputs=[robot_status],
        )
        stop_rollout.click(_stop_rollout, outputs=[robot_status])
        refresh_rollout.click(_get_rollout_status, outputs=[robot_status])
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
