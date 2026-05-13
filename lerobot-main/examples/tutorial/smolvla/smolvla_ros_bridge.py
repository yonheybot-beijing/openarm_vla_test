#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint


OPENARM_ROS_JOINT_NAMES = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}


class BridgeNode:
    def __init__(self, image_topic: str):
        self.node = rclpy.create_node("smolvla_ros_bridge")
        self.image_topic = image_topic
        self.lock = threading.Lock()
        self.joint_state: dict[str, dict[str, float]] = {}
        self.latest_image: Image | None = None

        self.node.create_subscription(JointState, "/joint_states", self._joint_state_callback, 10)
        self.node.create_subscription(Image, image_topic, self._image_callback, 10)
        self.trajectory_clients = {
            side: ActionClient(
                self.node,
                FollowJointTrajectory,
                f"/{side}_joint_trajectory_controller/follow_joint_trajectory",
            )
            for side in OPENARM_ROS_JOINT_NAMES
        }
        self.gripper_clients = {
            side: ActionClient(self.node, GripperCommand, f"/{side}_gripper_controller/gripper_cmd")
            for side in OPENARM_ROS_JOINT_NAMES
        }

    def _joint_state_callback(self, msg: JointState) -> None:
        with self.lock:
            for idx, name in enumerate(msg.name):
                self.joint_state[name] = {
                    "pos": msg.position[idx] if idx < len(msg.position) else 0.0,
                    "vel": msg.velocity[idx] if idx < len(msg.velocity) else 0.0,
                    "torque": msg.effort[idx] if idx < len(msg.effort) else 0.0,
                }

    def _image_callback(self, msg: Image) -> None:
        with self.lock:
            self.latest_image = msg

    def read_state_vector(self, arm_side: str) -> list[float]:
        if arm_side not in OPENARM_ROS_JOINT_NAMES:
            raise ValueError(f"unsupported arm_side: {arm_side}")
        with self.lock:
            values = []
            for name in OPENARM_ROS_JOINT_NAMES[arm_side]:
                item = self.joint_state.get(name, {})
                values.extend(
                    [
                        float(item.get("pos", 0.0)),
                        float(item.get("vel", 0.0)),
                        float(item.get("torque", 0.0)),
                    ]
                )
            values.extend([0.0, 0.0, 0.0])
        return values

    def read_image_payload(self) -> dict:
        with self.lock:
            msg = self.latest_image
        if msg is None:
            raise RuntimeError(f"no image received from {self.image_topic}")
        return {
            "encoding": msg.encoding,
            "height": msg.height,
            "width": msg.width,
            "step": msg.step,
            "data_b64": base64.b64encode(bytes(msg.data)).decode("ascii"),
            "topic": self.image_topic,
        }

    def send_action(
        self,
        arm_side: str,
        action: list[float],
        max_joint_delta: float,
        gripper_effort: float,
        trajectory_duration: float,
        execute: bool,
    ) -> dict:
        if arm_side not in OPENARM_ROS_JOINT_NAMES:
            raise ValueError(f"unsupported arm_side: {arm_side}")
        if len(action) != 24:
            raise ValueError(f"expected 24 action values, got {len(action)}")

        target_joints = [float(action[idx * 3]) for idx in range(7)]
        gripper_position = float(action[21])

        with self.lock:
            current = [
                self.joint_state.get(name, {}).get("pos") for name in OPENARM_ROS_JOINT_NAMES[arm_side]
            ]

        if max_joint_delta > 0 and all(value is not None for value in current):
            target_joints = [
                float(np.clip(target, now - max_joint_delta, now + max_joint_delta))
                for target, now in zip(target_joints, current, strict=True)
            ]

        if execute:
            trajectory_client = self.trajectory_clients[arm_side]
            if not trajectory_client.wait_for_server(timeout_sec=0.1):
                raise RuntimeError(
                    f"trajectory action server not available: "
                    f"/{arm_side}_joint_trajectory_controller/follow_joint_trajectory"
                )

            trajectory_goal = FollowJointTrajectory.Goal()
            trajectory_goal.trajectory.joint_names = OPENARM_ROS_JOINT_NAMES[arm_side]
            point = JointTrajectoryPoint()
            point.positions = target_joints
            point.time_from_start.sec = int(trajectory_duration)
            point.time_from_start.nanosec = int((trajectory_duration - int(trajectory_duration)) * 1e9)
            trajectory_goal.trajectory.points.append(point)
            trajectory_client.send_goal_async(trajectory_goal)

            client = self.gripper_clients[arm_side]
            if client.wait_for_server(timeout_sec=0.05):
                goal = GripperCommand.Goal()
                goal.command.position = gripper_position
                goal.command.max_effort = gripper_effort
                client.send_goal_async(goal)

        return {
            "execute": execute,
            "arm_side": arm_side,
            "joint_positions": target_joints,
            "gripper_position": gripper_position,
            "max_joint_delta": max_joint_delta,
            "trajectory_duration": trajectory_duration,
        }


def make_handler(bridge: BridgeNode):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status: int, exc: Exception) -> None:
            self._send_json(status, {"ok": False, "error": str(exc)})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/health":
                    self._send_json(200, {"ok": True, "image_topic": bridge.image_topic})
                elif parsed.path == "/state":
                    arm_side = query.get("arm_side", ["left"])[0]
                    self._send_json(200, {"ok": True, "state": bridge.read_state_vector(arm_side)})
                elif parsed.path == "/image":
                    payload = bridge.read_image_payload()
                    payload["ok"] = True
                    self._send_json(200, payload)
                else:
                    self._send_json(404, {"ok": False, "error": f"unknown path: {parsed.path}"})
            except Exception as exc:
                self._send_error_json(500, exc)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                if parsed.path != "/send_action":
                    self._send_json(404, {"ok": False, "error": f"unknown path: {parsed.path}"})
                    return
                result = bridge.send_action(
                    arm_side=data.get("arm_side", "left"),
                    action=data["action"],
                    max_joint_delta=float(data.get("max_joint_delta", 0.05)),
                    gripper_effort=float(data.get("gripper_effort", 10.0)),
                    trajectory_duration=float(data.get("trajectory_duration", 0.5)),
                    execute=bool(data.get("execute", False)),
                )
                result["ok"] = True
                self._send_json(200, result)
            except Exception as exc:
                self._send_error_json(500, exc)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[bridge-http] {fmt % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP bridge between LeRobot and ROS2.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    args = parser.parse_args()

    rclpy.init(args=None)
    bridge = BridgeNode(args.image_topic)

    ros_thread = threading.Thread(target=rclpy.spin, args=(bridge.node,), daemon=True)
    ros_thread.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(bridge))
    print(f"SmolVLA ROS bridge listening on http://{args.host}:{args.port}")
    print(f"Subscribing image topic: {args.image_topic}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        bridge.node.destroy_node()
        rclpy.shutdown()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
