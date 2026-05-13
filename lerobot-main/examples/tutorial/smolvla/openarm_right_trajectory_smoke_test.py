#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from PIL import Image as PILImage
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_SETS = {
    "single": [f"openarm_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}
GRIPPER_JOINTS = {
    "single": "openarm_finger_joint1",
    "right": "openarm_right_finger_joint1",
}
TRAJECTORY_ACTIONS = {
    "single": "/joint_trajectory_controller/follow_joint_trajectory",
    "right": "/right_joint_trajectory_controller/follow_joint_trajectory",
}
GRIPPER_ACTIONS = {
    "single": "/gripper_controller/gripper_cmd",
    "right": "/right_gripper_controller/gripper_cmd",
}
DEFAULT_IMAGE_TOPIC = "/camera/color/image_raw"


def image_msg_to_rgb(msg: Image) -> np.ndarray:
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


class VideoPreview:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.latest_info = "waiting for image"
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

    def update(self, msg: Image) -> None:
        rgb = image_msg_to_rgb(msg)
        image = PILImage.fromarray(rgb, mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        with self.lock:
            self.latest_jpeg = buffer.getvalue()
            self.latest_info = f"{msg.width}x{msg.height} {msg.encoding}"

    def _snapshot(self) -> tuple[bytes | None, str]:
        with self.lock:
            return self.latest_jpeg, self.latest_info

    def _make_handler(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in {"/", "/index.html"}:
                    body = (
                        "<html><head><title>OpenArm Camera</title></head>"
                        "<body style='margin:0;background:#111;color:#eee;font-family:sans-serif'>"
                        "<div style='padding:12px'>OpenArm camera preview</div>"
                        "<img src='/stream.mjpg' style='max-width:100vw;max-height:90vh;display:block;margin:auto'/>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
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
                        jpeg, info = preview._snapshot()
                        if jpeg is None:
                            jpeg = self._placeholder_jpeg(info)
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
            def _placeholder_jpeg(info: str) -> bytes:
                image = PILImage.new("RGB", (640, 360), color=(20, 20, 20))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=70)
                return buffer.getvalue()

        return Handler


class RightArmTrajectorySmokeTest(Node):
    def __init__(self, arm_mode: str, image_topic: str | None, preview: VideoPreview | None) -> None:
        super().__init__("openarm_right_trajectory_smoke_test")
        self.arm_mode = arm_mode
        self.image_topic = image_topic
        self.preview = preview
        self.latest_joint_names: list[str] = []
        self.joint_state: dict[str, float] = {}
        self.create_subscription(JointState, "/joint_states", self._joint_state_callback, 10)
        if image_topic is not None:
            self.create_subscription(Image, image_topic, self._image_callback, 10)
        self.trajectory_client: ActionClient | None = None
        self.gripper_client: ActionClient | None = None
        self.joint_names: list[str] | None = None
        self.gripper_joint_name: str | None = None
        self.trajectory_action: str | None = None
        self.gripper_action: str | None = None

    def _joint_state_callback(self, msg: JointState) -> None:
        self.latest_joint_names = list(msg.name)
        for index, name in enumerate(msg.name):
            if index < len(msg.position):
                self.joint_state[name] = float(msg.position[index])

    def _image_callback(self, msg: Image) -> None:
        if self.preview is None:
            return
        try:
            self.preview.update(msg)
        except Exception as exc:
            self.get_logger().warning(f"Failed to update camera preview: {exc}")

    def resolve_arm_mode(self) -> None:
        modes = ["right", "single"] if self.arm_mode == "auto" else [self.arm_mode]
        for mode in modes:
            joint_names = JOINT_SETS[mode]
            if all(name in self.joint_state for name in joint_names):
                self.joint_names = joint_names
                self.gripper_joint_name = GRIPPER_JOINTS[mode]
                self.trajectory_action = TRAJECTORY_ACTIONS[mode]
                self.gripper_action = GRIPPER_ACTIONS[mode]
                self.trajectory_client = ActionClient(self, FollowJointTrajectory, self.trajectory_action)
                self.gripper_client = ActionClient(self, GripperCommand, self.gripper_action)
                return
        available = ", ".join(self.latest_joint_names) if self.latest_joint_names else "<none>"
        expected = " or ".join(", ".join(JOINT_SETS[mode]) for mode in modes)
        raise RuntimeError(
            "Did not receive a complete supported arm joint set from /joint_states.\n"
            f"Expected: {expected}\n"
            f"Available: {available}"
        )

    def wait_for_joint_state(self, timeout: float) -> list[float]:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                self.resolve_arm_mode()
                assert self.joint_names is not None
                return [self.joint_state[name] for name in self.joint_names]
            except RuntimeError:
                pass
        self.resolve_arm_mode()
        raise AssertionError("unreachable")

    def wait_for_action_server(self, timeout: float) -> None:
        assert self.trajectory_client is not None
        assert self.trajectory_action is not None
        if not self.trajectory_client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"Trajectory action server is not available: {self.trajectory_action}")

    def wait_for_gripper_server(self, timeout: float) -> None:
        assert self.gripper_client is not None
        assert self.gripper_action is not None
        if not self.gripper_client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"Gripper action server is not available: {self.gripper_action}")

    def read_gripper_position(self) -> float | None:
        if self.gripper_joint_name is None:
            return None
        return self.joint_state.get(self.gripper_joint_name)

    def spin_for(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def send_positions(self, positions: list[float], duration: float) -> bool:
        assert self.joint_names is not None
        assert self.trajectory_client is not None
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(position) for position in positions]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        goal.trajectory.points.append(point)

        future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Right arm trajectory goal was rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            self.get_logger().error("Right arm trajectory finished without a result.")
            return False

        error_code = result.result.error_code
        error_string = result.result.error_string
        self.get_logger().info(f"Trajectory result: status={result.status}, error_code={error_code}, {error_string!r}")
        return error_code == 0

    def send_gripper(self, position: float, effort: float) -> bool:
        assert self.gripper_client is not None
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(effort)

        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Gripper goal was rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            self.get_logger().error("Gripper command finished without a result.")
            return False

        self.get_logger().info(f"Gripper result: status={result.status}")
        self.spin_for(0.3)
        observed = self.read_gripper_position()
        if observed is None:
            self.get_logger().warning(
                f"Gripper joint {self.gripper_joint_name} was not observed in /joint_states after command."
            )
        else:
            self.get_logger().info(f"Observed gripper joint {self.gripper_joint_name}: {observed:.4f}")
        return True

    def run_gripper_cycle(
        self,
        *,
        open_position: float,
        close_position: float,
        effort: float,
        pause: float,
        return_open: bool,
    ) -> bool:
        ok = self.send_gripper(open_position, effort)
        time.sleep(pause)
        ok = self.send_gripper(close_position, effort) and ok
        if return_open:
            time.sleep(pause)
            ok = self.send_gripper(open_position, effort) and ok
        return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Small single-arm trajectory action smoke test for OpenArm.")
    parser.add_argument(
        "--arm-mode",
        default="auto",
        choices=["auto", "single", "right"],
        help="auto detects openarm_joint* single-arm naming or openarm_right_joint* bimanual naming.",
    )
    parser.add_argument("--joint", type=int, default=4, choices=range(1, 8), help="Joint index to move.")
    parser.add_argument("--delta", type=float, default=0.05, help="Small joint offset in radians.")
    parser.add_argument("--duration", type=float, default=1.0, help="Seconds for the test trajectory.")
    parser.add_argument(
        "--gripper-position",
        type=float,
        default=None,
        help="Optional gripper target position. Omit this to leave the gripper untouched.",
    )
    parser.add_argument("--gripper-cycle", action="store_true", help="Open then close the gripper during the smoke test.")
    parser.add_argument("--gripper-open-position", type=float, default=0.044)
    parser.add_argument("--gripper-close-position", type=float, default=0.0)
    parser.add_argument("--gripper-effort", type=float, default=10.0)
    parser.add_argument("--gripper-pause", type=float, default=0.5)
    parser.add_argument("--return-gripper-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--video-host", default="127.0.0.1")
    parser.add_argument("--video-port", type=int, default=8766)
    parser.add_argument("--preview-seconds", type=float, default=5.0, help="Keep the preview alive after the motion.")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--return-home", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execute", action="store_true", help="Actually send the trajectory goal.")
    args = parser.parse_args()

    rclpy.init(args=None)
    preview = None if args.no_video else VideoPreview(args.video_host, args.video_port)
    if preview is not None:
        preview.start()
        print(f"Camera preview: {preview.url}")
        print(f"Image topic:    {args.image_topic}")

    node = RightArmTrajectorySmokeTest(
        args.arm_mode,
        image_topic=None if args.no_video else args.image_topic,
        preview=preview,
    )
    try:
        current = node.wait_for_joint_state(args.startup_timeout)
        target = current.copy()
        target[args.joint - 1] += args.delta

        print(f"Detected joints: {node.joint_names}")
        print(f"Gripper joint:   {node.gripper_joint_name}")
        print(f"Current joints:  {[round(value, 4) for value in current]}")
        gripper_position = node.read_gripper_position()
        if gripper_position is None:
            print("Current gripper: <not present in /joint_states>")
        else:
            print(f"Current gripper: {gripper_position:.4f}")
        print(f"Target joints:   {[round(value, 4) for value in target]}")
        print(f"Action server:   {node.trajectory_action}")
        print(f"Gripper server:  {node.gripper_action}")
        if args.gripper_cycle:
            print(
                "Gripper cycle:   "
                f"open={args.gripper_open_position:.4f}, "
                f"close={args.gripper_close_position:.4f}, "
                f"effort={args.gripper_effort:.4f}"
            )
        if args.gripper_position is not None:
            print(f"Gripper target:  position={args.gripper_position:.4f}, effort={args.gripper_effort:.4f}")

        if not args.execute:
            print("Dry-run only. Add --execute to publish the right-arm trajectory goal.")
            if preview is not None and args.preview_seconds > 0:
                end = time.monotonic() + args.preview_seconds
                while rclpy.ok() and time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.05)
            return

        node.wait_for_action_server(args.startup_timeout)
        ok = node.send_positions(target, args.duration)
        if args.gripper_cycle:
            node.wait_for_gripper_server(args.startup_timeout)
            ok = node.run_gripper_cycle(
                open_position=args.gripper_open_position,
                close_position=args.gripper_close_position,
                effort=args.gripper_effort,
                pause=args.gripper_pause,
                return_open=args.return_gripper_open,
            ) and ok
        elif args.gripper_position is not None:
            node.wait_for_gripper_server(args.startup_timeout)
            ok = node.send_gripper(args.gripper_position, args.gripper_effort) and ok
        if args.return_home:
            time.sleep(0.3)
            ok = node.send_positions(current, args.duration) and ok
        if not ok:
            raise RuntimeError("Right arm trajectory command did not complete successfully.")
        if preview is not None and args.preview_seconds > 0:
            end = time.monotonic() + args.preview_seconds
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        if preview is not None:
            preview.stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
