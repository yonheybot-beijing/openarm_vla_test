# OpenArm SmolVLA 真机部署说明

本目录用于 OpenArm 右臂、Astra USB 相机和 SmolVLA 策略的本地部署调试。当前 VLA 部署代码已经可以完成模型加载、相机读取、ROS2 机械臂控制和真机动作下发，但效果极差，仍需要根据VLA部署trick以及调整控制参数、夹爪映射、相机位置，并可能需要基于本机任务重新采集数据后再训练。

> 注意：训练好的模型权重不随本 README 提供。下文中的模型路径均使用通用占位符，请替换为实际下载或训练得到的本地路径。

## 目录说明

```text
/path/to/openarm
├── lerobot-main/
│   └── examples/tutorial/smolvla/
│       ├── smolvla_ros_control.py                 # VLA 核心运行脚本
│       ├── openarm_right_trajectory_smoke_test.py # 右臂/夹爪通信测试脚本
│       └── smolvla_gradio_*.py                    # Gradio/调试相关脚本
├── openarm_ws/
│   ├── install/setup.bash                         # ROS2 工作空间环境
│   └── elbow_lift_smoke_test.py                   # 抬肘烟雾测试脚本
├── lerobot_openarm-main/                          # OpenArm 主从遥操作与数据采集代码
├── openarm/                                       # OpenArm 上游源码与硬件/仿真/ROS2 相关包
├── openarm_ros2_gravcomp/                         # 重力补偿相关 ROS2 包和实验代码
├── smolvla_sponge_20k_pretrained_model/           # 本地微调模型目录示例，README 不提供权重
├── mujoco-3.8.0-linux-x86_64/                     # MuJoCo 本地安装包/仿真资源
├── x1/                                            # 其他前端/应用项目，非 VLA 部署主流程
└── 部署-代码/                                     # 旧版或补充部署代码备份
```

主要目录用途：

- `lerobot-main/`：LeRobot 主项目目录。VLA 部署、模型加载测试、右臂轨迹测试等命令主要在这里运行。
- `lerobot-main/examples/tutorial/smolvla/`：本次 SmolVLA 部署最核心的脚本目录。`smolvla_ros_control.py` 负责读取相机、调用模型、生成动作并通过 ROS2 下发；`openarm_right_trajectory_smoke_test.py` 用于单臂轨迹和夹爪测试。
- `openarm_ws/`：当前真机 ROS2 工作空间。启动机械臂控制前需要 `source /path/to/openarm_ws/install/setup.bash`，`elbow_lift_smoke_test.py` 可用于快速验证关节通信。
- `lerobot_openarm-main/`：OpenArm 主从遥操作和 LeRobot 数据采集代码。若现有公开数据集效果不好，后续建议用这里的代码重新采集本机任务数据。
- `openarm/`：OpenArm 上游代码集合，包含 CAN、ROS2、MuJoCo、Isaac Lab、遥操作、官网等模块，主要用于查硬件接口、底层控制和仿真相关实现。
- `openarm_ros2_gravcomp/`：重力补偿相关 ROS2 代码，包含 bringup、hardware、description、MoveIt 配置和 gravcomp controller 等包。
- `smolvla_sponge_20k_pretrained_model/`：本机曾使用的微调模型目录示例。正式交付时不要依赖该具体路径，应替换为自己的 `/path/to/smolvla-finetuned-model/pretrained_model`。
- `mujoco-3.8.0-linux-x86_64/`：MuJoCo 本地文件，主要用于仿真或模型调试，不是真机 VLA 部署的必经步骤。
- `x1/`、`部署-代码/`：与当前 SmolVLA 真机部署关系较弱，通常不需要在主流程中修改。

## 启动前准备

1. 开机时选择 Ubuntu 系统。
2. 如需联网或访问外部模型/数据集，可进入 `~/clash`，参考 `clash_command.txt` 中的命令在该目录终端运行代理。
3. 建议在本目录打开 VS Code：

```bash
cd /home/yangcw/openarm
code ./
```

后续命令尽量在 `/home/yangcw/openarm` 或对应子目录终端执行。

## 模型路径约定

完整 VLA 由基础 VLM 和微调后的策略权重组合而成。请根据本机实际情况替换：

```text
基础 VLM: /path/to/hf-cache/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct
微调模型: /path/to/smolvla-finetuned-model/pretrained_model
```

运行离线推理时建议设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 启动相机

相机通过 USB 连接。连接后启动 Astra 相机：

```bash
ros2 launch astra_camera astra_pro.launch.xml
```

查看 ROS2 topic：

```bash
ros2 topic list
```

如果出现以下 topic，说明相机话题可读，也可以用 `rviz2` 查看画面：

```text
/camera/color/camera_info
/camera/color/image_raw
/camera/depth/camera_info
/camera/depth/image_raw
/camera/depth/points
/camera/ir/camera_info
/camera/ir/image_raw
```

VLA 默认使用彩色图像话题：

```text
/camera/color/image_raw
```

## 配置右臂 CAN 口

右臂通过 USB 连接电脑后，先配置 CAN 口。常见设备名为 `/dev/ttyACM1`，如果不存在，可尝试 `/dev/ttyACM0`、`/dev/ttyACM2` 或其他 `/dev/ttyACM*`。

```bash
sudo slcand -o -c -s8 /dev/ttyACM1 can0
sudo ip link set can0 up
ip -details link show can0
```

没有报错通常表示 CAN 口配置成功。

## 启动 OpenArm ROS2 控制

```bash
source /home/yangcw/openarm/openarm_ws/install/setup.bash
ros2 launch openarm_bringup openarm.launch.py arm_type:=v10 can_interface:=can0
```

另开终端查看 topic：

```bash
ros2 topic list
```

如果能看到 `joint_state` 或类似关节状态/控制器话题，说明 ROS2 通信基本成功。

## 机械臂通信测试

### ROS2 action 抬起关节

```bash
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{
  trajectory: {
    joint_names: [
      'openarm_joint1',
      'openarm_joint2',
      'openarm_joint3',
      'openarm_joint4',
      'openarm_joint5',
      'openarm_joint6',
      'openarm_joint7'
    ],
    points: [
      {
        positions: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        time_from_start: {sec: 3}
      }
    ]
  }
}"
```

执行后机械臂向上抬起一定角度，说明通信配置成功。

### ROS2 action 放下关节

```bash
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{
  trajectory: {
    joint_names: [
      'openarm_joint1',
      'openarm_joint2',
      'openarm_joint3',
      'openarm_joint4',
      'openarm_joint5',
      'openarm_joint6',
      'openarm_joint7'
    ],
    points: [
      {
        positions: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        time_from_start: {sec: 3}
      }
    ]
  }
}"
```

### 抬肘烟雾测试

```bash
cd /home/yangcw/openarm
source /home/yangcw/openarm/openarm_ws/install/setup.bash

python3 openarm_ws/elbow_lift_smoke_test.py \
  --elbow-joint 4 \
  --lift-delta 0.25 \
  --duration 3.0 \
  --return-home
```

如果 `python3` 环境缺依赖，可尝试：

```bash
uv run python openarm_ws/elbow_lift_smoke_test.py \
  --elbow-joint 4 \
  --lift-delta 0.25 \
  --duration 3.0 \
  --return-home
```

### LeRobot 目录下的右臂轨迹测试

以下脚本需要在 `lerobot-main` 目录下运行：

```bash
cd /home/yangcw/openarm/lerobot-main
source /home/yangcw/openarm/openarm_ws/install/setup.bash

uv run python examples/tutorial/smolvla/openarm_right_trajectory_smoke_test.py \
  --arm-mode single \
  --joint 1 \
  --delta 0.6 \
  --duration 5.0 \
  --execute
```

该命令主要用于通信验证，参数已经过简单测试，初期可以不修改。

## 夹爪开合测试

训练集夹爪动作和当前机械臂夹爪命令不完全一致，需要根据当前夹爪的 open/close position 做映射。可以用下面命令观察 `0.044` 和 `0.0` 分别对应的夹爪位置：

```bash
cd /home/yangcw/openarm/lerobot-main
source /home/yangcw/openarm/openarm_ws/install/setup.bash

python3 examples/tutorial/smolvla/openarm_right_trajectory_smoke_test.py \
  --arm-mode single \
  --delta 0.0 \
  --duration 1.0 \
  --gripper-cycle \
  --gripper-open-position 0.044 \
  --gripper-close-position 0.0 \
  --gripper-effort 10.0 \
  --gripper-pause 1.0 \
  --return-gripper-open \
  --no-video \
  --execute
```

不加 `--execute` 时只跑数据流程，不会下发到真机；加上 `--execute` 才会控制真实机械臂。

## SmolVLA 模型加载测试

该命令用于确认模型能离线加载，并输出 5 步动作。输出动作通常在小数范围内变化；`raw gripper` 值可能较大，这是训练集原始夹爪数据，重点看映射后的夹爪和关节动作是否合理。

```bash
cd /home/yangcw/openarm/lerobot-main
source /home/yangcw/openarm/openarm_ws/install/setup.bash

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run python examples/tutorial/smolvla/smolvla_ros_control.py \
  --model-path /path/to/smolvla-finetuned-model/pretrained_model \
  --arm-side single \
  --image-topic /camera/color/image_raw \
  --task "pick and place the blue sponge" \
  --device cuda \
  --hz 1 \
  --max-steps 5 \
  --max-joint-delta 0.01 \
  --trajectory-duration 1.0 \
  --gripper-min-position 0.0 \
  --gripper-max-position 0.044 \
  --gripper-open-position 0.044 \
  --gripper-close-position 0.0
```

## SmolVLA 真机运行

确认相机、CAN、OpenArm ROS2 控制、模型加载都正常后，再运行真机部署。初期建议低频率、小步长测试，确认安全后再逐步提高频率。

```bash
cd /home/yangcw/openarm/lerobot-main
source /home/yangcw/openarm/openarm_ws/install/setup.bash

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run python examples/tutorial/smolvla/smolvla_ros_control.py \
  --model-path /path/to/smolvla-finetuned-model/pretrained_model \
  --arm-side single \
  --image-topic /camera/color/image_raw \
  --task "pick and place the blue sponge" \
  --device cuda \
  --hz 5 \
  --max-steps 100 \
  --max-joint-delta 0.035 \
  --trajectory-duration 0.18 \
  --gripper-min-position 0.0 \
  --gripper-max-position 0.0 \
  --gripper-open-position 0.0 \
  --gripper-close-position 0.0 \
  --execute
```

也可以加入下面参数，让动作更平滑：

```bash
--no-fresh-inference-each-step
```

运行 `smolvla_ros_control.py` 时，终端会显示一个本地网页链接。可以 `Ctrl` + 点击进入页面查看相机画面和当前动作；不打开网页时，也可以直接看终端输出动作，并用 `rviz2` 查看相机画面。

## 关键调参项

真机部署时需要重点关注以下参数：

- `--hz`：控制频率。建议从 `1`、`2`、`5`、`10`、`15` 慢慢试，不要一次跳太大。训练集控制频率约为 30Hz，但真机应逐步逼近。
- `--max-steps`：执行动作步数。`100` 可用于观察一段动作片段，可按测试需要调整。
- `--max-joint-delta`：每一步关节位置变化限幅。提高控制频率时通常需要降低这个值。
- `--trajectory-duration`：每一步轨迹执行时间。机械臂抖动时可以和 `hz`、`max-joint-delta` 一起调整。
- `--gripper-*`：夹爪映射参数。当前训练集夹爪动作和本机夹爪命令不完全一致，需要继续标定。

目前观察到机械臂有向上抬起、抬肘等趋势，但抖动明显。可能原因包括：

- 训练数据集任务空间和当前机械臂、相机视角、场景布置差距较大。
- 推理和执行阶段暂未加入足够的平滑、滤波、动作重采样等技巧。
- 控制频率、单步限幅、轨迹持续时间还需要联合调参。
- 夹爪映射仍未完全标定。

因此，该 VLA 部署代码更适合作为可运行的初始版本，而不是最终稳定版本。

## 数据采集与后续训练

如果公开数据集和当前任务空间差异过大，建议使用本机双臂主从系统重新采集数据，再进行训练和部署。相关代码位于：

```text
/home/yangcw/openarm/lerobot_openarm-main
```

该目录包含 OpenArm 主从遥操作、LeRobot 数据集采集和训练相关说明，可优先阅读其中的 README。

参考数据集：

```text
https://huggingface.co/datasets/aShunSasaki/openarm_sponge_pick_and_place_50ep
```

该数据集可用于参考任务、相机视角和数据格式，但真机部署时仍建议尽量保证相机位置、任务物体、桌面布局、机械臂初始位姿与训练数据一致。

## 外部参考

- 抖动原因和控制优化参考：https://zhuanlan.zhihu.com/p/2021887597098607995
- 数据采集、训练、部署流程参考：https://space.bilibili.com/452287406/dynamic

## 安全提示

- 真机运行前确认急停可用，机械臂周围无人员和障碍物。
- 首次运行不要直接使用高频率、大步长或长时间动作。
- 任何会下发真机动作的命令都需要明确加 `--execute`，运行前再次确认参数。
- 如果机械臂明显抖动、速度异常或接近限位，应立即停止程序并断开执行。
