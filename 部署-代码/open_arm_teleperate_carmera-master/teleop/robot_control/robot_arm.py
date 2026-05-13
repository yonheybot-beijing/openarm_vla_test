import numpy as np
import threading
import time
from enum import IntEnum

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

# 初始化ROS2节点
import rclpy
from rclpy.node import Node
import json
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
# 导入多线程执行器
from rclpy.executors import MultiThreadedExecutor
import os
import sys
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

def create_joint_msg(joint_positions):
    msg = Float64MultiArray()
    msg.data = [float(p) for p in joint_positions]
    return msg


Openarm_Num_Motors = 14
unitree_gripper_indices = [4, 9] # [thumb, index]
Gripper_Num_Motors = 2


class MotorState:
    def __init__(self):
        self.q = None


class Open_arm_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Openarm_Num_Motors)]


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data


class Open_arm_ArmController:
    def __init__(self,left_hand_array, right_hand_array, dual_gripper_data_lock = None, dual_gripper_state_out = None, dual_gripper_action_out = None, 
                       filter = True):
        print("Initialize X1_ArmController...")
        # self.q_target = np.zeros(10)
        self.q_target = np.array([0] * 14)
        self.q_target[3] = 1.57
        self.q_target[3+7] = 1.57

        # self.kp_wrist = 40.0
        # self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 12  #
        self.control_dt = 1.0 / 100

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        self.joint_names = [
            'openarm_left_joint1',
            'openarm_left_joint2',
            'openarm_left_joint3',
            'openarm_left_joint4',
            'openarm_left_joint5',
            'openarm_left_joint6',
            'openarm_left_joint7',
            'openarm_right_joint1',
            'openarm_right_joint2',
            'openarm_right_joint3',
            'openarm_right_joint4',
            'openarm_right_joint5',
            'openarm_right_joint6',
            'openarm_right_joint7'
        ]
        self.dual_gripper_state = [0.0] * 2
        self.gripper_joint_names = {"openarm_left_finger_joint1":0,'openarm_right_finger_joint1':1}
        if filter:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), Gripper_Num_Motors)
        else:
            self.smooth_filter = None

        self.arm_state_buffer = DataBuffer()

        # 初始化ros2节点
        rclpy.init()
        self.node = Node('arm_controller')
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)

        # 创建订阅者，订阅关节状态
        self.subscription = self.node.create_subscription(
            JointState,
            '/joint_states',
            self.arm_state_callback,
            10
        )

        ## 发布关节状态
        self.left_pub = self.node.create_publisher(
            Float64MultiArray,
            '/left_forward_position_controller/commands',
            10
        )

        self.right_pub = self.node.create_publisher(
            Float64MultiArray,
            '/right_forward_position_controller/commands',
            10
        )

        # # 夹爪动作客户端
        self.left_gripper_client = ActionClient(self.node, GripperCommand, '/left_gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(self.node, GripperCommand, '/right_gripper_controller/gripper_cmd')
        self.left_gripper_client.wait_for_server()
        self.right_gripper_client.wait_for_server()

        # 创建一个线程来运行ROS2的spin
        self.ros_spin_thread = threading.Thread(target=self._ros_spin)
        self.ros_spin_thread.daemon = True
        self.ros_spin_thread.start()

        while not self.arm_state_buffer.GetData():
            time.sleep(0.01)
            print("[ArmController] Waiting to subscribe ros2...")

        print(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        print("Lock all joints except two arms...\n")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()


        

        # while True:
        #     if any(state != 0.0 for state in self.dual_gripper_state):
        #         break
        #     time.sleep(0.01)
        #     print("[Gripper_Controller] Waiting to subscribe dds...")
        
        # self.gripper_control_thread = threading.Thread(target=self.control_thread, args=(left_hand_array, right_hand_array, self.dual_gripper_state,
        #                                                                                  dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out))
        # self.gripper_control_thread.daemon = True
        # self.gripper_control_thread.start()
        

        print("Initialize ArmController OK!\n")

    def send_gripper_command(self, left_position, right_position, effort=10.0):
        """ 夹爪控制 """
        
        if left_position is not None:
            """ 左夹爪控制 """
            left_goal_msg = GripperCommand.Goal()
            left_goal_msg.command.position = left_position
            left_goal_msg.command.max_effort = effort

            future = self.left_gripper_client.send_goal_async(left_goal_msg)
            # future.add_done_callback(self.gripper_goal_response_callback)
        if right_position is not None:
            """ 右夹爪控制 """
            right_goal_msg = GripperCommand.Goal()
            right_goal_msg.command.position = right_position
            right_goal_msg.command.max_effort = effort

            future = self.right_gripper_client.send_goal_async(right_goal_msg)
        
    
    # def open_grippers(self, effort=10.0):
    #     """打开两侧夹爪"""
    #     self.send_gripper_command(0.044, 0.044, effort)
    
    def close_grippers(self, effort=2.0):
        """关闭两侧夹爪"""
        self.send_gripper_command(0.00, 0.00, effort)
    def open_left_gripper(self, effort=2.0):
        """ 单独打开左夹爪（右夹爪保持当前位置不变） """
        self.send_gripper_command(left_position=0.044, right_position=None, effort=effort)

    def close_left_gripper(self, effort=2.0):
        """ 单独关闭左夹爪（右夹爪保持当前位置不变） """
        self.send_gripper_command(left_position=0.00, right_position=None, effort=effort)


    def open_right_gripper(self, effort=2.0):
        """ 单独打开右夹爪（左夹爪保持当前位置不变） """
        self.send_gripper_command(left_position=None, right_position=0.044, effort=effort)


    def close_right_gripper(self, effort=2.0):
        """ 单独关闭右夹爪（左夹爪保持当前位置不变） """
        self.send_gripper_command(left_position=None, right_position=0.00, effort=effort)



    def joint_name_to_id(self, joint_name):
        """将ROS2关节名称映射到电机ID"""

        mapping = {
            "openarm_left_joint1": Openarm_JointArmIndex.openarm_left_joint1.value,
            "openarm_left_joint2": Openarm_JointArmIndex.openarm_left_joint2.value,
            "openarm_left_joint3": Openarm_JointArmIndex.openarm_left_joint3.value,
            "openarm_left_joint4": Openarm_JointArmIndex.openarm_left_joint4.value,
            "openarm_left_joint5": Openarm_JointArmIndex.openarm_left_joint5.value,
            "openarm_left_joint6": Openarm_JointArmIndex.openarm_left_joint6.value,
            "openarm_left_joint7": Openarm_JointArmIndex.openarm_left_joint7.value,
            "openarm_right_joint1": Openarm_JointArmIndex.openarm_right_joint1.value,
            "openarm_right_joint2": Openarm_JointArmIndex.openarm_right_joint2.value,
            "openarm_right_joint3": Openarm_JointArmIndex.openarm_right_joint3.value,
            "openarm_right_joint4": Openarm_JointArmIndex.openarm_right_joint4.value,
            "openarm_right_joint5": Openarm_JointArmIndex.openarm_right_joint5.value,
            "openarm_right_joint6": Openarm_JointArmIndex.openarm_right_joint6.value,
            "openarm_right_joint7": Openarm_JointArmIndex.openarm_right_joint7.value
        }
        return mapping.get(joint_name)

    # def _executor_spin(self):
    #     """运行ROS2的多线程执行器"""
    #     try:
    #         self.executor.spin()
    #     finally:
    #         # 确保在线程结束时关闭执行器
    #         self.executor.shutdown()
    def _ros_spin(self):
        """运行ROS2的spin循环来处理回调"""
        import rclpy
        while True:
            rclpy.spin_once(self.node, timeout_sec=0.001)
            time.sleep(0.001)  # 短暂休眠以避免CPU占用过高

    def arm_state_callback(self, msg):

        """处理臂关节状态回调"""
        if msg is not None:
            # 提取左臂关节位置
            
            lowstate = Open_arm_LowState()
            for i, name in enumerate(msg.name):
                if name in self.joint_names:
                    motor_id = self.joint_name_to_id(name)
                    lowstate.motor_state[motor_id].q = msg.position[i]
                if name in self.gripper_joint_names.keys():
                    motor_id = self.gripper_joint_names[name]
                    self.dual_gripper_state[motor_id] = msg.position[i]
            # 更新左臂状态缓冲区
        
            self.arm_state_buffer.SetData(lowstate)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)

        # print("target_q:",target_q)
        # print("clip_arm_q_target:",cliped_arm_q_target)
        return np.round(cliped_arm_q_target, 2)  # 保留两位小数

    def _ctrl_motor_state(self):

        while True:

            start_time = time.time()
            with self.ctrl_lock:
                arm_q_target = self.q_target

            cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit=self.arm_velocity_limit)

            left_joint_msg = create_joint_msg(cliped_arm_q_target[:7])
            right_joint_msg = create_joint_msg(cliped_arm_q_target[7:])
            # print(left_joint_msg)
            self.left_pub.publish(left_joint_msg)
            self.right_pub.publish(right_joint_msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 5 + (1 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # print(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # print(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target):

        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
           

    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''

        arm_joints = np.array([
            self.arm_state_buffer.GetData().motor_state[id].q
            for id in Openarm_JointArmIndex
        ])

        return arm_joints

    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        print("[ArmController] ctrl_dual_arm_go_home start...")
        with self.ctrl_lock:
            # self.q_target = np.zeros(10)
            self.q_target = np.array([0.0] * 14)
            # self.tauff_target = np.zeros(10)
        tolerance = 0.01  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements

        while True:
            current_q = self.get_current_dual_arm_q()
            absolute_diff = np.abs(current_q - self.q_target)

            is_similar = np.all(absolute_diff < tolerance)

            if is_similar:
                print("[ArmController] both arms have reached the home position.")
                break
            time.sleep(0.05)

    def speed_gradual_max(self, t=5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 12

    ##控制夹爪
    def control_thread(self, left_hand_array, right_hand_array, dual_gripper_state_in, dual_hand_data_lock = None, 
                             dual_gripper_state_out = None, dual_gripper_action_out = None):
        self.running = True

        #爪机变化速率
        DELTA_GRIPPER_CMD = 0.035         # The motor rotates 5.4 radians, the clamping jaw slide open 9 cm, so 0.6 rad <==> 1 cm, 0.18 rad <==> 3 mm
        #VR中拇指和食指的欧式距离范围
        THUMB_INDEX_DISTANCE_MIN = 0.005  # Assuming a minimum Euclidean distance is 5 cm between thumb and index. 0.0047 0.0032
        THUMB_INDEX_DISTANCE_MAX = 0.09  # Assuming a maximum Euclidean distance is 9 cm between thumb and index. 0.0945 0.105 0.0989
        ##爪机电机的弧度范围
        LEFT_MAPPED_MIN  = 0.0           # The minimum initial motor position when the gripper closes at startup.
        RIGHT_MAPPED_MIN = 0.0           # The minimum initial motor position when the gripper closes at startup.
        # The maximum initial motor position when the gripper closes before calibration (with the rail stroke calculated as 0.6 cm/rad * 9 rad = 5.4 cm).
        LEFT_MAPPED_MAX = LEFT_MAPPED_MIN + 0.044
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 0.044

        left_target_action  = (LEFT_MAPPED_MAX - LEFT_MAPPED_MIN) / 2.0
        right_target_action = (RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN) / 2.0


        try:
            while self.running:
             
                start_time = time.time()
                # get dual hand skeletal point state from XR device
                left_hand_mat  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                right_hand_mat = np.array(right_hand_array[:]).reshape(25, 3).copy()
             
                # if not np.array_equal(left_hand, np.zeros((25, 3))) and not np.array_equal(right_hand, np.zeros((25, 3))):
                left_euclidean_distance  = np.linalg.norm(left_hand_mat[unitree_gripper_indices[1]] - left_hand_mat[unitree_gripper_indices[0]])
                right_euclidean_distance = np.linalg.norm(right_hand_mat[unitree_gripper_indices[1]] - right_hand_mat[unitree_gripper_indices[0]])
                left_target_action  = np.interp(left_euclidean_distance, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX])
                right_target_action = np.interp(right_euclidean_distance, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX])

              

                # get current dual gripper motor state
                dual_gripper_state = np.array(dual_gripper_state_in[:])

                # clip dual gripper action to avoid overflow
                left_actual_action  = np.clip(left_target_action,  dual_gripper_state[0] - DELTA_GRIPPER_CMD, dual_gripper_state[0] + DELTA_GRIPPER_CMD) 
                right_actual_action = np.clip(right_target_action, dual_gripper_state[1] - DELTA_GRIPPER_CMD, dual_gripper_state[1] + DELTA_GRIPPER_CMD)

                dual_gripper_action = np.array([left_actual_action,right_actual_action])

                if self.smooth_filter:
                    self.smooth_filter.add_data(dual_gripper_action)
                    dual_gripper_action = self.smooth_filter.filtered_data

                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array([LEFT_MAPPED_MIN,RIGHT_MAPPED_MIN])
                        dual_gripper_action_out[:] = dual_gripper_action - np.array([LEFT_MAPPED_MIN,RIGHT_MAPPED_MIN])
                # if left_euclidean_distance!=0.0:
                #     print("left_euclidean_distance：",left_euclidean_distance)
                # print(f"LEFT: euclidean:{left_euclidean_distance:.4f} \tstate:{dual_gripper_state_out[1]:.4f}\
                #       \ttarget_action:{right_target_action - RIGHT_MAPPED_MIN:.4f} \tactual_action:{dual_gripper_action_out[1]:.4f}")
                # print(f"RIGHT:euclidean:{right_euclidean_distance:.4f} \tstate:{dual_gripper_state_out[0]:.4f}\
                #       \ttarget_action:{left_target_action - LEFT_MAPPED_MIN:.4f} \tactual_action:{dual_gripper_action_out[0]:.4f}")
                
                # self.ctrl_dual_gripper(dual_gripper_action)
                self.send_gripper_command(dual_gripper_action[0], dual_gripper_action[1])

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1.0/10 - time_elapsed))
                
                time.sleep(sleep_time)
        finally:
            print("Gripper_Controller has been closed.")

class Openarm_JointArmIndex(IntEnum):
    # OpenARM
    openarm_left_joint1 = 0
    openarm_left_joint2 = 1
    openarm_left_joint3 = 2
    openarm_left_joint4 = 3
    openarm_left_joint5 = 4
    openarm_left_joint6 = 5
    openarm_left_joint7 = 6
    openarm_right_joint1 = 7
    openarm_right_joint2 = 8
    openarm_right_joint3 = 9
    openarm_right_joint4 = 10
    openarm_right_joint5 = 11
    openarm_right_joint6 = 12
    openarm_right_joint7 = 13




# import numpy as np
# import threading
# import time
# from enum import IntEnum

# from std_msgs.msg import Float64MultiArray
# from sensor_msgs.msg import JointState
# from builtin_interfaces.msg import Duration

# # 初始化ROS2节点
# import rclpy
# from rclpy.node import Node
# import json
# from rclpy.action import ActionClient
# from control_msgs.action import FollowJointTrajectory, GripperCommand


# def create_joint_msg(joint_positions):
#     msg = Float64MultiArray()
#     msg.data = [float(p) for p in joint_positions]
#     return msg


# Openarm_Num_Motors = 14


# class MotorState:
#     def __init__(self):
#         self.q = None


# class Open_arm_LowState:
#     def __init__(self):
#         self.motor_state = [MotorState() for _ in range(Openarm_Num_Motors)]


# class DataBuffer:
#     def __init__(self):
#         self.data = None
#         self.lock = threading.Lock()

#     def GetData(self):
#         with self.lock:
#             return self.data

#     def SetData(self, data):
#         with self.lock:
#             self.data = data


# class Open_arm_ArmController:
#     def __init__(self):
#         print("Initialize X1_ArmController...")
#         # self.q_target = np.zeros(10)
#         self.q_target = np.array([0] * 14)
#         self.q_target[3] = 1.57
#         self.q_target[3+7] = 1.57

#         # self.kp_wrist = 40.0
#         # self.kd_wrist = 1.5

#         self.all_motor_q = None
#         self.arm_velocity_limit = 12  #
#         self.control_dt = 1.0 / 100

#         self._speed_gradual_max = False
#         self._gradual_start_time = None
#         self._gradual_time = None

#         self.joint_names = [
#             'openarm_left_joint1',
#             'openarm_left_joint2',
#             'openarm_left_joint3',
#             'openarm_left_joint4',
#             'openarm_left_joint5',
#             'openarm_left_joint6',
#             'openarm_left_joint7',
#             'openarm_right_joint1',
#             'openarm_right_joint2',
#             'openarm_right_joint3',
#             'openarm_right_joint4',
#             'openarm_right_joint5',
#             'openarm_right_joint6',
#             'openarm_right_joint7'
#         ]
#         self.arm_state_buffer = DataBuffer()

#         # 初始化ros2节点
#         rclpy.init()
#         self.node = Node('arm_controller')

#         # 创建订阅者，订阅关节状态
#         self.subscription = self.node.create_subscription(
#             JointState,
#             '/joint_states',
#             self.arm_state_callback,
#             10
#         )

#         ## 发布关节状态
#         self.left_pub = self.node.create_publisher(
#             Float64MultiArray,
#             '/left_forward_position_controller/commands',
#             10
#         )

#         self.right_pub = self.node.create_publisher(
#             Float64MultiArray,
#             '/right_forward_position_controller/commands',
#             10
#         )

#         # 夹爪动作客户端
#         self.left_gripper_client = ActionClient(self.node, GripperCommand, '/left_gripper_controller/gripper_cmd')
#         self.right_gripper_client = ActionClient(self.node, GripperCommand, '/right_gripper_controller/gripper_cmd')

#         # 创建一个线程来运行ROS2的spin
#         self.ros_spin_thread = threading.Thread(target=self._ros_spin)
#         self.ros_spin_thread.daemon = True
#         self.ros_spin_thread.start()

#         while not self.arm_state_buffer.GetData():
#             time.sleep(0.01)
#             print("[ArmController] Waiting to subscribe ros2...")

#         print(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
#         print("Lock all joints except two arms...\n")

#         # initialize publish thread
#         self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
#         self.ctrl_lock = threading.Lock()
#         self.publish_thread.daemon = True
#         self.publish_thread.start()

#         print("Initialize ArmController OK!\n")

#     def send_gripper_command(self, left_position, right_position, effort=10.0):
#         """ 夹爪控制 """
#         self.left_gripper_client.wait_for_server()
#         self.right_gripper_client.wait_for_server()
#         if left_position is not None:
#             """ 左夹爪控制 """
#             left_goal_msg = GripperCommand.Goal()
#             left_goal_msg.command.position = left_position
#             left_goal_msg.command.max_effort = effort

#             future = self.left_gripper_client.send_goal_async(left_goal_msg)
#             # future.add_done_callback(self.gripper_goal_response_callback)
#         if right_position is not None:
#             """ 右夹爪控制 """
#             right_goal_msg = GripperCommand.Goal()
#             right_goal_msg.command.position = right_position
#             right_goal_msg.command.max_effort = effort

#             future = self.right_gripper_client.send_goal_async(right_goal_msg)
        
    
#     def open_grippers(self, effort=1.0):
#         """打开两侧夹爪"""
#         self.send_gripper_command(0.04, 0.04, effort)
    
#     def close_grippers(self, effort=1.0):
#         """关闭两侧夹爪"""
#         self.send_gripper_command(0.00, 0.00, effort)
#     def open_left_gripper(self, effort=1.0):
#         """ 单独打开左夹爪（右夹爪保持当前位置不变） """
       
#         self.send_gripper_command(left_position=0.04, right_position=None, effort=effort)


#     def close_left_gripper(self, effort=1.0):
#         """ 单独关闭左夹爪（右夹爪保持当前位置不变） """
#         self.send_gripper_command(left_position=0.00, right_position=None, effort=effort)


#     def open_right_gripper(self, effort=1.0):
#         """ 单独打开右夹爪（左夹爪保持当前位置不变） """
#         self.send_gripper_command(left_position=None, right_position=0.04, effort=effort)


#     def close_right_gripper(self, effort=1.0):
#         """ 单独关闭右夹爪（左夹爪保持当前位置不变） """
#         self.send_gripper_command(left_position=None, right_position=0.00, effort=effort)

#     def joint_name_to_id(self, joint_name):
#         """将ROS2关节名称映射到电机ID"""

#         mapping = {
#             "openarm_left_joint1": Openarm_JointArmIndex.openarm_left_joint1.value,
#             "openarm_left_joint2": Openarm_JointArmIndex.openarm_left_joint2.value,
#             "openarm_left_joint3": Openarm_JointArmIndex.openarm_left_joint3.value,
#             "openarm_left_joint4": Openarm_JointArmIndex.openarm_left_joint4.value,
#             "openarm_left_joint5": Openarm_JointArmIndex.openarm_left_joint5.value,
#             "openarm_left_joint6": Openarm_JointArmIndex.openarm_left_joint6.value,
#             "openarm_left_joint7": Openarm_JointArmIndex.openarm_left_joint7.value,
#             "openarm_right_joint1": Openarm_JointArmIndex.openarm_right_joint1.value,
#             "openarm_right_joint2": Openarm_JointArmIndex.openarm_right_joint2.value,
#             "openarm_right_joint3": Openarm_JointArmIndex.openarm_right_joint3.value,
#             "openarm_right_joint4": Openarm_JointArmIndex.openarm_right_joint4.value,
#             "openarm_right_joint5": Openarm_JointArmIndex.openarm_right_joint5.value,
#             "openarm_right_joint6": Openarm_JointArmIndex.openarm_right_joint6.value,
#             "openarm_right_joint7": Openarm_JointArmIndex.openarm_right_joint7.value
#         }
#         return mapping.get(joint_name)

#     def _ros_spin(self):
#         """运行ROS2的spin循环来处理回调"""
#         import rclpy
#         while True:
#             rclpy.spin_once(self.node, timeout_sec=0.001)
#             time.sleep(0.001)  # 短暂休眠以避免CPU占用过高

#     def arm_state_callback(self, msg):

#         """处理臂关节状态回调"""
#         if msg is not None:
#             # 提取左臂关节位置
#             lowstate = Open_arm_LowState()
#             for i, name in enumerate(msg.name):
#                 if name in self.joint_names:
#                     motor_id = self.joint_name_to_id(name)
#                     lowstate.motor_state[motor_id].q = msg.position[i]
#             # 更新左臂状态缓冲区
#             self.arm_state_buffer.SetData(lowstate)

#     def clip_arm_q_target(self, target_q, velocity_limit):
#         current_q = self.get_current_dual_arm_q()
#         delta = target_q - current_q
#         motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
#         cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)

#         # print("target_q:",target_q)
#         # print("clip_arm_q_target:",cliped_arm_q_target)
#         return np.round(cliped_arm_q_target, 2)  # 保留两位小数

#     def _ctrl_motor_state(self):

#         while True:

#             start_time = time.time()
#             with self.ctrl_lock:
#                 arm_q_target = self.q_target

#             cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit=self.arm_velocity_limit)

#             left_joint_msg = create_joint_msg(cliped_arm_q_target[:7])
#             right_joint_msg = create_joint_msg(cliped_arm_q_target[7:])
#             # print(left_joint_msg)
#             self.left_pub.publish(left_joint_msg)
#             self.right_pub.publish(right_joint_msg)

#             if self._speed_gradual_max is True:
#                 t_elapsed = start_time - self._gradual_start_time
#                 self.arm_velocity_limit = 8 + (1 * min(1.0, t_elapsed / 5.0))

#             current_time = time.time()
#             all_t_elapsed = current_time - start_time
#             sleep_time = max(0, (self.control_dt - all_t_elapsed))
#             time.sleep(sleep_time)
#             # print(f"arm_velocity_limit:{self.arm_velocity_limit}")
#             # print(f"sleep_time:{sleep_time}")

#     def ctrl_dual_arm(self, q_target):

#         '''Set control target values q & tau of the left and right arm motors.'''
#         with self.ctrl_lock:
#             self.q_target = q_target
           

#     def get_current_dual_arm_q(self):
#         '''Return current state q of the left and right arm motors.'''

#         arm_joints = np.array([
#             self.arm_state_buffer.GetData().motor_state[id].q
#             for id in Openarm_JointArmIndex
#         ])

#         return arm_joints

#     def ctrl_dual_arm_go_home(self):
#         '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
#         print("[ArmController] ctrl_dual_arm_go_home start...")
#         with self.ctrl_lock:
#             # self.q_target = np.zeros(10)
#             self.q_target = np.array([0.0] * 14)
#             # self.tauff_target = np.zeros(10)
#         tolerance = 0.01  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements

#         while True:
#             current_q = self.get_current_dual_arm_q()
#             absolute_diff = np.abs(current_q - self.q_target)

#             is_similar = np.all(absolute_diff < tolerance)

#             if is_similar:
#                 print("[ArmController] both arms have reached the home position.")
#                 break
#             time.sleep(0.05)

#     def speed_gradual_max(self, t=5.0):
#         '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
#         self._gradual_start_time = time.time()
#         self._gradual_time = t
#         self._speed_gradual_max = True

#     def speed_instant_max(self):
#         '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
#         self.arm_velocity_limit = 12


# class Openarm_JointArmIndex(IntEnum):
#     # OpenARM
#     openarm_left_joint1 = 0
#     openarm_left_joint2 = 1
#     openarm_left_joint3 = 2
#     openarm_left_joint4 = 3
#     openarm_left_joint5 = 4
#     openarm_left_joint6 = 5
#     openarm_left_joint7 = 6
#     openarm_right_joint1 = 7
#     openarm_right_joint2 = 8
#     openarm_right_joint3 = 9
#     openarm_right_joint4 = 10
#     openarm_right_joint5 = 11
#     openarm_right_joint6 = 12
#     openarm_right_joint7 = 13




