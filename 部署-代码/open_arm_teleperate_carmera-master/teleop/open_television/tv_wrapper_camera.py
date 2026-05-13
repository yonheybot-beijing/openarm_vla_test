import numpy as np
import os 
import sys
import math
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append("/home/openarm/avp_teleoperate_open_arm/avp_teleoperate_open_arm_camera")
from teleop.open_television.constants import *
from teleop.utils.mat_tool import mat_update, fast_mat_inv
import time
from multiprocessing import Array, Process, shared_memory, Lock
import asyncio
import cv2
import pyrealsense2 as rs
from multiprocessing import context
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from threading import Thread, Event
from queue import Queue, Empty, Full
import quaternion
import threading
import queue
# from utils import parse_keypoint_2d,parse_keypoint_3d,calculate_hand_rot,robust_depth_m,center_uv,LandmarkSmoother
from teleop.open_television.utils import parse_keypoint_2d,parse_keypoint_3d,calculate_hand_rot,robust_depth_m,center_uv,LandmarkSmoother,KalmanFilterDepth
from teleop.utils.filter import LowPassFilter
from teleop.open_television.hand_rot_smooth import rotation_matrix_to_quaternion_stable, SmoothRotator
from concurrent.futures import ThreadPoolExecutor


MEDIAPIPE_POSE_KEYPOINTS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]   # 33

MEDIAPIPE_HAND_KEYPOINTS = [
    "wrist", "thumb1", "thumb2", "thumb3", "thumb4",
    "index1", "index2", "index3", "index4",
    "middle1", "middle2", "middle3", "middle4",
    "ring1", "ring2", "ring3", "ring4",
    "pinky1", "pinky2", "pinky3", "pinky4"
]   # 21
"""
(basis) OpenXR Convention : y up, z back, x right. 
(basis) Robot  Convention : z up, y left, x front.  
p.s. Vuer's all raw data follows OpenXR Convention, WORLD coordinate.

under (basis) Robot Convention, wrist's initial pose convention:

    # (Left Wrist) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from index toward pinky.
        - the z-axis pointing from palm toward back of the hand.

    # (Right Wrist) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from pinky toward index.
        - the z-axis pointing from palm toward back of the hand.
  
    # (Left Wrist URDF) Unitree Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from palm toward back of the hand.
        - the z-axis pointing from pinky toward index.

    # (Right Wrist URDF) Unitree Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from back of the hand toward palm. 
        - the z-axis pointing from pinky toward index.

under (basis) Robot Convention, hand's initial pose convention:

    # (Left Hand) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from index toward pinky.
        - the z-axis pointing from palm toward back of the hand.

    # (Right Hand) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from pinky toward index.
        - the z-axis pointing from palm toward back of the hand.

    # (Left Hand URDF) Unitree Convention:   
        - The x-axis pointing from palm toward back of the hand. 
        - The y-axis pointing from middle toward wrist.
        - The z-axis pointing from pinky toward index.

    # (Right Hand URDF) Unitree Convention: 
        - The x-axis pointing from palm toward back of the hand. 
        - The y-axis pointing from middle toward wrist.
        - The z-axis pointing from index toward pinky. 

    p.s. From website: https://registry.khronos.org/OpenXR/specs/1.1/man/html/openxr.html.
         You can find **(Left/Right Wrist) XR/AppleVisionPro Convention** related information like this below:
           "The wrist joint is located at the pivot point of the wrist, which is location invariant when twisting the hand without moving the forearm. 
            The backward (+Z) direction is parallel to the line from wrist joint to middle finger metacarpal joint, and points away from the finger tips. 
            The up (+Y) direction points out towards back of the hand and perpendicular to the skin at wrist. 
            The X direction is perpendicular to the Y and Z directions and follows the right hand rule."
         Note: The above context is of course under **(basis) OpenXR Convention**.

    p.s. **(Wrist/Hand URDF) Unitree Convention** information come from URDF files.
"""

'''
Openarm:
(basis) OpenXR Convention : y up, z back, x right.   z qian y shang x left
机器人坐标系x：前,y：左,z：上
x 前 y 右，z下

手腕坐标系:

XR/AppleVisionPro 规范（左右手共性）：
左：
X 轴统一从手腕指向中指方向
Z 轴均从掌心指向手背方向
Y 轴：食指指向小指
右：
X 轴统一从手腕指向中指方向
Z 轴均从掌心指向手背方向
Y 轴：小指指向食指

openarm:
左：
x；小拇指到大拇指
y：掌背到掌内
z：腕到指
右：
x；小拇指到大拇指
y：掌内到掌背
z：腕到指

手部坐标系：
XR/AppleVisionPro 规范（左右手共性）：
左：
X 轴统一从手腕指向中指方向
Z 轴均从掌心指向手背方向
Y 轴：食指指向小指
右：
X 轴统一从手腕指向中指方向
Z 轴均从掌心指向手背方向
Y 轴：小指指向食指

openarm
左：
x；小拇指到大拇指
y：掌背到掌内
z：腕到指
右：
x；小拇指到大拇指
y：掌内到掌背
z：腕到指



'''

class TeleVisionWrapper:
    def __init__(self, img_shape):
                
        self.im_width,self.im_height = img_shape[0], img_shape[1]
        ## 初始化旋转矩阵
        self.last_left_wrist_rot = np.eye(3)
        self.last_right_wrist_rot = np.eye(3)

        ## 初始化历史深度
        self.last_head_depth = 0.0
        self.last_left_depth = 0.0
        self.last_right_depth = 0.0
        self.last_left2hand_depth = 0.0
        self.last_right2hand_depth = 0.0

        # 初始化卡尔曼滤波器用于深度平滑
        # 头部相对稳定，使用较小的过程噪声
        self.kalman_head = KalmanFilterDepth(
            process_noise=0.001, 
            measurement_noise=0.05, 
            error_estimate=0.05
        )
        # 手部运动较频繁，使用较大的过程噪声以快速响应运动
        self.kalman_left = KalmanFilterDepth(
            process_noise=0.01, 
            measurement_noise=0.08, 
            error_estimate=0.1,
            acceleration_noise=0.002
        )
        self.kalman_right = KalmanFilterDepth(
            process_noise=0.01, 
            measurement_noise=0.08, 
            error_estimate=0.1,
            acceleration_noise=0.002
        )
        
        # 设置深度一致性阈值为0.6米（根据用户提示）
        self.depth_consistency_threshold = 0.56
        
        # 添加运动检测标志
        self.is_left_moving = False
        self.is_right_moving = False
        self.movement_threshold = 0.02  # 2厘米作为运动检测阈值
        self.last_left_velocity = 0
        self.last_right_velocity = 0
        
        # 初始化模型
        self.detector_pose = mp.solutions.pose.Pose(
            static_image_mode=False,  # 视频流模式
            model_complexity=1,       # 0: Lite, 1: Full, 2: Heavy
            smooth_landmarks=True,    # 平滑关键点
            enable_segmentation=False, # 是否输出分割掩码
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector_hand = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ## # 用于从检测线程接收结果的队列
        # self.pose_result_queue = queue.Queue(maxsize=1)
        # self.hand_result_queue = queue.Queue(maxsize=1)
        # # 用于通知主线程检测已完成的事件
        # self.pose_event = threading.Event()
        # self.hand_event = threading.Event()
        # # 用于从检测线程接收结果的队列
        # self.pose_result_queue = queue.Queue(maxsize=1)
        # self.hand_result_queue = queue.Queue(maxsize=1)
        # # 用于通知主线程检测已完成的事件
        # self.pose_event = threading.Event()
        # self.hand_event = threading.Event()

        ## 使用多线程进行AI模型推理
        # self.executor = ThreadPoolExecutor(max_workers=2)




        ## 可视化关键点相关变量
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        ## 初始化检测结果
        self.detection_result_pose = None
        self.detection_result_hand = None

        ##平滑pose
        self.filter_2d = LowPassFilter(alpha=0.5)
        self.filter_3d = LowPassFilter(alpha=0.5)

        ## 手部初始状态
        '''
        双手抬起大臂与小臂垂直
            ###
             #  
         ##  #  ##
         #       #
         # # #   # # #  
             #
             #
            # #
          #     #
        '''
        self.left_hand = np.array([[0, -1, 0, 0.14],
                                   [0, 0, -1, 0.3],
                                   [1, 0, 0, -0.23],
                                   [0, 0, 0, 1]])
        self.right_hand = np.array([[0, -1, 0, -0.14],
                            [0, 0, -1, 0.3],
                            [1, 0, 0, -0.23],
                            [0, 0, 0, 1]])

        self.left_landmarks = np.zeros((21,3))
        self.right_landmarks = np.zeros((21,3))

        ###手部旋转矩阵平滑
        ## 手部初始四元数
        self.valid_orientation_left = rotation_matrix_to_quaternion_stable(self.left_hand[:3, :3])
        self.valid_orientation_right = rotation_matrix_to_quaternion_stable(self.right_hand[:3, :3])
        self.last_valid_orientation_left = rotation_matrix_to_quaternion_stable(self.left_hand[:3, :3])
        self.last_valid_orientation_right = rotation_matrix_to_quaternion_stable(self.right_hand[:3, :3])

        self.hand_smoothing_factor = 0.3 # 平滑系数 (0-1)，值越小越平滑
        self.left_hand_smoother = SmoothRotator(self.hand_smoothing_factor)
        self.right_hand_smoother = SmoothRotator(self.hand_smoothing_factor)

        ## 深度缩放因子
        self.depth_scale =  0.0010000000474974513
        # 用于从检测线程接收结果的队列
        self.pose_result_queue = queue.Queue(maxsize=1)
        self.hand_result_queue = queue.Queue(maxsize=1)
        # 用于通知主线程检测已完成的事件
        self.pose_event = threading.Event()
        self.hand_event = threading.Event()

        ## 手臂伸直阈值
        self.straight_threshold = 0.8
        self.is_straight_left = False
        self.is_straight_right = False
       


    def pose_detection_thread(self,frame_rgb):
        result = self.detector_pose.process(frame_rgb)
        try:
            self.pose_result_queue.put(result, block=False)
        except queue.Full:
            self.pose_result_queue.get()
            self.pose_result_queue.put(result)
        self.pose_event.set()

    def hand_detection_thread(self,frame_rgb):
        result = self.detector_hand.process(frame_rgb)
        try:
            self.hand_result_queue.put(result, block=False)
        except queue.Full:
            self.hand_result_queue.get()
            self.hand_result_queue.put(result)
        self.hand_event.set()

    def get_data(self,color_bgr,depth_image):
        # --------------------------------wrist-------------------------------------
       
        image_show = self._detection_processor(color_bgr,depth_image)
        


        # TeleVision obtains a basis coordinate that is OpenXR Convention
        left_wrist_vuer_mat, left_wrist_flag  = mat_update(const_left_wrist_vuer_mat, self.left_hand.copy())
        right_wrist_vuer_mat, right_wrist_flag = mat_update(const_right_wrist_vuer_mat, self.right_hand.copy())
    
        T_robot_openxr = np.array([[0, 0, -1, 0],
                                   [1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, 0, 1]])
        
        '''
        openarm:x：前,y：左,z：上
        mediapipe:后（正方向）z
                下（正方向）y
                左（正方向）x
        '''


        # head_mat = T_robot_openxr @ head_vuer_mat @ fast_mat_inv(T_robot_openxr)
        left_wrist_mat  = T_robot_openxr @ left_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)
        right_wrist_mat = T_robot_openxr @ right_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)

        T_to_openarm_left_wrist = np.array([
            [0, -1, 0, 0],   
            [0, 0, -1, 0],   
            [1, 0, 0, 0],  
            [0, 0, 0, 1] 
            ]
        )  
        T_to_openarm_right_wrist = np.array([
            [0, -1, 0, 0],   
            [0, 0, -1, 0],   
            [1, 0, 0, 0],  
            [0, 0, 0, 1] 
            ]
        )  



        unitree_left_wrist = left_wrist_mat @ (T_to_openarm_left_wrist)
        unitree_right_wrist = right_wrist_mat @ (T_to_openarm_right_wrist)


        
        T_rotate_left = np.array([
            [0, 1, 0, 0],   
            [-1, 0, 0, 0],   
            [0, 0, 1, 0],  
            [0, 0, 0, 1] 
            ]
        )  
        T_rotate_right = np.array([
            [0, 1, 0, 0],   
            [-1, 0, 0, 0],   
            [0, 0, 1, 0],  
            [0, 0, 0, 1] 
            ]
        )  

        ## 围绕哪个轴顺时针和逆时针旋转，从这个轴看原点.
        ## 修改映射矩阵，左手腕绕 z 轴逆时针旋转 90°，T_rotate_left为顺时针，求逆为逆时针，
        unitree_left_wrist =   unitree_left_wrist @ fast_mat_inv(T_rotate_left)
        ## 修改映射矩阵，右手腕绕 z 轴顺时针旋转 90°，T_rotate_left为逆时针，求逆位顺时针，
        unitree_right_wrist = unitree_right_wrist@ fast_mat_inv(T_rotate_right)

        # unitree_left_wrist = left_wrist_mat @ (T_to_openarm_left_wrist if left_wrist_flag else np.eye(4))
        # unitree_right_wrist = right_wrist_mat @ (T_to_openarm_right_wrist if right_wrist_flag else np.eye(4))



        # Transfer from WORLD to HEAD coordinate (translation only).
        # unitree_left_wrist[0:3, 3]  = unitree_left_wrist[0:3, 3] - head_mat[0:3, 3]
        # unitree_right_wrist[0:3, 3] = unitree_right_wrist[0:3, 3] - head_mat[0:3, 3]

        # --------------------------------hand-------------------------------------

        # Homogeneous, [xyz] to [xyz1]
        # p.s. np.concatenate([25,3]^T,(1,25)) ==> hand_vuer_mat.shape is (4,25)
        # Now under (basis) OpenXR Convention, mat shape like this:
        #    x0 x1 x2 ··· x23 x24
        #    y0 y1 y1 ··· y23 y24
        #    z0 z1 z2 ··· z23 z24
        #     1  1  1 ···   1   1
        if left_wrist_flag and right_wrist_flag:
            unitree_left_hand  = self.left_landmarks.copy()
            unitree_right_hand = self.right_landmarks.copy()
            if not np.array_equal(unitree_left_hand, np.zeros((21, 3))) and not np.array_equal(unitree_right_hand, np.zeros((21, 3))):

                self.left_gesture = self.recognize_gesture(unitree_left_hand)
                self.right_gesture = self.recognize_gesture(unitree_right_hand)
            else:
                self.left_gesture = "open"
                self.right_gesture = "open"


            # --------------------------------offset-------------------------------------
            # print(unitree_left_wrist)
            # head_rmat = head_mat[:3, :3]
            # The origin of the coordinate for IK Solve is the WAIST joint motor. You can use teleop/robot_control/robot_arm_ik.py Unit_Test to check it.
            # The origin of the coordinate of unitree_left_wrist is HEAD. So it is necessary to translate the origin of unitree_left_wrist from HEAD to WAIST.
            # unitree_left_wrist[0, 3] +=0.15
            # unitree_right_wrist[0,3] +=0.15
            unitree_left_wrist[2, 3] +=0.72
            unitree_right_wrist[2,3] +=0.72
            # unitree_left_wrist[0, 3] -=0.45
            # unitree_right_wrist[0,3] -=0.45

            ## 处理异常数据
            if unitree_left_wrist[0, 3]>0.48:
                unitree_left_wrist[0, 3] = 0.48
            if unitree_right_wrist[0, 3]>0.48:
                unitree_right_wrist[0, 3] = 0.48
            if unitree_left_wrist[0,3]<0:
                unitree_left_wrist[0, 3] = 0
            if unitree_right_wrist[0,3]<0:
                unitree_right_wrist[0, 3] = 0
             
            if self.is_straight_left:
                left_loc = unitree_left_wrist[:,3]
                is_vaild = left_loc[0] > 0 and left_loc[2] < 0.15 and left_loc[1]>0.21 and left_loc[1]<0.3 and left_loc[2]>0.0 and left_loc[2]<0.2
            
                if is_vaild:
                    unitree_left_wrist[:3,:3] = np.array(
                        [
                            [1., 0. , 0. ],
                            [0., -1., 0. ],
                            [0., 0. , -1.],
                        ]
                    )
                    
                if self.is_straight_right:
                    right_loc = unitree_right_wrist[:,3]
                    is_vaild = right_loc[0] < 0 and right_loc[2] < 0.15 and right_loc[1]>-0.3 and right_loc[1]<-0.21 and right_loc[2]>0.0 and right_loc[2]<0.2
                    if is_vaild:
                        unitree_right_wrist[:3,:3] = np.array(
                            [
                                [1., 0. , 0. ],
                                [0., -1., 0. ],
                                [0., 0. , -1.],
                            ]
                        )
                        

                # print(unitree_left_wrist[:,-1])
            return  unitree_left_wrist, unitree_right_wrist, self.left_gesture, self.right_gesture,image_show


        else:
            self.left_gesture = "open"
            self.right_gesture = "open"
            
            unitree_left_wrist = np.array(
                [
                    [1., 0. , 0. , 0.],
                    [0., -1., 0. , 0.14],
                    [0., 0. , -1., 0.14],
                    [0., 0. , 0. , 1.],
                ]
            )
            unitree_right_wrist = np.array(
                [
                    [1., 0. , 0. , 0.],
                    [0., -1., 0. , -0.14],
                    [0., 0. , -1., 0.14],
                    [0., 0. , 0. , 1.],
                ]
            )
            return  unitree_left_wrist, unitree_right_wrist, self.left_gesture, self.right_gesture,image_show

        
        #

    def _detection_processor(self,color_bgr,depth):
        """处理图像并计算位姿"""
        # --- 关键时间点 0: 方法开始 ---
        start_total_time = time.time()
        
        image_show = color_bgr.copy()

        # --- 关键时间点 A: 颜色空间转换 ---
        start_time = time.time()
        image_show = color_bgr.copy()

        frame_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        # print(f"[耗时] A. 颜色空间转换 (BGR -> RGB): {time.time() - start_time:.4f} 秒")
        
        # 重置事件标志
        self.pose_event.clear()
        self.hand_event.clear()
        start_time = time.time()
        # 启动两个检测线程
        pose_thread = threading.Thread(target=self.pose_detection_thread, args=(frame_rgb.copy(),))
        hand_thread = threading.Thread(target=self.hand_detection_thread, args=(frame_rgb.copy(),))
        pose_thread.start()
        hand_thread.start()
        
        # 等待两个线程完成，设置超时避免无限等待
        self.pose_event.wait(timeout=0.1)
        self.hand_event.wait(timeout=0.1)
        
        # 获取检测结果
        self.detection_result_pose = self.pose_result_queue.get() if not self.pose_result_queue.empty() else None
        self.detection_result_hand = self.hand_result_queue.get() if not self.hand_result_queue.empty() else None
        # print(f"[耗时] 等待两个线程完成: {time.time() - start_time:.4f} 秒")



        # # --- 关键时间点 B: AI模型推理 (Pose + Hand) ---
        # start_time = time.time()
        # self.detection_result_pose = self.detector_pose.process(frame_rgb)
        # print(f"[耗时] B.1 身体关键点跟踪与处理: {time.time() - start_time:.4f} 秒")
        # start_time = time.time()
        # self.detection_result_hand = self.detector_hand.process(frame_rgb)
        # print(f"[耗时] B.2 手部关键点跟踪与处理: {time.time() - start_time:.4f} 秒")
        
       



        # --- 关键时间点 C: 身体关键点跟踪与处理 ---
        start_time = time.time()
        keypoint_2d_body, keypoint_2d_body_array, keypoint_3d_body, keypoint_3d_body_array, visib = self.body_track()
        # print(f"[耗时] C. 身体关键点跟踪与处理: {time.time() - start_time:.4f} 秒")
         # --- 关键时间点 D: 手部关键点跟踪与处理 ---
        start_time = time.time()
        keypoint_2d_left_array, keypoint_2d_right_array, keypoint_3d_left_array, keypoint_3d_right_array,keypoint_2d_left,keypoint_2d_right = self.hand_track()
        # print(f"[耗时] D. 手部关键点跟踪与处理: {time.time() - start_time:.4f} 秒")
        
        
        if keypoint_3d_body_array is not None:
            
            # --- 关键时间点 E: 身体关键点滤波 (2D + 3D) ---
            start_time = time.time()
            
            keypoint_2d_body_array = self.filter_2d.next(keypoint_2d_body_array)
            keypoint_3d_body_array = self.filter_3d.next(keypoint_3d_body_array)
            # print(f"[耗时] E. 身体关键点滤波 (2D + 3D): {time.time() - start_time:.4f} 秒")
            self.is_straight_left = self.is_arm_straight(keypoint_3d_body_array, "left")
            self.is_straight_right = self.is_arm_straight(keypoint_3d_body_array, "right")
            
            
            # --- 关键时间点 F: 深度计算 ---
            start_time = time.time()
            # ## 计算头部和左右手深度
            head_depth_m, left_depth_m, right_depth_m = self.cal_left_right_head_depth(keypoint_2d_body_array, visib, depth)
            # print(f"[耗时] F. 深度计算: {time.time() - start_time:.4f} 秒")

            
             # --- 关键时间点 G: 计算相对位移 ---
            start_time = time.time()
            # ## 计算相对位移
            NOSE_IDX = MEDIAPIPE_POSE_KEYPOINTS.index('nose')
            LW_IDX   = MEDIAPIPE_POSE_KEYPOINTS.index('left_wrist')
            RW_IDX   = MEDIAPIPE_POSE_KEYPOINTS.index('right_wrist')
            
            left_loc = keypoint_3d_body_array[LW_IDX].copy()
            right_loc = keypoint_3d_body_array[RW_IDX].copy()
            head_loc = keypoint_3d_body_array[NOSE_IDX].copy()
            left_loc[-1] = left_depth_m
            right_loc[-1] = right_depth_m
            head_loc[-1] = head_depth_m
            left2head = np.round(left_loc - head_loc, 2)
            right2head = np.round(right_loc - head_loc, 2)

            # print(f"[耗时] G. 计算相对位移: {time.time() - start_time:.4f} 秒")
  
            self.last_left2hand_depth = left2head[-1]
            self.last_right2hand_depth = right2head[-1]

            cv2.putText(image_show, f"left2head: {left2head}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_show, f"right2head: {right2head}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


            # --- 关键时间点 H: 计算手部旋转矩阵 ---
            start_time = time.time()
            ## 计算手部旋转矩阵
            left_wrist_rot = calculate_hand_rot(keypoint_3d_left_array) if keypoint_3d_left_array is not None else self.last_left_wrist_rot
            right_wrist_rot = calculate_hand_rot(keypoint_3d_right_array) if keypoint_3d_right_array is not None else self.last_right_wrist_rot
            # print(f"[耗时] H. 计算手部旋转矩阵: {time.time() - start_time:.4f} 秒")

            # --- 关键时间点 I: 平滑手部旋转 ---
            start_time = time.time()
            ## 平滑手部旋转矩阵
            self.valid_orientation_left = rotation_matrix_to_quaternion_stable(left_wrist_rot)
            self.valid_orientation_right = rotation_matrix_to_quaternion_stable(right_wrist_rot)
            
            smoothed_rotation_left = self.left_hand_smoother.smooth_rotation(self.valid_orientation_left)
            smoothed_rotation_right = self.right_hand_smoother.smooth_rotation(self.valid_orientation_right)
            left_wrist_rot = quaternion.as_rotation_matrix(smoothed_rotation_left)
            right_wrist_rot = quaternion.as_rotation_matrix(smoothed_rotation_right)
            # print(f"[耗时] I. 平滑手部旋转: {time.time() - start_time:.4f} 秒")

            self.last_left_wrist_rot = left_wrist_rot  
            self.last_right_wrist_rot = right_wrist_rot
        
            # --- 关键时间点 J: 构建最终变换矩阵 ---
            start_time = time.time()
            left_hand_matrix = np.eye(4)
            left_hand_matrix[:3, :3] = left_wrist_rot
            left_hand_matrix[:3, 3] = left2head
            self.left_hand = left_hand_matrix
            
            right_hand_matrix = np.eye(4)
            right_hand_matrix[:3, :3] = right_wrist_rot
            right_hand_matrix[:3, 3] = right2head
            self.right_hand = right_hand_matrix
            # print(f"[耗时] J. 构建左右手最终变换矩阵: {time.time() - start_time:.4f} 秒")

            ## 更新共享内存中的关键点
            if keypoint_3d_left_array is not None:
                self.left_landmarks  = keypoint_3d_left_array
            if keypoint_3d_right_array is not None:
                self.right_landmarks  = keypoint_3d_right_array

            # --- 关键时间点 K: 在图像上绘制结果 ---
            start_time = time.time()
            image_show = self.draw_pose_on_image(image_show,keypoint_2d_body)
            image_show = self.draw_skeleton_on_image(image_show, keypoint_2d_left, keypoint_2d_right)
            self.draw_coordinate_system(image_show,keypoint_2d_left_array,left_wrist_rot)
            self.draw_coordinate_system(image_show,keypoint_2d_right_array,right_wrist_rot)
            # print(f"[耗时] K. 在图像上绘制结果: {time.time() - start_time:.4f} 秒")
            # cv2.imshow("realtime_retargeting_demo", image_show)
            
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     exit(0)
        return image_show
    

    def body_track(self):
        if self.detection_result_pose is None or not self.detection_result_pose.pose_landmarks:
            return None, None, None, None, None
        

        keypoint_2d_body = self.detection_result_pose.pose_landmarks.landmark
        keypoint_2d_body_array= np.array([[lm.x * self.im_width, lm.y * self.im_height] for lm in keypoint_2d_body])
        
        keypoint_3d_body = self.detection_result_pose.pose_world_landmarks.landmark
        keypoint_3d_body_array = np.array([[lm.x, lm.y, lm.z] for lm in keypoint_3d_body])
        visib = np.array([lm.visibility > 0.2 for lm in self.detection_result_pose.pose_landmarks.landmark])
    
        return keypoint_2d_body, keypoint_2d_body_array, keypoint_3d_body, keypoint_3d_body_array, visib

    def hand_track(self):
        keypoint_2d_left_array, keypoint_2d_right_array, keypoint_3d_left_array, keypoint_3d_right_array,keypoint_2d_left,keypoint_2d_right = None, None, None, None , None, None      
        
        if self.detection_result_hand is None or not self.detection_result_hand.multi_hand_landmarks:
            return keypoint_2d_left_array, keypoint_2d_right_array, keypoint_3d_left_array, keypoint_3d_right_array,keypoint_2d_left,keypoint_2d_right
        
        for i in range(len(self.detection_result_hand.multi_hand_landmarks)):
            label = self.detection_result_hand.multi_handedness[i].ListFields()[0][1][0].label
            if label == 'Right':
                keypoint_3d_left = self.detection_result_hand.multi_hand_world_landmarks[i]
                keypoint_2d_left = self.detection_result_hand.multi_hand_landmarks[i]
                keypoint_2d_left_array = parse_keypoint_2d(keypoint_2d_left, (self.im_height, self.im_width))
                keypoint_3d_left_array = parse_keypoint_3d(keypoint_3d_left)
            else:
                keypoint_3d_right = self.detection_result_hand.multi_hand_world_landmarks[i]
                keypoint_2d_right = self.detection_result_hand.multi_hand_landmarks[i]
                keypoint_2d_right_array = parse_keypoint_2d(keypoint_2d_right, (self.im_height, self.im_width))
                keypoint_3d_right_array = parse_keypoint_3d(keypoint_3d_right)

        # for i, handedness in enumerate(self.detection_result_hand.handedness):
        #     hand_landmarks = self.detection_result_hand.hand_landmarks[i]
        #     hand_world_landmarks = self.detection_result_hand.hand_world_landmarks[i]
            
        #     if handedness[0].category_name == 'Left':
        #         keypoint_2d_left_array = parse_keypoint_2d(hand_landmarks, (self.im_height, self.im_width))
        #         keypoint_3d_left_array = parse_keypoint_3d(hand_world_landmarks)
        #     elif handedness[0].category_name == 'Right':
        #         keypoint_2d_right_array = parse_keypoint_2d(hand_landmarks, (self.im_height, self.im_width))
        #         keypoint_3d_right_array = parse_keypoint_3d(hand_world_landmarks)
            
        return keypoint_2d_left_array, keypoint_2d_right_array, keypoint_3d_left_array, keypoint_3d_right_array,keypoint_2d_left,keypoint_2d_right

  
  
    def cal_left_right_head_depth(self, keypoint_2d_body_array, visib, depth):
        HEAD_IDXS = [0, 9, 10]                 # nose, mouth_left, mouth_right
        LEFT_HAND_IDXS = [15, 17, 19, 21]      # 左手: wrist/pinky/index/thumb
        RIGHT_HAND_IDXS = [16, 18, 20, 22]     # 右手: wrist/pinky/index/thumb
        
        # 为不同部位使用不同的窗口大小
        head_patch = 15  # 头部使用适中窗口
        hand_patch = 12  # 手部使用稍小窗口，更精确捕捉运动
        
        # 获取当前时间戳
        current_time = time.time()
        
        # 获取原始深度测量
        head_uv = center_uv(keypoint_2d_body_array, visib, HEAD_IDXS)
        head_depth_m = robust_depth_m(depth, self.depth_scale, *head_uv, patch=head_patch) if head_uv is not None else None
        
        left_uv = center_uv(keypoint_2d_body_array, visib, LEFT_HAND_IDXS)
        left_depth_m = robust_depth_m(depth, self.depth_scale, *left_uv, patch=hand_patch) if left_uv is not None else None
        
        right_uv = center_uv(keypoint_2d_body_array, visib, RIGHT_HAND_IDXS)
        right_depth_m = robust_depth_m(depth, self.depth_scale, *right_uv, patch=hand_patch) if right_uv is not None else None
        
        # 检测手部运动状态
        # 计算左右手的速度估计
        if hasattr(self.kalman_left, 'v') and hasattr(self.kalman_right, 'v'):
            left_velocity = abs(self.kalman_left.v)
            right_velocity = abs(self.kalman_right.v)
            
            # 更新运动状态标志
            self.is_left_moving = left_velocity > self.movement_threshold
            self.is_right_moving = right_velocity > self.movement_threshold
            
            # 保存当前速度
            self.last_left_velocity = left_velocity
            self.last_right_velocity = right_velocity
        
        # 动态调整滤波器参数：手部运动时减少平滑，提高响应速度
        if self.is_left_moving:
            self.kalman_left.R = 0.05  # 运动时增加测量信任度
        else:
            self.kalman_left.R = 0.08  # 静止时增加平滑效果
        
        if self.is_right_moving:
            self.kalman_right.R = 0.05
        else:
            self.kalman_right.R = 0.08
        
        # 使用卡尔曼滤波器进行平滑
        self.last_head_depth = self.kalman_head.update(head_depth_m, current_time)
        self.last_left_depth = self.kalman_left.update(left_depth_m, current_time)
        self.last_right_depth = self.kalman_right.update(right_depth_m, current_time)
        
        # 应用深度一致性约束（手和头的深度差异不超过0.6米）
        if self.last_head_depth > 0:
            # 手的深度应该在头部深度的合理范围内
            head_range_min = max(0.2, self.last_head_depth - 0.6)
            head_range_max = self.last_head_depth + 0.6
            
            # 修复左手深度
            if self.last_left_depth > 0:
                if not (head_range_min <= self.last_left_depth <= head_range_max):
                    # 计算与合理范围的差值
                    distance_to_min = abs(self.last_left_depth - head_range_min)
                    distance_to_max = abs(self.last_left_depth - head_range_max)
                    
                    # 选择更近的边界作为参考点
                    if distance_to_min < distance_to_max:
                        reference_depth = head_range_min
                    else:
                        reference_depth = head_range_max
                    
                    # 根据运动状态调整修复策略
                    if self.is_left_moving:
                        # 运动时，保留更多当前测量值，但限制在合理范围内
                        if left_depth_m is not None:
                            # 先将测量值限制在合理范围内
                            clamped_meas = max(head_range_min, min(head_range_max, left_depth_m))
                            # 加权平均，更信任当前测量
                            self.last_left_depth = 0.8 * clamped_meas + 0.2 * self.last_left_depth
                    else:
                        # 静止时，更严格地应用一致性约束
                        # 平滑地将深度拉回合理范围
                        self.last_left_depth = 0.6 * reference_depth + 0.4 * self.last_left_depth
            
            # 修复右手深度
            if self.last_right_depth > 0:
                if not (head_range_min <= self.last_right_depth <= head_range_max):
                    # 计算与合理范围的差值
                    distance_to_min = abs(self.last_right_depth - head_range_min)
                    distance_to_max = abs(self.last_right_depth - head_range_max)
                    
                    # 选择更近的边界作为参考点
                    if distance_to_min < distance_to_max:
                        reference_depth = head_range_min
                    else:
                        reference_depth = head_range_max
                    
                    # 根据运动状态调整修复策略
                    if self.is_right_moving:
                        # 运动时，保留更多当前测量值，但限制在合理范围内
                        if right_depth_m is not None:
                            # 先将测量值限制在合理范围内
                            clamped_meas = max(head_range_min, min(head_range_max, right_depth_m))
                            # 加权平均，更信任当前测量
                            self.last_right_depth = 0.8 * clamped_meas + 0.2 * self.last_right_depth
                    else:
                        # 静止时，更严格地应用一致性约束
                        # 平滑地将深度拉回合理范围
                        self.last_right_depth = 0.6 * reference_depth + 0.4 * self.last_right_depth
        
        # 额外的合理性检查：深度不应为负
        self.last_head_depth = max(0.1, self.last_head_depth)
        self.last_left_depth = max(0.1, self.last_left_depth)
        self.last_right_depth = max(0.1, self.last_right_depth)
        
        return self.last_head_depth, self.last_left_depth, self.last_right_depth


    def draw_coordinate_system(self, image, keypoint_2d, wrist_rot, scale=50):
        if keypoint_2d is None or wrist_rot is None:
            return image
        
        wrist_x = int(keypoint_2d[0][0])
        wrist_y = int(keypoint_2d[0][1])
        
        colors = {'x': (0, 0, 255), 'y': (0, 255, 0), 'z': (255, 0, 0)}
        
        for i, axis in enumerate(['x', 'y', 'z']):
            axis_vector = wrist_rot[:, i]
            endpoint_x = int(wrist_x + axis_vector[0] * scale)
            endpoint_y = int(wrist_y + axis_vector[1] * scale) # Y轴在图像坐标中是向下的
            cv2.line(image, (wrist_x, wrist_y), (endpoint_x, endpoint_y), colors[axis], 2)
            cv2.putText(image, axis, (endpoint_x, endpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[axis], 2)
        return image
    

    def draw_pose_on_image(
        self,image, pose_landmarks):
        
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            self.detection_result_pose.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
        )

        return image


    def draw_skeleton_on_image(self,image, keypoint_2d_l,keypoint_2d_r: landmark_pb2.NormalizedLandmarkList):
       
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            keypoint_2d_l,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            keypoint_2d_r,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )
      

        return image

    def recognize_gesture(self, landmarks):
  
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # 计算手掌基准大小（手腕到中指根部的距离）
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        palm_size = self.calculate_distance(wrist, middle_mcp)
        
        # 获取各指尖
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # 计算指尖与拇指尖的相对距离（相对于手掌大小）
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip) / palm_size
        thumb_index_dist_new = self.calculate_distance_new(thumb_tip, index_tip) / palm_size
        thumb_middle_dist = self.calculate_distance(thumb_tip, middle_tip) / palm_size
        thumb_ring_dist = self.calculate_distance(thumb_tip, ring_tip) / palm_size
        thumb_pinky_dist = self.calculate_distance(thumb_tip, pinky_tip) / palm_size
        
        # 计算各指尖是否弯曲（与对应MCP关节的距离）
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = landmarks[self.mp_hands.HandLandmark.PINKY_MCP]
        
        index_extended = self.calculate_distance(index_tip, wrist) > self.calculate_distance(index_mcp, wrist)
        middle_extended = self.calculate_distance(middle_tip, wrist) > self.calculate_distance(middle_mcp, wrist)
        ring_extended = self.calculate_distance(ring_tip, wrist) > self.calculate_distance(ring_mcp, wrist)
        pinky_extended = self.calculate_distance(pinky_tip, wrist) > self.calculate_distance(pinky_mcp, wrist)
        thumb_extended = self.calculate_distance(thumb_tip, wrist) > self.calculate_distance(
            landmarks[self.mp_hands.HandLandmark.THUMB_IP], wrist)
        
        # self.gripper_cmd = (self.clamp(abs(thumb_index_dist_new), 0.7, 1.4)-0.7)/10
        # if(self.gripper_cmd>=0.035):
        #     self.gripper_cmd = 0.07
        # else:
        #     self.gripper_cmd = 0
        # print(self.gripper_cmd)
       
        # 判断手势
        ## 暂时先使用固定的夹爪打开和关闭，后续更新捏合到夹爪的映射
        if thumb_index_dist < 0.3 and all([middle_extended, ring_extended, pinky_extended]):
            return "close"  # 拇指+食指捏合，其他不弯曲
        elif thumb_ring_dist < 0.3 and all([index_extended,  middle_extended,  pinky_extended]):
            return "end"  # 拇指+无名指捏合，其他不弯曲
        elif thumb_pinky_dist < 0.3 and all([ index_extended, middle_extended, ring_extended]):
            return "start"  # 拇指+小指捏合，其他不弯曲
        else:
            return "open"
        
    def calculate_distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
    
    def calculate_distance_new(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def is_arm_straight(self,keypoints_3d, side="right"):
        """
        判断指定的手臂是否伸直。

        :param keypoints_3d: MediaPipe Pose 输出的 3D 关键点数组。
        :param side: "right" 或 "left"，指定要判断的手臂。
        :return: 如果手臂伸直，则返回 True，否则返回 False。
        """
        if side == "right":
            shoulder_idx = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
            elbow_idx = self.mp_pose.PoseLandmark.RIGHT_ELBOW
            wrist_idx = self.mp_pose.PoseLandmark.RIGHT_WRIST
        elif side == "left":
            shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER
            elbow_idx = self.mp_pose.PoseLandmark.LEFT_ELBOW
            wrist_idx = self.mp_pose.PoseLandmark.LEFT_WRIST
        else:
            print(f"Error: Unknown side '{side}'. Use 'right' or 'left'.")
            return False

        # 提取关键点坐标 (x, y, z)
        shoulder = np.array([
            keypoints_3d[shoulder_idx][0],
            keypoints_3d[shoulder_idx][1],
            keypoints_3d[shoulder_idx][2]
        ])
        elbow = np.array([
            keypoints_3d[elbow_idx][0],
            keypoints_3d[elbow_idx][1],
            keypoints_3d[elbow_idx][2]
        ])
        wrist = np.array([
            keypoints_3d[wrist_idx][0],
            keypoints_3d[wrist_idx][1],
            keypoints_3d[wrist_idx][2]
        ])

        # 计算向量
        vec1 = elbow - shoulder
        vec2 = wrist - elbow

        # 计算点积
        dot_product = np.dot(vec1, vec2)

        # 计算向量的 L2 范数（长度）
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # 防止除以零
        if norm_vec1 == 0 or norm_vec2 == 0:
            # 如果关键点检测失败（坐标为0），则认为不是直线
            return False

        # 计算夹角的余弦值
        cos_theta = dot_product / (norm_vec1 * norm_vec2)

        # 设置一个阈值来判断是否为直线。1.0 表示完美直线。
        # 考虑到检测误差，0.95 是一个比较合理的阈值。
        # 你可以根据实际情况调整这个值。
        
       
        # 如果余弦值大于阈值，则认为手臂是伸直的
        return cos_theta > self.straight_threshold

if __name__ == '__main__':
    INPUT_IMAGE_SIZE = (640, 480)
    tv_wrapper = TeleVisionWrapper(INPUT_IMAGE_SIZE)
    while True:
        tv_wrapper.get_data()
        time.sleep(0.03)