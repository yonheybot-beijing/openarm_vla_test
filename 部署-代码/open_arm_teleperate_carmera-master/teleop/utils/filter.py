#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.position_window = deque(maxlen=window_size)
        self.orientation_window = deque(maxlen=window_size)
        
    def filter(self, pose_msg):
        # 提取位置和方向
        position = [pose_msg.pose.position.x, 
                   pose_msg.pose.position.y, 
                   pose_msg.pose.position.z]
        orientation = [pose_msg.pose.orientation.x,
                      pose_msg.pose.orientation.y,
                      pose_msg.pose.orientation.z,
                      pose_msg.pose.orientation.w]
        
        # 添加到窗口
        self.position_window.append(position) 
        self.orientation_window.append(orientation)
        
        # 计算平均值
        avg_position = [sum(x)/len(self.position_window) for x in zip(*self.position_window)]
        avg_orientation = [sum(x)/len(self.orientation_window) for x in zip(*self.orientation_window)]
        
        # 创建新的PoseStamped消息
        filtered_pose = PoseStamped()
        filtered_pose.header = pose_msg.header
        filtered_pose.pose.position.x = avg_position[0]
        filtered_pose.pose.position.y = avg_position[1]
        filtered_pose.pose.position.z = avg_position[2]
        filtered_pose.pose.orientation.x = avg_orientation[0]
        filtered_pose.pose.orientation.y = avg_orientation[1]
        filtered_pose.pose.orientation.z = avg_orientation[2]
        filtered_pose.pose.orientation.w = avg_orientation[3]
        
        return filtered_pose

class PoseKalmanFilter:
    def __init__(self):
        # 创建卡尔曼滤波器 (状态: x, y, z, vx, vy, vz, qx, qy, qz, qw)
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        
        # 初始化状态转移矩阵 (简单模型)
        self.kf.F = np.eye(10)
        dt = 0.1  # 时间间隔
        # 位置和速度关系
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt
        
        # 测量矩阵
        self.kf.H = np.zeros((7, 10))
        np.fill_diagonal(self.kf.H[:7, :7], 1)
        
        # 协方差矩阵
        self.kf.P *= 1000
        self.kf.R = np.eye(7) * 5  # 测量噪声
        self.kf.Q = np.eye(10) * 0.1  # 过程噪声
        
        self.last_time = None
        
    def filter(self, pose_msg):
        current_time = rospy.Time.now().to_sec()
        
        # 计算时间间隔
        if self.last_time is None:
            dt = 0.1
        else:
            dt = current_time - self.last_time
        self.last_time = current_time
        
        # 更新状态转移矩阵中的时间参数
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt
        
        # 预测
        self.kf.predict()
        
        # 更新测量值
        z = np.array([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        ])
        
        self.kf.update(z)
        
        # 创建过滤后的消息
        filtered_pose = PoseStamped()
        filtered_pose.header = pose_msg.header
        filtered_pose.pose.position.x = self.kf.x[0]
        filtered_pose.pose.position.y = self.kf.x[1]
        filtered_pose.pose.position.z = self.kf.x[2]
        filtered_pose.pose.orientation.x = self.kf.x[3]
        filtered_pose.pose.orientation.y = self.kf.x[4]
        filtered_pose.pose.orientation.z = self.kf.x[5]
        filtered_pose.pose.orientation.w = self.kf.x[6]
        
        return filtered_pose

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False