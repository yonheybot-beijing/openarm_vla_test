import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark



    # def parse_keypoint_2d(
    #         keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size
    #     ) -> np.ndarray:
    #         keypoint = np.empty([21, 2])
    #         for i in range(21):
    #             keypoint[i][0] = keypoint_2d[i].x
    #             keypoint[i][1] = keypoint_2d[i].y
    #         keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
    #         return keypoint

    # def parse_keypoint_3d(
    #         keypoint_3d: framework.formats.landmark_pb2.LandmarkList,
    #     ) -> np.ndarray:
    #         keypoint = np.empty([21, 3])
    #         for i in range(21):
    #             keypoint[i][0] = keypoint_3d[i].x
    #             keypoint[i][1] = keypoint_3d[i].y
    #             keypoint[i][2] = keypoint_3d[i].z
    #         return keypoint


def parse_keypoint_3d(
    keypoint_3d: framework.formats.landmark_pb2.LandmarkList,
) -> np.ndarray:
    keypoint = np.empty([21, 3])
    for i in range(21):
        keypoint[i][0] = keypoint_3d.landmark[i].x
        keypoint[i][1] = keypoint_3d.landmark[i].y
        keypoint[i][2] = keypoint_3d.landmark[i].z
    return keypoint


def parse_keypoint_2d(
    keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size
) -> np.ndarray:
    keypoint = np.empty([21, 2])
    for i in range(21):
        keypoint[i][0] = keypoint_2d.landmark[i].x
        keypoint[i][1] = keypoint_2d.landmark[i].y
    keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
    return keypoint

def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

def calculate_hand_rot(keypoint_3d_array):
    ## 初始版本
    # Parse 3d keypoint from MediaPipe hand detector
    keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
    mediapipe_wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)

    return mediapipe_wrist_rot



# def robust_depth_m(depth_image, depth_scale,u, v, patch=10):
#     h, w = depth_image.shape
#     x0 = max(0, int(u - patch // 2)); x1 = min(w, int(u + patch // 2))
#     y0 = max(0, int(v - patch // 2)); y1 = min(h, int(v + patch // 2))
#     roi = depth_image[y0:y1, x0:x1]
#     roi = roi[roi > 0]
    
#     if roi.size == 0:
#         return None
#     return float(np.median(roi)) * depth_scale

def robust_depth_m(depth_image, depth_scale, u, v, patch=15, min_valid_ratio=0.3, max_depth=5.0):
    h, w = depth_image.shape
    # 扩大搜索窗口，增加获取有效深度的概率
    x0 = max(0, int(u - patch // 2)); x1 = min(w, int(u + patch // 2))
    y0 = max(0, int(v - patch // 2)); y1 = min(h, int(v + patch // 2))
    

    if x0 >= x1 or y0 >= y1:
        return None
    roi = depth_image[y0:y1, x0:x1]
    valid_depth = roi[(roi > 0) & (roi * depth_scale < max_depth)]  # 过滤无效值和过大深度值
    
    if valid_depth.size == 0:
        return None
    
    # 计算有效像素比例
    valid_ratio = valid_depth.size / (roi.size)
    
    # 如果有效像素太少，说明可能是噪声区域
    if valid_ratio < min_valid_ratio:
        return None
    
    # 使用中位数和标准差来进一步过滤异常值
    median_depth = np.median(valid_depth)
    std_depth = np.std(valid_depth)
    
    # 只保留在中位数±2倍标准差范围内的值
    filtered_depth = valid_depth[np.abs(valid_depth - median_depth) < 2 * std_depth]
    
    if filtered_depth.size == 0:
        return float(median_depth) * depth_scale
    
    # 使用加权平均，中心像素权重更高
    # 创建距离中心的权重图
    center_x, center_y = (x1 - x0) // 2, (y1 - y0) // 2
    weights = np.zeros((y1 - y0, x1 - x0))
    for i in range(y1 - y0):
        for j in range(x1 - x0):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            weights[i, j] = np.exp(-dist**2 / (2 * (patch/4)**2))  # 高斯权重
    
    # 应用权重到过滤后的深度值
    valid_mask = (roi > 0) & (roi * depth_scale < max_depth) & (np.abs(roi - median_depth) < 2 * std_depth)
    if np.sum(valid_mask) > 0:
        weighted_depth = np.sum(roi[valid_mask] * weights[valid_mask]) / np.sum(weights[valid_mask])
        return float(weighted_depth) * depth_scale
    else:
        return float(median_depth) * depth_scale



def center_uv(pose_kpts2d,visib,idxs):
    if pose_kpts2d is None:
        return None
    
    pts = [(pose_kpts2d[i][0], pose_kpts2d[i][1]) for i in idxs if visib[i]]
    if not pts:
        return None
    xs, ys = zip(*pts)
    return float(np.mean(xs)), float(np.mean(ys))


## 窗口平均平滑
class LandmarkSmoother:
    def __init__(self):
        self.frame_sets = []
        self.frame_size = 8
        self.smooth_frame = [[] for _ in range(33)]
        self.is_3d = False
    def smooth_landmarks(self,results):
            """
            平滑Mediapipe提供的landmarks数据
            :param results: Mediapipe的结果对象
            :param on_results: 可选回调函数
            :return: 平滑后的结果对象
            """
            # 将当前帧的landmarks添加到frame_sets
            
            self.frame_sets.append(results)
            if len(results[0])==3:
                self.is_3d = True
            else:
                self.is_3d = False

            if len(self.frame_sets) < self.frame_size:
                return results

            if len(self.frame_sets) > self.frame_size:
                self.frame_sets.pop(0)  

            # 如果收集到8帧，开始处理
            if len(self.frame_sets) == self.frame_size:
                # 遍历每个关节（33个）
                for i in range(33):
                    # 提取每帧中当前关节的x, y, z, visibility
                    
                    x = [frame[i][0] for frame in self.frame_sets]
                    y = [frame[i][1] for frame in self.frame_sets]
                    
                    # visibility = [frame[i]['visibility'] for frame in self.frame_sets]

                    # 对每个坐标排序
                    x.sort()
                    y.sort()
                    
                    # visibility.sort()

                    # 丢弃最大值和最小值的2个
                    x = x[2:6]
                    y = y[2:6]
                    
                    # visibility = visibility[2:6]
                   
                    if self.is_3d:
                        z = [frame[i][2] for frame in self.frame_sets]
                        z.sort()
                        z = z[2:6]
                    
                        self.smooth_frame[i] = [
                            sum(x) / len(x),
                            sum(y) / len(y),
                            sum(z) / len(z)
                        ]
                    else:
                        self.smooth_frame[i] = [
                            sum(x) / len(x),
                            sum(y) / len(y)
                        ]
            
   
            return np.array(self.smooth_frame)

## 卡尔曼滤波平滑深度
class KalmanFilterDepth:
    def __init__(self, process_noise=0.01, measurement_noise=0.1, error_estimate=0.1, initial_depth=0.0, acceleration_noise=0.001):
        # 初始化卡尔曼滤波器参数
        self.Q = process_noise  # 过程噪声协方差
        self.R = measurement_noise  # 测量噪声协方差
        self.P = error_estimate  # 估计误差协方差
        self.K = 0  # 卡尔曼增益
        self.x = initial_depth  # 初始状态估计（位置）
        self.v = 0  # 初始速度估计
        self.A = np.array([[1, 1], [0, 1]])  # 状态转移矩阵（考虑速度）
        self.B = np.array([[0.5], [1]])      # 控制输入矩阵
        self.C = np.array([1, 0])            # 测量矩阵
        self.Q_matrix = np.array([[self.Q, 0], [0, acceleration_noise]])  # 扩展的过程噪声矩阵
        self.is_initialized = False if initial_depth == 0.0 else True
        self.last_time = None
        
    def update(self, measurement, timestamp=None):
        # 如果未初始化且测量有效，初始化
        if not self.is_initialized and measurement is not None:
            self.x = measurement
            self.is_initialized = True
            self.last_time = timestamp if timestamp is not None else time.time()
            return self.x
        
        # 如果没有测量值，返回当前估计
        if measurement is None:
            return self.x
        
        # 计算时间间隔
        current_time = timestamp if timestamp is not None else time.time()
        if self.last_time is not None:
            dt = current_time - self.last_time
            # 更新状态转移矩阵
            self.A = np.array([[1, dt], [0, 1]])
            self.B = np.array([[0.5 * dt * dt], [dt]])
        else:
            dt = 0.033  # 假设30fps
        self.last_time = current_time
        
        # 预测步骤
        # 状态预测：x = Ax + Bu
        state = np.array([self.x, self.v])
        state_pred = self.A @ state
        
        # 误差协方差预测：P = APA' + Q
        P_pred = self.A @ np.array([[self.P, 0], [0, self.P]]) @ self.A.T + self.Q_matrix
        
        # 更新步骤
        # 计算卡尔曼增益：K = P C' / (C P C' + R)
        denominator = self.C @ P_pred @ self.C.T + self.R
        if denominator == 0:
            K = 0
        else:
            K = P_pred @ self.C.T / denominator
        
        # 更新状态估计：x = x + K (measurement - Cx)
        innovation = measurement - self.C @ state_pred
        state_updated = state_pred + K * innovation
        
        # 更新误差协方差：P = (I - K C) P
        I = np.eye(2)
        P_updated = (I - np.outer(K, self.C)) @ P_pred
        
        # 更新状态变量
        self.x = state_updated[0]
        self.v = state_updated[1]
        self.P = P_updated[0, 0]  # 只保留位置的误差协方差
        
        return self.x


