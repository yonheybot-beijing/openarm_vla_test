import numpy as np
import math
import quaternion

class SmoothRotator:
    def __init__(self, alpha=0.3):
        self.smoothed_rotation = None
        self.rotation_history = []
        self.history_size = 5  # 历史数据队列大小
        self.smoothing_factor = alpha # 平滑系数 (0-1)，值越小越平滑
    def smooth_rotation(self, new_rotation):
        """平滑旋转四元数 - 改进版本"""
        if self.smoothed_rotation is None:
            self.smoothed_rotation = new_rotation
            self.rotation_history = [new_rotation] * self.history_size  # 用当前值初始化历史
            return new_rotation
        
        # 检查四元数是否有效
        if not self.is_valid_quaternion(new_rotation):
            return self.smoothed_rotation
        
        # 添加到历史队列
        self.rotation_history.append(new_rotation)
        if len(self.rotation_history) > self.history_size:
            self.rotation_history.pop(0)
        
        # 使用加权滑动平均和球面线性插值
        if len(self.rotation_history) >= 2:
            # 方法1：加权平均（更重视最近的数据）
            smoothed_quat = self.weighted_quaternion_average(self.rotation_history)
            
            # 方法2：与上一帧进行SLERP插值（更平滑的过渡）
            
            current_q = np.array([self.smoothed_rotation.w, self.smoothed_rotation.x, 
                          self.smoothed_rotation.y, self.smoothed_rotation.z])

            target_q = np.array([smoothed_quat[0], smoothed_quat[1], 
                                smoothed_quat[2], smoothed_quat[3]])
            
            # 使用SLERP进行平滑插值
            final_quat = self.slerp(current_q, target_q, self.smoothing_factor)
            
            self.smoothed_rotation = np.quaternion(
                final_quat[0], final_quat[1], final_quat[2], final_quat[3]
            )
        
        return self.smoothed_rotation

    def is_valid_quaternion(self, quat):
        """检查四元数是否有效"""
        magnitude_sq = (quat.x ** 2 + quat.y ** 2 + quat.z ** 2 + quat.w ** 2)
        return 0.9 < magnitude_sq < 1.1  # 允许一定的数值误差
    def weighted_quaternion_average(self, quaternions):
        """加权四元数平均，更重视最近的数据"""
        n = len(quaternions)
        weights = np.linspace(0.1, 1.0, n)  # 线性权重，最近的最大
        weights = weights / np.sum(weights)  # 归一化权重
        
        # 转换为numpy数组
        quats_array = np.array([[q.w, q.x, q.y, q.z] for q in quaternions])
        
        # 使用特征值方法计算加权平均（更稳定）
        Q = np.zeros((4, 4))
        for i, q in enumerate(quats_array):
            # 确保四元数单位化
            q_normalized = q / np.linalg.norm(q)
            # 构建外积矩阵并加权
            outer_product = np.outer(q_normalized, q_normalized)
            Q += weights[i] * outer_product
        
        # 计算最大特征值对应的特征向量（即平均四元数）
        eigenvalues, eigenvectors = np.linalg.eig(Q)
        max_eigenvalue_index = np.argmax(eigenvalues)
        avg_quat = eigenvectors[:, max_eigenvalue_index]
        
        # 确保四元数在正确的半球
        if avg_quat[0] < 0:
            avg_quat = -avg_quat
        
        return avg_quat / np.linalg.norm(avg_quat)
    def slerp(self, q1, q2, t):
        """球面线性插值"""
        # 归一化四元数
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # 计算点积来确定插值方向
        dot = np.dot(q1, q2)
        
        # 如果点积为负，反转一个四元数以取最短路径
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # 如果四元数非常接近，使用线性插值避免数值问题
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # 计算插值角度
        theta_0 = np.arccos(dot)  # 角度
        theta = theta_0 * t       # 插值角度
        
        # 计算插值四元数
        q3 = q2 - q1 * dot
        q3 = q3 / np.linalg.norm(q3)
        
        return q1 * np.cos(theta) + q3 * np.sin(theta)
    def exponential_moving_average_quaternion(self, current, new, alpha):
        """四元数指数移动平均"""
        # 确保四元数单位化
        current = current / np.linalg.norm(current)
        new = new / np.linalg.norm(new)
        
        # 计算点积来确定插值方向
        dot = np.dot(current, new)
        
        # 如果点积为负，反转新四元数
        if dot < 0.0:
            new = -new
        
        # 使用SLERP进行指数移动平均
        return self.slerp(current, new, alpha)

def rotation_matrix_to_quaternion_stable(R):
        """稳定的旋转矩阵到四元数转换"""
        # 确保矩阵是正交的
        U, S, Vt = np.linalg.svd(R)
        R = np.dot(U, Vt)
        
        # 使用更稳定的转换方法
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            S = math.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        # 创建四元数并确保单位化
        np_quat = np.quaternion(w, x, y, z)
        np_quat = np_quat.normalized()
        # 确保w分量为正（标准形式）
        if np_quat.components[0] < 0:
            np_quat = -np_quat
        
        return np_quat