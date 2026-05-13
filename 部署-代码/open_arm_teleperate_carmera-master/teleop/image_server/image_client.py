import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory
import pyrealsense2 as rs
class ImageClient:
    def __init__(self, tv_img_shape = None, tv_img_shm_name = None, tv_depth_shm_name = None, image_show = False, Unit_Test=False):
        

        ## 初始化相机
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, tv_img_shape[1], tv_img_shape[0], rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, tv_img_shape[1], tv_img_shape[0], rs.format.z16, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        self.device = self.profile.get_device()
        self.depth_sensor = self.device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        


        self.running = True
        self._image_show = image_show
        self._enable_performance_eval = Unit_Test  # 使用Unit_Test参数来启用性能评估
       
        self.tv_img_shape = tv_img_shape
        
        # 初始化性能评估参数
        if self._enable_performance_eval:
            self._init_performance_metrics()
        
        # 初始化共享内存
        self.tv_enable_shm = tv_img_shm_name is not None
        self.tv_enable_depth_shm = tv_depth_shm_name is not None
        
        if self.tv_enable_shm:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=self.tv_image_shm.buf)
        
        if self.tv_enable_depth_shm:
            tv_depth_shape = (tv_img_shape[0], tv_img_shape[1])
            self.tv_depth_shm = shared_memory.SharedMemory(name=tv_depth_shm_name)
            self.tv_depth_array = np.ndarray(tv_depth_shape, dtype=np.uint16, buffer=self.tv_depth_shm.buf)
        

    def _init_performance_metrics(self):
        self._frame_count = 0  # Total frames received
        self._last_frame_id = -1  # Last received frame ID

        # Real-time FPS calculation using a time window
        self._time_window = 1.0  # Time window size (in seconds)
        self._frame_times = deque()  # Timestamps of frames received within the time window

        # Data transmission quality metrics
        self._latencies = deque()  # Latencies of frames within the time window
        self._lost_frames = 0  # Total lost frames
        self._total_frames = 0  # Expected total frames based on frame IDs

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        # Update latency
        latency = receive_time - timestamp
        self._latencies.append(latency)

        # Remove latencies outside the time window
        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()

        # Update frame times
        self._frame_times.append(receive_time)
        # Remove timestamps outside the time window
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()

        # Update frame counts for lost frame calculation
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                print(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                print(f"[Image Client] Detected lost frames: {lost}, Expected frame ID: {expected_frame_id}, Received frame ID: {frame_id}")
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1

        self._frame_count += 1

    def _print_performance_metrics(self, receive_time):
        if self._frame_count % 30 == 0:
            # Calculate real-time FPS
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0

            # Calculate latency metrics
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0

            # Calculate lost frame rate
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            print(f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Avg Latency: {avg_latency*1000:.2f} ms, Max Latency: {max_latency*1000:.2f} ms, \
                  Min Latency: {min_latency*1000:.2f} ms, Jitter: {jitter*1000:.2f} ms, Lost Frame Rate: {lost_frame_rate:.2f}%")
    
    def _close(self):
        try:
            self.pipeline.stop()
            if self.tv_enable_shm:
                self.tv_image_shm.close()
            if self.tv_enable_depth_shm:
                self.tv_depth_shm.close()
            if self._image_show:
                cv2.destroyAllWindows()
            print("Image client has been closed properly.")
        except Exception as e:
            print(f"Error during closing: {e}")

    
    def receive_process(self):
        

        print("\nImage client has started, waiting to receive data...")
        try:
            frame_id = 0
            while self.running:
                receive_time = time.time()
                # 获取realsense相机的帧
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                

                if self.tv_enable_shm:
                    np.copyto(self.tv_img_array, color_image)
                    
                if self.tv_enable_depth_shm:
                    np.copyto(self.tv_depth_array, depth_image)
                
                if self._image_show:
                
                    cv2.imshow('Image Client Stream', color_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

                if self._enable_performance_eval:
                    self._update_performance_metrics(time.time(), frame_id, receive_time)
                    self._print_performance_metrics(receive_time)
                frame_id += 1

        except KeyboardInterrupt:
            print("Image client interrupted by user.")
        except Exception as e:
            print(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()

if __name__ == "__main__":
        # 创建共享内存（主进程中执行）
        tv_img_shape = (480, 640, 3)
        depth_shape = (480, 640)
        
        img_size = int(np.prod(tv_img_shape) * np.uint8().itemsize)

        # 彩色图像共享内存
        img_shm = shared_memory.SharedMemory(
            create=True, 
            size=img_size
        )
        tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = img_shm.buf)
        depth_size = int(np.prod(depth_shape) * np.uint16().itemsize)
        # 深度图像共享内存
        depth_shm = shared_memory.SharedMemory(
            create=True,
            size=depth_size
        )
       
        try:
            # 启动客户端
            client = ImageClient(
                tv_img_shape=tv_img_shape,
                tv_img_shm_name=img_shm.name,
                tv_depth_shm_name=depth_shm.name,
                image_show=True,
                Unit_Test=False
            )
            client.receive_process()
        
        finally:
            # 释放共享内存（主进程中执行）
            img_shm.close()
            img_shm.unlink()
            depth_shm.close()
            depth_shm.unlink()