import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.open_television.tv_wrapper_camera import TeleVisionWrapper
from teleop.robot_control.robot_arm import Open_arm_ArmController
# from teleop.robot_control.robot_hand_openarm import Gripper_Controller
from teleop.robot_control.robot_arm_ik import Open_ArmIK

from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.hand_gesture import staticGestureRec


# 全局变量（用于线程间通信）
running = False
image_show = None
image_lock = threading.Lock()  # 保护 image_show 的线程安全
##夹爪状态确认控制
GESTURE_CONFIRM_TIME = 0.5 
# 为每个手的每个动作维护一个计时器和状态
# 使用字典来存储，键是 "left" 或 "right"，值是另一个字典
gesture_timers = {
    "left": {"state": None, "start_time": None},
    "right": {"state": None, "start_time": None}
}

def confirm_gesture(hand, new_state):
    """
    检查手势是否已经稳定保持了一段时间。
    :param hand: "left" 或 "right"
    :param new_state: 当前检测到的手势状态 ("open", "close", 或其他)
    :return: 如果状态稳定，则返回确认的状态，否则返回 None
    """
    global gesture_timers
    
    # 如果检测到的是无效状态（如 "UNKNOWN"），则重置计时器
    if new_state not in ["open", "close"]:
        gesture_timers[hand]["state"] = None
        gesture_timers[hand]["start_time"] = None
        return None

    current_time = time.time()
    
    # 如果状态没有变化
    if gesture_timers[hand]["state"] == new_state:
        # 检查计时器是否已经启动
        if gesture_timers[hand]["start_time"] is not None:
            # 计算已经保持的时间
            elapsed_time = current_time - gesture_timers[hand]["start_time"]
            # 如果超过确认时间，则返回该状态
            if elapsed_time >= GESTURE_CONFIRM_TIME:
                return new_state
        # 如果计时器未启动（可能是第一次检测到），则启动它
        else:
            gesture_timers[hand]["start_time"] = current_time
    # 如果状态发生了变化
    else:
        # 更新状态并重置计时器
        gesture_timers[hand]["state"] = new_state
        gesture_timers[hand]["start_time"] = current_time
        
    # 如果还未确认，则返回 None
    return None

def visualization_thread():
    """专门用于显示图像的线程"""
    global running, image_show
    cv2.namedWindow("record image", cv2.WINDOW_NORMAL)
    
    while running:
        with image_lock:
            if image_show is not None:
                cv2.imshow("record image", image_show)
        
        # 监听键盘事件（必须在显示线程中调用 waitKey）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False  # 收到退出信号
            break
    
    cv2.destroyAllWindows()  # 关闭窗口

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type = int, default = 20.0, help = 'save data\'s frequency')
    parser.add_argument('--gesture_confirm_time', type=float, default=0.5, help='Time in seconds to confirm a gesture.')
    parser.set_defaults(record = False)
    args = parser.parse_args()
    print(f"args:{args}\n")

    # --- 使用命令行参数更新全局变量 ---
    GESTURE_CONFIRM_TIME = args.gesture_confirm_time


    INPUT_IMAGE_SIZE = (640, 480)
    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    

    # arm_ctrl = Open_arm_ArmController(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
    arm_ctrl = Open_arm_ArmController(None, None, None, None, None,True)
   

    sol_q = None
    start_count = 0
    initial_delay_done = False
    # 使用 confirmed_*_action 来记录最终确认的动作
    confirmed_left_action = "open"
    confirmed_right_action = "open"
    last_left_hand = confirmed_left_action
    last_right_hand = confirmed_right_action

    ##
    tv_img_shape = (480, 640, 3)
    depth_shape = (480, 640)
    
    img_size = int(np.prod(tv_img_shape) * np.uint8().itemsize)

    # 彩色图像共享内存
    img_shm = shared_memory.SharedMemory(
        create=True, 
        size=img_size
    )
    depth_size = int(np.prod(depth_shape) * np.uint16().itemsize)
    # 深度图像共享内存
    depth_shm = shared_memory.SharedMemory(
        create=True,
        size=depth_size
    )
    
    
    client = ImageClient(
                tv_img_shape=tv_img_shape,
                tv_img_shm_name=img_shm.name,
                tv_depth_shm_name=depth_shm.name,
                image_show=False,
                Unit_Test=False
            )
    
    image_receive_thread = threading.Thread(target = client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()
    
        
    tv_wrapper = TeleVisionWrapper(INPUT_IMAGE_SIZE)
    arm_ik = Open_ArmIK(Visualization=False)
    
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = img_shm.buf)
    depth_array = np.ndarray(depth_shape, dtype = np.uint16, buffer = depth_shm.buf)


    # arm_ctrl.open_grippers()
    try:
    # 启动可视化线程
        user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
        if user_input.lower() == 'r':
            vis_thread = threading.Thread(target=visualization_thread)
            vis_thread.start()
            arm_ctrl.speed_gradual_max()
            left_action = "open"
            running = True
            while running:
                start_time = time.time()
                left_wrist, right_wrist, left_gesture, right_gesture,img= tv_wrapper.get_data(tv_img_array,depth_array)
                # print("get data time:",round(time.time() - start_time, 3))
                # 线程安全地更新图像
                with image_lock:
                    image_show = img
                
                # 使用新的确认函数来获取稳定的动作 ---
                confirmed_left_action = confirm_gesture("left", left_gesture)
                confirmed_right_action = confirm_gesture("right", right_gesture)

                ## 启动和结束
                if left_gesture == "start" and right_gesture == "start":
                    start_count = start_count + 1
                if left_gesture == "end" and right_gesture == "end":
                    running = False
                
                # # 检查是否达到启动条件
                if start_count < 2:
                    continue  # 尚未满足姿势条件，继续等待
                # 如果还没完成初始延迟，执行延迟操作
                if not initial_delay_done:
                    print("姿势已确认，将在3秒后开始控制...")
                    time.sleep(3)  # 延迟3秒，可以根据需要调整
                    initial_delay_done = True  # 标记延迟已完成
                    print("开始控制！")
                    if confirmed_left_action == "open": arm_ctrl.open_left_gripper()
                    if confirmed_right_action == "close": arm_ctrl.close_left_gripper()
                    if confirmed_right_action == "open": arm_ctrl.open_right_gripper()
                    if confirmed_right_action == "close": arm_ctrl.close_right_gripper()
                    last_left_hand = confirmed_left_action
                    last_right_hand = confirmed_right_action
                
                # --- 使用确认后的动作来控制夹爪 ---
                # 只有当确认的动作不为 None 且发生变化时，才执行
                if confirmed_left_action is not None and confirmed_left_action != last_left_hand:
                    if confirmed_left_action == "open":
                        arm_ctrl.open_left_gripper()
                    elif confirmed_left_action == "close":
                        arm_ctrl.close_left_gripper()
                    last_left_hand = confirmed_left_action
                    # print(f"左手动作确认: {confirmed_left_action}")

                if confirmed_right_action is not None and confirmed_right_action != last_right_hand:
                    if confirmed_right_action == "open":
                        arm_ctrl.open_right_gripper()
                    elif confirmed_right_action == "close":
                        arm_ctrl.close_right_gripper()
                    last_right_hand = confirmed_right_action
                    # print(f"右手动作确认: {confirmed_right_action}")


                # # get current state data.
                current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            

                # solve ik using motor data and wrist pose, then use ik results to control arms.
                time_ik_start = time.time()
                sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist, right_wrist, current_lr_arm_q, None)
                time_ik_end = time.time()
                # print(f"ik:\t{round(time_ik_end - time_ik_start, 3)}")
                arm_ctrl.ctrl_dual_arm(sol_q)


                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
                time.sleep(sleep_time)
                # print(f"main process sleep: {time_elapsed}")
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting program...")
    finally:
        arm_ctrl.close_grippers()
        arm_ctrl.ctrl_dual_arm_go_home()
        # # gripper_ctrl.close_grippers()
        img_shm.unlink()
        img_shm.close()
        depth_shm.unlink()
        depth_shm.close()
        # if 'vis_thread' in locals() and vis_thread.is_alive():
        #     vis_thread.join()  # 等待可视化线程结束
        print("Finally, exiting program...")
        exit(0)