import numpy as np


# T_to_unitree_left_wrist = np.array([[1, 0, 0, 0],
#                                     [0, 0, -1, 0],
#                                     [0, 1, 0, 0],
#                                     [0, 0, 0, 1]])

# T_to_unitree_right_wrist = np.array([[1, 0, 0, 0],
#                                      [0, 0, 1, 0],
#                                      [0, -1, 0, 0],
#                                      [0, 0, 0, 1]])

# T_to_unitree_hand = np.array([[0, 0, 1, 0],
#                               [-1,0, 0, 0], 
#                               [0, -1,0, 0],
#                               [0, 0, 0, 1]])

# T_robot_openxr = np.array([[0, 0, -1, 0],
#                            [-1, 0, 0, 0],
#                            [0, 1, 0, 0],
#                            [0, 0, 0, 1]])


## 初始姿态，双手抬起，小臂与大臂垂直
const_right_wrist_vuer_mat = np.array([[0, -1, 0, 0.14],
                                   [0, 0, -1, 0.3],
                                   [1, 0, 0, -0.23],
                                   [0, 0, 0, 1]])
const_left_wrist_vuer_mat = np.array([[0, -1, 0, -0.14],
                            [0, 0, -1, 0.3],
                            [1, 0, 0, -0.23],
                            [0, 0, 0, 1]])




