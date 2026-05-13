import numpy as np
import time
import os 
import sys
from flask import Flask, request, jsonify

# 导入机械臂控制模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from teleop.robot_control.robot_arm_inter import Open_arm_ArmController

# 初始化Flask应用
# app = Flask(__name__)

# 初始化机械臂控制器
arm_ctrl = Open_arm_ArmController()
arm_ctrl.speed_gradual_max()
arm_ctrl.ctrl_dual_arm_go_home()
time.sleep(2)
exit()

@app.route('/move_to_position', methods=['POST'])
def move_to_position():
    """
    接收sol_q位置信息并控制机械臂移动到指定位置
    请求体应为JSON格式: {"sol_q": [值1, 值2, ..., 值14]}
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        # 验证数据
        if 'sol_q' not in data:
            return jsonify({"status": "error", "message": "缺少sol_q参数"}), 400
        
        sol_q = data['sol_q']
        if not isinstance(sol_q, list) or len(sol_q) != 14:
            return jsonify({"status": "error", "message": "sol_q必须是包含14个元素的列表"}), 400
        
        # 转换为numpy数组
        sol_q_np = np.array(sol_q, dtype=np.float64)
        
        # 控制机械臂移动
        arm_ctrl.ctrl_dual_arm(sol_q_np)
        
        return jsonify({"status": "success", "message": "机械臂已移动到指定位置"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/go_home', methods=['POST'])
def home_position():
    """
    控制机械臂回到零位
    """
    try:
        # 控制机械臂归位
        arm_ctrl.ctrl_dual_arm_go_home()
        # 等待归位完成
        time.sleep(5)
        
        return jsonify({"status": "success", "message": "机械臂已回到零位"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health_check', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "healthy", "message": "机械臂控制API正常运行中"})

if __name__ == '__main__':
    # 启动服务，默认端口5000，允许外部访问
    app.run(host='0.0.0.0', port=5000, debug=True)
