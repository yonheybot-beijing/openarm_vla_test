import cv2
import sys

# def list_ports():
#     """列出所有可用的摄像头端口"""
#     dev_port = 0
#     working_ports = []
#     while dev_port < 10:
#         camera = cv2.VideoCapture(dev_port)
#         if camera.isOpened():
#             ret, frame = camera.read()
#             if ret:
#                 working_ports.append(dev_port)
#             camera.release()
#         dev_port += 1
#     return working_ports
import cv2
def main():
    
    # 使用第一个可用的摄像头
    cap = cv2.VideoCapture(4)
    
   
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("无法读取摄像头画面")
            break
            
        cv2.imshow("Camera", image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()