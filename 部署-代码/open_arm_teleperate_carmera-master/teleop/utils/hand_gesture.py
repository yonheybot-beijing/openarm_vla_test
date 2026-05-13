import math
def vectorSize(p1, p2):
    """计算两点间的欧氏距离"""
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def vectorAngle(p1, p2, p3):
    """计算三点形成的角度"""
    b = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    c = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
    a = math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)
    if ((2 * b * c) > 1e-10):
        angle = math.acos(((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
    return math.degrees(angle)

def mkVector(p1, p2):
    """由坐标点构造向量"""
    return [(p1[0]-p2[0]), (p1[1]-p2[1])]

def vectorAngle2(v1, v2):
    """计算两个向量的夹角"""
    nor = 0.0
    a = 0.0
    b = 0.0
    for x, y in zip(v1, v2):
        nor += x*y  # 向量内积
        a += x**2
        b += y**2
    if a == 0 or b == 0:
        return None
    cosTheta = nor/math.sqrt(a*b)
    angle = math.acos(cosTheta)
    return math.degrees(angle)

# def fingersUp(landmarks):

#     """判断手指是否张开"""
#     fingers = []
#     # 大拇指
#     if vectorAngle(landmarks[0], landmarks[3], landmarks[4]) > 130:
#         fingers.append(1)
#     else:
#         fingers.append(0)

#     # 食指
#     if vectorSize(landmarks[0], landmarks[8]) > vectorSize(landmarks[0], landmarks[6]):
#         fingers.append(1)
#     else:
#         fingers.append(0)

#     # 中指
#     if vectorSize(landmarks[0], landmarks[12]) > vectorSize(landmarks[0], landmarks[10]):
#         fingers.append(1)
#     else:
#         fingers.append(0)

#     # 无名指
#     if vectorSize(landmarks[0], landmarks[16]) > vectorSize(landmarks[0], landmarks[14]):
#         fingers.append(1)
#     else:
#         fingers.append(0)

#     # 小拇指
#     if vectorSize(landmarks[0], landmarks[20]) > vectorSize(landmarks[0], landmarks[18]):
#         fingers.append(1)
#     else:
#         fingers.append(0)

#     return fingers

def fingersUp(landmarks):

    """判断手指是否张开"""
    fingers = []
    finger_names = ["大拇指", "食指", "中指", "无名指", "小拇指"]
    # 大拇指
    thumb_state = vectorAngle(landmarks[2], landmarks[3], landmarks[4]) > 150
    fingers.append(1 if thumb_state else 0)
    # print(f"{finger_names[0]}: {vectorAngle(landmarks[2], landmarks[3], landmarks[4])}")

    # 食指
    index_state = vectorAngle(landmarks[6], landmarks[7], landmarks[8]) > 90
    fingers.append(1 if index_state else 0)
    # print(f"{finger_names[1]}: {vectorAngle(landmarks[6], landmarks[7], landmarks[8])}")

    # 中指
    middle_state = vectorAngle(landmarks[12], landmarks[13], landmarks[14]) > 130
    fingers.append(1 if middle_state else 0)
    # print(f"{finger_names[2]}: {vectorAngle(landmarks[12], landmarks[13], landmarks[14])}")

    # 无名指
    ring_state = vectorAngle(landmarks[17], landmarks[18], landmarks[19]) > 130
    fingers.append(1 if ring_state else 0)
    # print(f"{finger_names[3]}: {vectorAngle(landmarks[17], landmarks[18], landmarks[19])}")

    # 小拇指
    pinky_state = vectorAngle(landmarks[22], landmarks[23], landmarks[24]) > 130
    fingers.append(1 if pinky_state else 0)
    # print(f"{finger_names[4]}: {vectorAngle(landmarks[22], landmarks[23], landmarks[24])}")
    
    return fingers

def staticGestureRec(landmark):
    try:
        """静态手势识别"""

        fingers = fingersUp(landmark)


        if (fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0):

            return "dianzan"
        if (fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0):

            return "woquan"
        if (fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1):
            return "ok"

    except Exception as e:
        print(e)
        return "None"
    return "None"
