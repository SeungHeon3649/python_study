import mediapipe as mp
import cv2

hdata = mp.solutions.hands
# dw = mp.solutions.drawing_utils
# dw_st = mp.solutions.drawing_styles

in_img = cv2.imread('love.png', cv2.IMREAD_UNCHANGED)
# in_img = in_img[:, :178]
in_img = cv2.resize(in_img, (0, 0), fx = 0.1, fy = 0.1)
in_h, in_w = in_img.shape[:2]

hm = hdata.Hands(max_num_hands = 2, min_detection_confidence=0.5,
                  min_tracking_confidence=0.5)

cam = cv2.VideoCapture(0)

while True:
    r, f = cam.read()
    f = cv2.flip(f, 1)
    hmk = hm.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    ck = 0

    if hmk.multi_hand_landmarks:
        ck = 1
        for landmarks in hmk.multi_hand_landmarks:
            point_dot = landmarks.landmark[8]
            print(landmarks)
            print(len(landmarks))
            if ck:
                break
            # sx, sy = int(point_dot.x * f.shape[1] - in_w // 2), int(point_dot.y * f.shape[0] - in_h // 2) 
            # ex, ey = int(point_dot.x * f.shape[1] + in_w // 2), int(point_dot.y * f.shape[0] + in_h // 2) 
            
            # if sx > 0 and sy > 0 and ex < f.shape[1] and ey < f.shape[0]:
            #     # 4채널중 마지막이 png이미지의 알파값, 투명도
            #     ap = in_img[:, :, 3:] / 255
            #     f[sy : ey, sx : ex] = f[sy : ey, sx : ex] * (1 - ap) + in_img[:, :, :3] * ap
        break
    cv2.imshow('w', f)
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()