import mediapipe as mp
import cv2

# png이미지를 불러올 때는 뒤 옵션을 써주자
# 그래야 4채널로 제대로 가져올 수 있음
in_img = cv2.imread('love.png', cv2.IMREAD_UNCHANGED)
#in_img = in_img[:, :178]
in_img = cv2.resize(in_img, (0, 0), fx = 0.1, fy = 0.1)
in_h, in_w = in_img.shape[:2]

# 일종의 패키지를 불러온 것
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5)

cam = cv2.VideoCapture(0)

while True:
    r, f = cam.read()

    rec = face_detection.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    if rec.detections:
        for det in rec.detections:
            point_dot = mp_face_detection.get_key_point(det, mp_face_detection.FaceKeyPoint.LEFT_EYE)
            sx, sy = int(point_dot.x * f.shape[1] - in_w // 2), int(point_dot.y * f.shape[0] - in_h // 2) 
            ex, ey = int(point_dot.x * f.shape[1] + in_w // 2), int(point_dot.y * f.shape[0] + in_h // 2) 
            
            if sx > 0 and sy > 0 and ex < f.shape[1] and ey < f.shape[0]:
                # 4채널중 마지막이 png이미지의 알파값, 투명도
                ap = in_img[:, :, 3:] / 255
                f[sy : ey, sx : ex] = f[sy : ey, sx : ex] * (1 - ap) + in_img[:, :, :3] * ap
            
    cv2.imshow('w', f)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()

# 교체기능, 투명도 조절, (붙이는거에서 반전, 원본, 상하, 좌우)