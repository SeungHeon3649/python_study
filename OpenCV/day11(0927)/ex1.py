import mediapipe as mp
import cv2

img = cv2.imread('lenna.png')

# 일종의 패키지를 불러온 것
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# FaceDetection클래스에서 기능 선택(모델 선택, 디텍션 신뢰도)
face_detection = mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5)
rec = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if not rec.detections:
    print('얼굴 검출 실패')
else:
    for i in rec.detections:
        # 눈, 코, 귀에 대한 위치를 뽑음, 입체적으로 
        mp_draw.draw_detection(img, i)
    cv2.imshow('f', img)
cv2.waitKey()
cv2.destroyAllWindows()