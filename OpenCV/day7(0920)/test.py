import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# SIFT 초기화
sift = cv2.SIFT_create()

# 테스트 이미지 로드 (축구공 또는 다른 객체)
test_image = cv2.imread('soccer_ball.jpg')  # 'soccer_ball.png'를 검출하려는 객체의 이미지 파일명으로 변경하세요.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # SIFT로 특징점과 기술자 계산
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
    kp_test, des_test = sift.detectAndCompute(gray_test_image, None)

    # 특징점 매칭
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_test, des_frame, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    detected_objects = []  # 검출된 객체의 경계 상자 좌표를 저장할 리스트

    if len(good_matches) > 10:
        src_pts = np.float32([kp_test[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 변환 행렬 계산
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 검출된 객체의 경계 상자 좌표 계산
        h, w = test_image.shape[:2]
        box = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        transformed_box = cv2.perspectiveTransform(box, M)

        # 검출된 객체의 경계 상자 좌표를 리스트에 추가
        detected_objects.append(transformed_box)

    # 검출된 모든 객체에 대한 경계 상자 그리기
    print(len(detected_objects))
    for obj in detected_objects:
        frame = cv2.polylines(frame, [np.int32(obj)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()

