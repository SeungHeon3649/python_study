import cv2
import numpy as np
import sys

dst = None
cnt = False
show_def = False

def cut(event, x, y, flags, param):
    global ptx, pty, dst, cnt
    if event == cv2.EVENT_LBUTTONDOWN:
        ptx, pty = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        dst = frame[pty:y, ptx:x]
        #cv2.imshow("dst", dst)
        cnt = True

cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print("카메라가 연결되지 않았습니다.")

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", cut)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    cv2.imshow("frame", frame)
    if dst is not None:
        if cnt:
            global test1_kp, test1_desc, test1, sift
            test1 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT.create()
            test1_kp, test1_desc = sift.detectAndCompute(test1, None)
            if test1_desc is None:
                print("test1_desc 비었음")
            cnt = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_desc = sift.detectAndCompute(gray, None)
        if frame_desc is not None and len(frame_desc) > 1:
            flann_matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)
            knn_matcher = flann_matcher.knnMatch(test1_desc, frame_desc, 2)
            good_matches = []
            for m, n in knn_matcher:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
            if len(good_matches) > 10:
                src_pts = np.float32([test1_kp[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches])
                print(src_pts)
                print("한번")
                
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
                h, w = test1.shape
                box1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                box2 = cv2.perspectiveTransform(box1, M)
                frame_dst = cv2.polylines(frame, [np.int32(box2)], True, 255, 3, cv2.LINE_AA)
                final_dst = cv2.drawMatches(dst, test1_kp, frame_dst, frame_kp, good_matches, None,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
            cv2.imshow("final_dst", final_dst)

    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

