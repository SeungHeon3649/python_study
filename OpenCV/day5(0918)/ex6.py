# 저장된 사진과 실시간 비디오 특징점 매칭

# import cv2
# import sys
# import numpy as np

# src1 = cv2.imread("test1.jpg", cv2.IMREAD_COLOR)
# src2 = cv2.imread("test2.jpg", cv2.IMREAD_COLOR)
# dst1 = cv2.resize(src1, (300, 300))
# dst2 = cv2.resize(src2, (300, 300))
# test1 = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
# test2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)
# if (test1 is None) or (test2 is None):
#     print("Image load failed")
#     sys.exit()
# cv2.imshow("dst1", dst1)

# sift = cv2.SIFT.create()
# test1_kp, test1_desc = sift.detectAndCompute(test1, None)
# if test1_desc is None:
#     print("test1_desc 비었음")

# cap = cv2.VideoCapture(0)
# if cap.isOpened() == False:
#     print("카메라가 연결되지 않았습니다.")

# while True:
#     ret, frame = cap.read()
#     if not ret: break
#     #frame = cv2.flip(frame, 1)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     #test2_kp, test2_desc = sift.detectAndCompute(test2, None)
#     #cv2.imshow("gray", gray)
#     frame_kp, frame_desc = sift.detectAndCompute(gray, None)
#     if frame_desc is not None and len(frame_desc) > 1:
#         flann_matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)
#         knn_matcher = flann_matcher.knnMatch(test1_desc, frame_desc, 2)
#         good_matches = []
#         for m, n in knn_matcher:
#             if m.distance < 0.7 * n.distance:
#                 good_matches.append(m)
        
#         print(good_matches)
#         #combined_image = np.empty((frame.shape[0], frame.shape[1] + dst1.shape[1]), np.uint8)
#         frame_dst = cv2.drawMatches(dst1, test1_kp, frame, frame_kp, good_matches, None,
#                         flags = cv2.DrawMatchesFlags_DEFAULT)
#         #final_dst = cv2.cvtColor(frame_dst, cv2.COLOR_GRAY2BGR)
#         cv2.imshow("frame", frame)
#         cv2.imshow("combine", frame_dst)
#         key = cv2.waitKey(33)
#         if key == 27:
#             break
#     else:
#         print("frame_desc 비었음")

# cap.release()   
# cv2.destroyAllWindows()


# 실시간 영상에서 캡쳐해서 비교 

import cv2
import sys
import numpy as np

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
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            #print(good_matches)
            frame_dst = cv2.drawMatches(dst, test1_kp, frame, frame_kp, good_matches, None,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
            cv2.imshow("frame_dst", frame_dst)

    key = cv2.waitKey(33)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# while True:
#     ret, frame = cap.read()
#     if not ret: break
#     #frame = cv2.flip(frame, 1)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     #test2_kp, test2_desc = sift.detectAndCompute(test2, None)
#     #cv2.imshow("gray", gray)
#     frame_kp, frame_desc = sift.detectAndCompute(gray, None)
#     if frame_desc is not None and len(frame_desc) > 1:
#         flann_matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)
#         knn_matcher = flann_matcher.knnMatch(test1_desc, frame_desc, 2)
#         good_matches = []
#         for m, n in knn_matcher:
#             if m.distance < 0.7 * n.distance:
#                 good_matches.append(m)
        
#         print(good_matches)
#         #combined_image = np.empty((frame.shape[0], frame.shape[1] + dst1.shape[1]), np.uint8)
#         frame_dst = cv2.drawMatches(dst1, test1_kp, frame, frame_kp, good_matches, None,
#                         flags = cv2.DrawMatchesFlags_DEFAULT)
#         #final_dst = cv2.cvtColor(frame_dst, cv2.COLOR_GRAY2BGR)
#         cv2.imshow("frame", frame)
#         cv2.imshow("combine", frame_dst)
#         key = cv2.waitKey(33)
#         if key == 27:
#             break
#     else:
#         print("frame_desc 비었음")

# cap.release()   
# cv2.destroyAllWindows()