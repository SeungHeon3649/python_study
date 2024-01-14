import cv2
import sys
import numpy as np

old_img = cv2.imread("mot_color70.jpg")
new_img = cv2.imread("mot_color83.jpg")
if (new_img is None) or (old_img is None):
    print("Image load failed")
    sys.exit()

old_gray = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT.create()
old_kp, old_desc = sift.detectAndCompute(old_gray, None)
new_kp, new_desc = sift.detectAndCompute(new_gray, None)

flann_matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)
knn_matcher = flann_matcher.knnMatch(old_desc, new_desc, 2)
T = 0.7
m_l = []
for old_desc, new_desc in knn_matcher:
    if (old_desc.distance / new_desc.distance) < T:
        m_l.append(old_desc)
mc_img = np.empty((old_img.shape[0], old_img.shape[1] + new_img.shape[1], 3), np.uint8)
cv2.drawMatches(old_img, old_kp, new_img, new_kp, m_l, mc_img,
                flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("end_img", mc_img)
cv2.waitKey()
cv2.destroyAllWindows()

