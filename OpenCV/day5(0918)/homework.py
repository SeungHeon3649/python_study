import cv2
import sys
import numpy as np

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()

def draw(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        dst = src[iy:y, ix:x]
        old_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

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
        mc_img = np.empty((src.shape[0], src.shape[1] + dst.shape[1], 3), np.uint8)
        cv2.drawMatches(src, old_kp, dst, new_kp, m_l, mc_img,
                        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("end_img", mc_img)


cv2.imshow("src", src)
cv2.setMouseCallback("src", draw)
cv2.waitKey()
cv2.destroyAllWindows()
