# 사각형 그리기

import cv2
import sys
import numpy as np

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()
re_src = cv2.resize(src, (0, 0), fx = 0.4, fy = 0.4)
dst = cv2.cvtColor(re_src, cv2.COLOR_BGR2GRAY)
cv2.putText(dst, "old", (100,200), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))
cv2.imshow("dst", dst)

g1 = cv2.GaussianBlur(dst, (3, 3), 0, 0)
g2 = cv2.GaussianBlur(dst, (5, 5), 0, 0)
g3 = cv2.GaussianBlur(dst, (9, 9), 0, 0)
g_all = np.hstack((g1, g2, g3))
cv2.imshow("g_all", g_all)

mask = np.array([[-1, 0, 0],
                 [0, 0, 0],
                 [0, 0, 1]])

g_img16 = np.int16(dst)
filter_dst = cv2.filter2D(g_img16, -1, mask) + 128
filter_dst = np.uint8(filter_dst)

tr1 = np.uint8(np.clip(cv2.filter2D(g_img16, -1, mask) + 128, 0, 255))
tr2 = np.uint8(cv2.filter2D(g_img16, -1, mask) + 128)
tr3 = cv2.filter2D(dst, -1, mask)

cv2.imshow("tr1", tr1)
cv2.imshow("tr2", tr2)
cv2.imshow("tr3", tr3)
cv2.waitKey()
cv2.destroyAllWindows()