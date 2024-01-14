# 사각형 그리기

import cv2
import sys
import numpy as np

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()
dst = src[300:400, 300:500]
cv2.rectangle(src, (300, 300), (500, 500), (0, 0, 255), 3)
cv2.imshow("src", src)
i1 = cv2.resize(dst, (0, 0), fx = 5, fy = 5, interpolation = cv2.INTER_NEAREST)
i2 = cv2.resize(dst, (0, 0), fx = 5, fy = 5, interpolation = cv2.INTER_LINEAR)
i3 = cv2.resize(dst, (0, 0), fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
cv2.imshow("dst", dst)
cv2.imshow("i1", i1)
cv2.imshow("i2", i2)
cv2.imshow("i3", i3)
cv2.waitKey()
cv2.destroyAllWindows()