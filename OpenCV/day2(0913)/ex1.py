# 사각형 그리기

import cv2
import sys

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()
dst = src.copy()
cv2.rectangle(dst, (100, 100), (200, 200), (0, 0, 255), 3)
cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()    