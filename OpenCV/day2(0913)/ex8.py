# 모폴로지연산

import cv2
import sys
import numpy as np

src = cv2.imread('morphology_test.png', cv2.IMREAD_GRAYSCALE)
if src is None:
    print("Image load failed")
    sys.exit()
#dst = src[src.shape[0]//3 * 2:, :]
#cv2.imshow("dst", dst)
mask = np.uint8([[0, 0, 1, 0, 0],
                 [0, 1, 1, 1, 0 ],
                 [1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 0],
                 [0, 0, 1, 0, 0]])
cv2.imshow("src", src)
# 팽창
dst1 = cv2.dilate(src, mask, iterations = 2)
cv2.imshow("dst1", dst1)
# 침식
dst2 = cv2.erode(src, mask, iterations = 2)
cv2.imshow("dst2", dst2)
# 닫기(팽창 -> 침식)
dst3 = cv2.erode(dst2, mask, iterations = 2)
cv2.imshow("dst3", dst3)
cv2.waitKey()
cv2.destroyAllWindows()

