import cv2
import sys
import numpy as np

src = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
if src is None:
    print("Image load failed")
    sys.exit()
dst1 = cv2.Sobel(src, cv2.CV_32F, 1, 0, 3) # x 방향 미분(수직 마스크)
dst2 = cv2.Sobel(src, cv2.CV_32F, 0, 1, 3) # y 방향 미분(수평 마스크)
dst3 = cv2.convertScaleAbs(dst1) # 절댓값 및 uint8(CV_8U) 형변환
dst4 = cv2.convertScaleAbs(dst2)
eg_d = cv2.addWeighted(dst3, 0.5, dst4, 0.5, 0)
cv2.imshow('src', src)
cv2.imshow("dst3", dst3)
cv2.imshow("dst4", dst4)
cv2.imshow("eg_d", eg_d)
cv2.waitKey()
cv2.destroyAllWindows()