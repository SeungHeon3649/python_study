# 이진화

import cv2
import sys
import matplotlib.pyplot as plt

img = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)
if img is None:
    print('Image load failed')
    sys.exit()

# 선그리기
# cv2.line(img, (50,50), (50, 100), 0, 10)
# cv2.line(img, (50,50), (100, 50), 0, 10)
# cv2.line(img, (50,100), (100, 100), 0, 10)
# cv2.line(img, (100,50), (100, 100), 0, 10)

# 이진화
ret, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('cat', img)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()


cv2.cvtColor(cv2.COLOR_BGR2)