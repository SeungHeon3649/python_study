import cv2
import sys

src = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
if src is None:
    print("Image load failed")
    sys.exit()
cv2.imshow("src", src)
t, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("dst", dst)
cv2.waitKey()

